"""End-to-end decision pipeline (spec section 4 execution sequence).

Runs the full sequence for one snapshot: validate, analyse, classify, forecast,
generate, evaluate, veto, decide, execute, monitor, persist.

Two invariants hold at this level.

**Point in time.** The only market input is the snapshot passed in. Nothing here
reaches for data at another timestamp, which is what lets the backtest reuse
this exact code path and still satisfy ``Decision_t = f(Data_{<=t})``.

**Fail closed.** Any stage that cannot complete produces a no-trade decision
with a recorded reason, never a partially evaluated trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from uuid import uuid4

from spy_der.data.persistence.audit import AuditStore, ComponentVersions, DecisionRecord
from spy_der.data.validators.quality import DataQualityConfig, DataQualityReport, assess
from spy_der.domain.enums import ExitReason
from spy_der.domain.execution import Fill, MultiLegOrder, Position, RiskDecision
from spy_der.domain.forecast import ForecastBundle
from spy_der.domain.market import MarketSnapshot
from spy_der.domain.strategy import StrategyCandidate
from spy_der.execution.fills import CostModel
from spy_der.execution.manager import ExitSignal, PositionManager
from spy_der.execution.orders import build_open_order
from spy_der.execution.paper_broker import PaperBroker
from spy_der.execution.reconciliation import apply_to_broker_state, reconcile
from spy_der.features.builder import AnalyticsBundle, FeatureContext, build_features
from spy_der.models.forecast_engine import ForecastEngine
from spy_der.optimizer.engine import OptimizationResult, StrategyOptimizer
from spy_der.risk.engine import PortfolioState, RiskEngine
from spy_der.risk.sizing import SizingResult

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    costs: CostModel = field(default_factory=CostModel)
    persist: bool = True
    max_poll_attempts: int = 4
    versions: ComponentVersions = field(default_factory=ComponentVersions)


@dataclass(frozen=True)
class PipelineResult:
    """Everything one pass produced, for the API, dashboard, and audit."""

    timestamp: datetime
    snapshot: MarketSnapshot
    quality: DataQualityReport
    analytics: AnalyticsBundle | None
    forecast: ForecastBundle | None
    optimization: OptimizationResult | None
    candidate: StrategyCandidate | None
    risk: RiskDecision
    sizing: SizingResult | None
    order: MultiLegOrder | None
    fill: Fill | None
    position: Position | None
    exits: tuple[ExitSignal, ...]
    decision: str
    reasons: tuple[str, ...]

    @property
    def traded(self) -> bool:
        return self.fill is not None

    @property
    def is_no_trade(self) -> bool:
        return self.decision == "no_trade"


@dataclass
class DecisionPipeline:
    """Orchestrates one decision cycle."""

    forecast_engine: ForecastEngine = field(default_factory=ForecastEngine)
    optimizer: StrategyOptimizer = field(default_factory=StrategyOptimizer)
    risk_engine: RiskEngine = field(default_factory=RiskEngine)
    broker: PaperBroker = field(default_factory=PaperBroker)
    manager: PositionManager = field(default_factory=PositionManager)
    portfolio: PortfolioState = field(default_factory=PortfolioState)
    features: FeatureContext = field(default_factory=FeatureContext)
    config: PipelineConfig = field(default_factory=PipelineConfig)
    audit: AuditStore | None = None

    def reset(self) -> None:
        """Clear all temporal state; used between walk-forward folds."""
        self.forecast_engine.reset()
        self.features = FeatureContext()
        self.portfolio = PortfolioState()

    def step(self, snapshot: MarketSnapshot) -> PipelineResult:
        """Run one full decision cycle against ``snapshot``."""
        quality = assess(snapshot, self.config.data_quality)
        snapshot = replace(snapshot, data_quality_score=quality.score)

        # Manage what is already open before considering anything new. An exit
        # that frees risk budget must land before the portfolio limits are read.
        exits = self._manage_open_positions(snapshot)

        if not quality.is_tradeable:
            return self._no_trade(
                snapshot,
                quality,
                exits,
                reasons=(f"data quality lockout: {quality.summary()}",),
            )

        analytics = build_features(snapshot, self.features)
        forecast, paths = self.forecast_engine.forecast(analytics)

        # Establish the affordable risk envelope before ranking, so the
        # optimizer never selects a structure sizing would have to refuse.
        broker_state = self._broker_state()
        risk_cap = self.risk_engine.risk_budget(
            broker=broker_state,
            portfolio=self.portfolio,
            confidence=forecast.confidence,
            stability=forecast.state_stability,
            calibration_quality=max(forecast.calibration_score, 1e-6),
        )

        optimization = self.optimizer.optimize(
            snapshot=snapshot,
            features=analytics.features,
            forecast=forecast,
            paths=paths,
            max_risk_per_contract=risk_cap,
        )

        candidate = optimization.best if optimization.trade_selected else None

        risk, sizing = self.risk_engine.evaluate(
            candidate,
            snapshot=snapshot,
            forecast=forecast,
            portfolio=self.portfolio,
            broker=broker_state,
        )

        if candidate is None or not risk.approved:
            reasons = risk.veto_reasons
            if candidate is None and optimization.best is not None:
                shortfall = (
                    optimization.no_trade_utility
                    + optimization.no_trade_margin
                    - optimization.best.utility
                )
                reasons = (
                    f"best candidate utility {optimization.best.utility:.2f} short of the "
                    f"no-trade bar by {shortfall:.2f}",
                    *reasons,
                )
            return self._no_trade(
                snapshot,
                quality,
                exits,
                reasons=reasons,
                analytics=analytics,
                forecast=forecast,
                optimization=optimization,
                risk=risk,
                sizing=sizing,
            )

        order, fill, position = self._execute(candidate, risk, snapshot)
        decision = candidate.strategy_id if fill else "no_trade"
        reasons = () if fill else ("order did not fill",)

        result = PipelineResult(
            timestamp=snapshot.timestamp,
            snapshot=snapshot,
            quality=quality,
            analytics=analytics,
            forecast=forecast,
            optimization=optimization,
            candidate=candidate,
            risk=risk,
            sizing=sizing,
            order=order,
            fill=fill,
            position=position,
            exits=exits,
            decision=decision,
            reasons=reasons,
        )
        self._persist(result)
        return result

    # -- stages ------------------------------------------------------------

    def _manage_open_positions(self, snapshot: MarketSnapshot) -> tuple[ExitSignal, ...]:
        signals: list[ExitSignal] = []
        kill_active = self.risk_engine.kill_switch.is_tripped

        for position in list(self.portfolio.open_positions):
            marked = self.manager.mark(position, snapshot)
            self.portfolio.close(position.position_id, marked)

            signal = self.manager.check_exit(
                marked, snapshot, kill_switch_active=kill_active
            )
            if signal is None:
                continue

            closed = self._close_position(marked, snapshot, signal)
            if closed is not None:
                signals.append(signal)
        return tuple(signals)

    def _close_position(
        self, position: Position, snapshot: MarketSnapshot, signal: ExitSignal
    ) -> Position | None:
        order = self.manager.build_exit_order(position, snapshot)
        if order is None:
            return None

        self.broker.submit(order, snapshot)
        for _ in range(self.config.max_poll_attempts):
            order, fill = self.broker.poll(order.order_id, snapshot)
            if fill is not None:
                break
            if order.is_terminal:
                break
        else:  # pragma: no cover - loop always terminates via break above
            fill = None

        # A close that cannot be filled leaves the position open on purpose:
        # marking it closed without an offsetting fill would desynchronize the
        # book from the broker and defeat reconciliation.
        if fill is None:
            return None

        exit_price = -order.limit_price
        pnl = self.manager.realized_pnl(position, exit_price) - fill.commission
        closed = position.closed(at=snapshot.timestamp, pnl=pnl, reason=signal.reason)

        self.portfolio.close(position.position_id, closed)
        self.broker.settle_position(closed, pnl=pnl)
        self.risk_engine.session.record_trade(pnl, self.broker.equity)
        self._persist_trade(closed, exit_price)
        return closed

    def _execute(
        self, candidate: StrategyCandidate, risk: RiskDecision, snapshot: MarketSnapshot
    ) -> tuple[MultiLegOrder | None, Fill | None, Position | None]:
        order = build_open_order(candidate, risk, now=snapshot.timestamp)
        order = self.broker.submit(order, snapshot)

        fill: Fill | None = None
        for _ in range(self.config.max_poll_attempts):
            order, fill = self.broker.poll(order.order_id, snapshot)
            if fill is not None or order.is_terminal:
                break

        if fill is None:
            if not order.is_terminal:
                order = self.broker.cancel(order.order_id, now=snapshot.timestamp)
            return order, None, None

        position = Position(
            position_id=f"P-{uuid4().hex[:12]}",
            opened_at=snapshot.timestamp,
            family=candidate.family,
            strategy=candidate.strategy,
            contracts=fill.contracts,
            entry_price=fill.fill_price,
            max_loss_per_contract=candidate.max_loss,
            max_profit_per_contract=candidate.max_profit,
            exit_plan=candidate.exit_plan,
            entry_commission=fill.commission,
            current_price=fill.fill_price,
            marked_at=snapshot.timestamp,
        )
        self.portfolio.add(position)
        self.broker.register_position(position)
        self.risk_engine.session.record_slippage(
            candidate.expected_slippage * fill.contracts, fill.slippage_vs_mid
        )
        return order, fill, position

    def _broker_state(self):
        state = self.broker.state()
        report = reconcile(tuple(self.portfolio.open_positions), self.broker.positions())
        return apply_to_broker_state(state, report)

    def _no_trade(
        self,
        snapshot: MarketSnapshot,
        quality: DataQualityReport,
        exits: tuple[ExitSignal, ...],
        *,
        reasons: tuple[str, ...],
        analytics: AnalyticsBundle | None = None,
        forecast: ForecastBundle | None = None,
        optimization: OptimizationResult | None = None,
        risk: RiskDecision | None = None,
        sizing: SizingResult | None = None,
    ) -> PipelineResult:
        result = PipelineResult(
            timestamp=snapshot.timestamp,
            snapshot=snapshot,
            quality=quality,
            analytics=analytics,
            forecast=forecast,
            optimization=optimization,
            candidate=None,
            risk=risk or RiskDecision(approved=False, veto_reasons=reasons),
            sizing=sizing,
            order=None,
            fill=None,
            position=None,
            exits=exits,
            decision="no_trade",
            reasons=reasons,
        )
        self._persist(result)
        return result

    # -- persistence -------------------------------------------------------

    def _persist(self, result: PipelineResult) -> None:
        if self.audit is None or not self.config.persist:
            return

        forecast = result.forecast
        selected = None
        if result.candidate is not None:
            selected = {
                "strategy_id": result.candidate.strategy_id,
                "family": result.candidate.family.value,
                "expected_value": result.candidate.expected_value,
                "utility": result.candidate.utility,
                "max_loss": result.candidate.max_loss,
                "max_profit": result.candidate.max_profit,
                "expected_return_on_risk": result.candidate.expected_return_on_risk,
                "cvar": result.candidate.cvar,
                "assignment_risk": result.candidate.assignment_risk,
                "legs": [
                    {
                        "expiry": leg.expiry.isoformat(),
                        "strike": leg.strike,
                        "right": leg.right.value,
                        "quantity": leg.quantity,
                        "entry_price": leg.entry_price,
                    }
                    for leg in result.candidate.legs
                ],
                "diagnostics": result.candidate.diagnostics,
            }

        record = DecisionRecord(
            timestamp=result.timestamp,
            spot=result.snapshot.spot,
            decision=result.decision,
            traded=result.traded,
            market_state=forecast.dominant_state.value if forecast else "unknown",
            state_probabilities=(
                {k.value: v for k, v in forecast.state_probabilities.items()} if forecast else {}
            ),
            confidence=forecast.confidence if forecast else 0.0,
            data_quality=result.quality.score,
            forecast=self._forecast_payload(forecast),
            candidates=[
                {
                    "strategy_id": c.strategy_id,
                    "family": c.family.value,
                    "utility": c.utility,
                    "expected_value": c.expected_value,
                    "max_loss": c.max_loss,
                    "expected_return_on_risk": c.expected_return_on_risk,
                }
                for c in (result.optimization.top(10) if result.optimization else ())
            ],
            rejected_candidates=[
                {"strategy_id": sid, "reasons": list(reasons)}
                for sid, reasons in (result.optimization.rejected if result.optimization else ())
            ],
            registry_rejections=(
                result.optimization.registry_rejections if result.optimization else {}
            ),
            risk_decision={
                "approved": result.risk.approved,
                "max_contracts": result.risk.max_contracts,
                "approved_risk": result.risk.approved_risk,
                "required_price_limit": result.risk.required_price_limit,
            },
            veto_reasons=list(result.risk.veto_reasons) + list(result.reasons),
            selected=selected,
            sizing=(
                {
                    "contracts": result.sizing.contracts,
                    "base_risk": result.sizing.base_risk,
                    "adjusted_risk": result.sizing.adjusted_risk,
                    "binding_constraint": result.sizing.binding_constraint,
                }
                if result.sizing
                else None
            ),
            order=(
                {
                    "order_id": result.order.order_id,
                    "status": result.order.status.value,
                    "limit_price": result.order.limit_price,
                    "contracts": result.order.contracts,
                    "reprice_attempts": result.order.reprice_attempts,
                }
                if result.order
                else None
            ),
            fill=(
                {
                    "fill_price": result.fill.fill_price,
                    "contracts": result.fill.contracts,
                    "commission": result.fill.commission,
                    "slippage_vs_mid": result.fill.slippage_vs_mid,
                }
                if result.fill
                else None
            ),
            versions=self.config.versions,
            diagnostics={
                "exits": [
                    {"reason": s.reason.value, "detail": s.detail, "pnl": s.unrealized_pnl}
                    for s in result.exits
                ],
                "kill_switches": list(self.risk_engine.kill_switch.reasons),
                "open_positions": self.portfolio.open_count,
                "eligible_families": sorted(
                    f.value for f in (result.optimization.eligible_families if result.optimization else ())
                ),
            },
        )
        self.audit.record_decision(record)

    @staticmethod
    def _forecast_payload(forecast: ForecastBundle | None) -> dict[str, object]:
        if forecast is None:
            return {}
        return {
            "confidence": forecast.confidence,
            "model_disagreement": forecast.model_disagreement,
            "state_entropy": forecast.state_entropy,
            "state_stability": forecast.state_stability,
            "dealer_agreement": forecast.dealer_agreement,
            "calibration_score": forecast.calibration_score,
            "expected_state_duration_minutes": forecast.expected_state_duration_minutes,
            "horizons": {
                name: {
                    "minutes": h.minutes,
                    "expected_price": h.expected_price,
                    "median_price": h.median_price,
                    "price_std": h.price_std,
                    "quantiles": {str(q): v for q, v in h.price_quantiles.items()},
                    "expected_mfe": h.expected_mfe,
                    "expected_mae": h.expected_mae,
                }
                for name, h in forecast.horizons.items()
            },
            "pin_probabilities": {str(k): v for k, v in forecast.pin_probabilities.items()},
            "range_probabilities": {
                f"{lo}-{hi}": v for (lo, hi), v in forecast.range_probabilities.items()
            },
            "diagnostics": forecast.diagnostics,
        }

    def _persist_trade(self, position: Position, exit_price: float) -> None:
        if self.audit is None or not self.config.persist:
            return
        self.audit.record_trade(
            {
                "position_id": position.position_id,
                "opened_at": position.opened_at.isoformat(),
                "closed_at": position.closed_at.isoformat() if position.closed_at else None,
                "family": position.family.value,
                "contracts": position.contracts,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "realized_pnl": position.realized_pnl,
                "exit_reason": position.exit_reason.value if position.exit_reason else None,
                "max_loss": position.total_max_loss,
                "legs": [
                    {
                        "expiry": leg.expiry.isoformat(),
                        "strike": leg.strike,
                        "right": leg.right.value,
                        "quantity": leg.quantity,
                    }
                    for leg in position.legs
                ],
            }
        )


def force_close_all(
    pipeline: DecisionPipeline, snapshot: MarketSnapshot, reason: ExitReason
) -> tuple[Position, ...]:
    """Close every open position, used at session end and by the kill switch."""
    closed: list[Position] = []
    for position in list(pipeline.portfolio.open_positions):
        marked = pipeline.manager.mark(position, snapshot)
        signal = ExitSignal(
            position_id=marked.position_id,
            reason=reason,
            detail=f"forced close: {reason.value}",
            mark_price=marked.current_price,
            unrealized_pnl=marked.unrealized_pnl,
        )
        result = pipeline._close_position(marked, snapshot, signal)
        if result is not None:
            closed.append(result)
    return tuple(closed)
