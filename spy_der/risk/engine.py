"""Risk engine with absolute veto authority (spec sections 28, 29, 42).

Two properties this module is built around:

**The veto cannot be bypassed.** Every check appends to one list of reasons and
the verdict is ``not reasons``. There is no early return that skips remaining
checks, no override parameter, and no path that produces ``approved=True``
without having run every check.

**It fails closed.** Missing data, an unreadable forecast, or an unexpected
exception all produce a veto rather than an approval. Spec 42: when uncertain,
stop trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from spy_der.domain.enums import KillSwitchReason, MarketState
from spy_der.domain.execution import BrokerState, Position, RiskDecision
from spy_der.domain.forecast import ForecastBundle
from spy_der.domain.market import MarketSnapshot
from spy_der.domain.strategy import StrategyCandidate
from spy_der.risk.limits import KillSwitch, RiskConfig, SessionState
from spy_der.risk.sizing import SizingResult, size_position


@dataclass
class PortfolioState:
    """Open positions and their aggregate exposure."""

    positions: list[Position] = field(default_factory=list)

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_open]

    @property
    def open_count(self) -> int:
        return len(self.open_positions)

    @property
    def total_open_risk(self) -> float:
        return sum(p.total_max_loss for p in self.open_positions)

    def risk_for_expiration(self, expiry: datetime) -> float:
        return sum(
            p.total_max_loss
            for p in self.open_positions
            if any(leg.expiry == expiry for leg in p.legs)
        )

    def zero_dte_risk(self, now: datetime) -> float:
        return sum(
            p.total_max_loss
            for p in self.open_positions
            if any(leg.expiry.date() == now.date() for leg in p.legs)
        )

    def aggregate_greek(self, name: str) -> float:
        from spy_der.strategies.metrics import structure_greeks

        total = 0.0
        for position in self.open_positions:
            marked_at = position.marked_at or position.opened_at
            greeks = structure_greeks(position.strategy, position.current_price or 0.0, marked_at)
            total += getattr(greeks, name, 0.0) * position.contracts
        return total

    def add(self, position: Position) -> None:
        self.positions.append(position)

    def close(self, position_id: str, closed: Position) -> None:
        self.positions = [closed if p.position_id == position_id else p for p in self.positions]


@dataclass
class RiskEngine:
    """Approves, sizes, or vetoes a candidate."""

    config: RiskConfig = field(default_factory=RiskConfig)
    kill_switch: KillSwitch = field(default_factory=KillSwitch)
    session: SessionState = field(default_factory=SessionState)

    def evaluate(
        self,
        candidate: StrategyCandidate | None,
        *,
        snapshot: MarketSnapshot,
        forecast: ForecastBundle,
        portfolio: PortfolioState,
        broker: BrokerState,
    ) -> tuple[RiskDecision, SizingResult | None]:
        """Return the verdict and, when approved, the sizing that produced it."""
        self.session.roll_to(snapshot.timestamp.date(), broker.equity)
        self._update_kill_switches(snapshot=snapshot, forecast=forecast, broker=broker)

        reasons: list[str] = list(self.kill_switch.reasons)
        reasons.extend(self._session_reasons(snapshot=snapshot, broker=broker))
        reasons.extend(self._environment_reasons(snapshot=snapshot, forecast=forecast))
        reasons.extend(self._broker_reasons(broker))

        if candidate is None:
            reasons.append("no candidate submitted")
            return RiskDecision(approved=False, veto_reasons=tuple(reasons)), None

        reasons.extend(self._candidate_reasons(candidate, forecast=forecast))
        reasons.extend(self._portfolio_reasons(candidate, portfolio=portfolio, broker=broker))

        available = self._available_risk(portfolio, broker)
        sizing = size_position(
            candidate,
            equity=broker.equity,
            confidence=forecast.confidence,
            stability=forecast.state_stability,
            calibration_quality=max(forecast.calibration_score, 1e-6),
            config=self.config,
            available_risk=available,
        )
        if not sizing.is_tradeable:
            reasons.append(f"position sizing yields zero contracts ({sizing.binding_constraint})")

        approved = not reasons
        return (
            RiskDecision(
                approved=approved,
                veto_reasons=tuple(reasons),
                max_contracts=sizing.contracts if approved else 0,
                approved_risk=sizing.total_risk if approved else 0.0,
                required_price_limit=self._price_limit(candidate) if approved else 0.0,
            ),
            sizing,
        )

    # -- kill switches -----------------------------------------------------

    def _update_kill_switches(
        self, *, snapshot: MarketSnapshot, forecast: ForecastBundle, broker: BrokerState
    ) -> None:
        cfg = self.config

        if snapshot.data_quality_score < cfg.trade.min_data_quality:
            self.kill_switch.trip(
                KillSwitchReason.DATA_QUALITY,
                f"DQ {snapshot.data_quality_score:.2f} below {cfg.trade.min_data_quality:.2f}",
            )
        if not broker.is_healthy:
            self.kill_switch.trip(
                KillSwitchReason.BROKER_STATE, "; ".join(broker.discrepancies) or "disconnected"
            )
        if forecast.model_disagreement > cfg.trade.max_model_disagreement:
            self.kill_switch.trip(
                KillSwitchReason.MODEL_DISAGREEMENT,
                f"dispersion {forecast.model_disagreement:.2f}",
            )
        if forecast.calibration_score < cfg.min_calibration_quality:
            self.kill_switch.trip(
                KillSwitchReason.CALIBRATION, f"quality {forecast.calibration_score:.2f}"
            )

        loss_limit = min(
            cfg.session.daily_loss_limit, broker.equity * cfg.session.daily_loss_fraction
        )
        if self.session.realized_pnl <= -loss_limit:
            self.kill_switch.trip(
                KillSwitchReason.DAILY_LOSS, f"realized {self.session.realized_pnl:.2f}"
            )
        if self.session.consecutive_losses >= cfg.session.max_consecutive_losses:
            self.kill_switch.trip(
                KillSwitchReason.CONSECUTIVE_LOSSES, f"{self.session.consecutive_losses} in a row"
            )
        drawdown = self.session.drawdown_fraction(broker.equity)
        if drawdown >= cfg.session.max_drawdown_fraction:
            self.kill_switch.trip(KillSwitchReason.DRAWDOWN, f"{drawdown:.1%} from peak")
        if self.session.slippage_ratio > cfg.max_slippage_ratio:
            self.kill_switch.trip(
                KillSwitchReason.SLIPPAGE, f"realized/expected {self.session.slippage_ratio:.2f}"
            )

    # -- individual check groups -------------------------------------------

    def _session_reasons(
        self, *, snapshot: MarketSnapshot, broker: BrokerState
    ) -> list[str]:
        cfg = self.config.session
        out: list[str] = []
        context = snapshot.context

        if context.minutes_to_close < cfg.min_minutes_to_close:
            out.append(
                f"only {context.minutes_to_close:.0f} minutes to close, "
                f"minimum is {cfg.min_minutes_to_close:.0f}"
            )
        if cfg.block_on_scheduled_event and context.has_blocking_event:
            out.append(f"scheduled event lockout: {', '.join(context.scheduled_events)}")
        if self.session.trades_today >= cfg.max_trades_per_session:
            out.append(f"session trade limit of {cfg.max_trades_per_session} reached")
        return out

    def _environment_reasons(
        self, *, snapshot: MarketSnapshot, forecast: ForecastBundle
    ) -> list[str]:
        cfg = self.config.trade
        out: list[str] = []

        if snapshot.data_quality_score < cfg.min_data_quality:
            out.append(
                f"data quality {snapshot.data_quality_score:.2f} below "
                f"{cfg.min_data_quality:.2f}"
            )
        if forecast.confidence < cfg.min_confidence:
            out.append(
                f"confidence {forecast.confidence:.2f} below {cfg.min_confidence:.2f}"
            )
        if forecast.model_disagreement > cfg.max_model_disagreement:
            out.append(f"model disagreement {forecast.model_disagreement:.2f} too high")

        # Structural veto: the mechanics model can refuse a statistically
        # attractive trade in a state where no structure is appropriate.
        for state in (MarketState.TRANSITION, MarketState.UNSTABLE):
            probability = forecast.state_probability(state)
            if probability >= 0.35:
                out.append(
                    f"structural veto: {state.value} probability {probability:.2f}"
                )
        return out

    @staticmethod
    def _broker_reasons(broker: BrokerState) -> list[str]:
        out: list[str] = []
        if not broker.connected:
            out.append("broker is disconnected")
        if not broker.reconciled:
            out.append("broker positions are not reconciled")
        if broker.discrepancies:
            out.append(f"broker discrepancies: {', '.join(broker.discrepancies)}")
        if broker.equity <= 0.0:
            out.append("account equity is non-positive")
        return out

    def _candidate_reasons(
        self, candidate: StrategyCandidate, *, forecast: ForecastBundle
    ) -> list[str]:
        cfg = self.config.trade
        out: list[str] = []

        if candidate.rejection_reasons:
            out.extend(candidate.rejection_reasons)
        if not math.isfinite(candidate.max_loss) or candidate.max_loss <= 0.0:
            out.append("candidate maximum loss is not a positive finite number")
            return out  # every downstream ratio would be meaningless

        if candidate.max_loss > cfg.max_dollar_risk:
            out.append(
                f"per-contract risk {candidate.max_loss:.0f} exceeds "
                f"{cfg.max_dollar_risk:.0f}"
            )
        if candidate.expected_return_on_risk < cfg.min_return_on_risk:
            out.append(
                f"return on risk {candidate.expected_return_on_risk:.3f} below "
                f"{cfg.min_return_on_risk:.3f}"
            )
        if candidate.liquidity.score < cfg.min_liquidity_score:
            out.append(f"liquidity score {candidate.liquidity.score:.2f} too low")
        if candidate.liquidity.worst_spread_ratio > cfg.max_spread_ratio:
            out.append(
                f"worst leg spread ratio {candidate.liquidity.worst_spread_ratio:.3f} "
                f"exceeds {cfg.max_spread_ratio:.3f}"
            )
        if candidate.expected_slippage > cfg.max_expected_slippage_fraction * candidate.max_loss:
            out.append(
                f"expected slippage {candidate.expected_slippage:.2f} exceeds "
                f"{cfg.max_expected_slippage_fraction:.0%} of maximum loss"
            )
        if candidate.strategy.leg_count > cfg.max_legs:
            out.append(f"{candidate.strategy.leg_count} legs exceeds limit of {cfg.max_legs}")
        if candidate.assignment_risk > cfg.max_assignment_risk:
            out.append(
                f"assignment risk {candidate.assignment_risk:.2f} exceeds "
                f"{cfg.max_assignment_risk:.2f}"
            )

        width = self._max_strike_width(candidate)
        if width > cfg.max_spread_width:
            out.append(f"strike width {width:.1f} exceeds {cfg.max_spread_width:.1f}")

        entry_debit = max(0.0, candidate.strategy.net_entry_cost)
        if entry_debit > cfg.max_debit_per_contract:
            out.append(
                f"net debit {entry_debit:.0f} exceeds {cfg.max_debit_per_contract:.0f}"
            )

        greeks = candidate.greeks
        for name, value, limit in (
            ("gamma", abs(greeks.gamma), cfg.max_gamma),
            ("vega", abs(greeks.vega), cfg.max_vega),
            ("theta", abs(greeks.theta), cfg.max_theta),
        ):
            if value > limit:
                out.append(f"{name} exposure {value:.0f} exceeds {limit:.0f}")

        if not self._has_complete_exit_plan(candidate):
            out.append("candidate does not carry a complete exit plan")
        return out

    @staticmethod
    def _has_complete_exit_plan(candidate: StrategyCandidate) -> bool:
        plan = candidate.exit_plan
        return (
            plan.profit_target_fraction > 0.0
            and plan.stop_loss_fraction > 0.0
            and plan.max_holding_minutes > 0.0
            and plan.close_before_expiry_minutes > 0.0
        )

    @staticmethod
    def _max_strike_width(candidate: StrategyCandidate) -> float:
        strikes = candidate.strategy.strikes
        return strikes[-1] - strikes[0] if len(strikes) > 1 else 0.0

    def _portfolio_reasons(
        self,
        candidate: StrategyCandidate,
        *,
        portfolio: PortfolioState,
        broker: BrokerState,
    ) -> list[str]:
        cfg = self.config.portfolio
        out: list[str] = []

        if portfolio.open_count >= cfg.max_open_strategies:
            out.append(
                f"{portfolio.open_count} open strategies, limit is {cfg.max_open_strategies}"
            )

        allowed_total = min(
            cfg.max_total_open_risk, broker.equity * cfg.max_total_risk_fraction
        )
        if portfolio.total_open_risk + candidate.max_loss > allowed_total:
            out.append(
                f"open risk {portfolio.total_open_risk:.0f} plus new "
                f"{candidate.max_loss:.0f} exceeds {allowed_total:.0f}"
            )

        expiry = candidate.strategy.near_expiry
        if portfolio.risk_for_expiration(expiry) + candidate.max_loss > cfg.max_same_expiration_risk:
            out.append(f"same-expiration risk limit exceeded for {expiry:%Y-%m-%d}")

        if candidate.capital_usage > broker.buying_power * cfg.max_capital_fraction:
            out.append("capital usage exceeds the permitted fraction of buying power")
        return out

    def _available_risk(self, portfolio: PortfolioState, broker: BrokerState) -> float:
        cfg = self.config.portfolio
        allowed = min(cfg.max_total_open_risk, broker.equity * cfg.max_total_risk_fraction)
        return max(0.0, allowed - portfolio.total_open_risk)

    def risk_budget(
        self,
        *,
        broker: BrokerState,
        portfolio: PortfolioState,
        confidence: float,
        stability: float,
        calibration_quality: float,
        liquidity: float = 0.8,
    ) -> float:
        """Largest per-contract loss that could be sized to at least one contract.

        Handed to the optimizer *before* ranking. Without it the optimizer
        happily selects the highest-utility structure, and only afterwards does
        sizing discover it cannot afford a single contract — producing a
        no-trade even when an affordable candidate with real edge existed. The
        budget is an estimate, so it is used only to filter; the authoritative
        sizing still runs in :meth:`evaluate`.

        ``liquidity`` is a placeholder for the not-yet-known per-candidate
        score, so the estimate is deliberately generous: filtering too
        aggressively here would discard candidates that sizing would in fact
        have accepted.
        """
        cfg = self.config
        base = max(0.0, broker.equity) * cfg.base_risk_fraction
        quality = (
            max(0.0, min(1.0, confidence))
            * max(0.0, min(1.0, liquidity))
            * max(0.0, min(1.0, stability))
            * max(0.0, min(1.0, calibration_quality))
        )
        return min(
            base * quality,
            broker.equity * cfg.trade.max_account_fraction_at_risk,
            cfg.trade.max_dollar_risk,
            self._available_risk(portfolio, broker),
        )

    @staticmethod
    def _price_limit(candidate: StrategyCandidate) -> float:
        """Worst net price the order may be filled at.

        For a debit the limit sits above the estimate; for a credit it sits
        below. Either way the tolerance is one tick per leg, so a wider
        structure is allowed proportionally more room without being open-ended.
        """
        tolerance = 0.01 * candidate.strategy.leg_count
        entry = candidate.entry_price
        return entry + tolerance if entry >= 0.0 else entry - tolerance
