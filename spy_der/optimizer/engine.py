"""Candidate search, evaluation, and ranking (spec sections 24-27).

The full path from a forecast to a decision:

1. filter families by market state (spec 24);
2. generate parameterizations within each family (spec 25);
3. admit only structures that pass the defined-risk proof (spec 44);
4. value each on shared simulated paths, net of realistic costs (spec 26);
5. score expected utility (spec 3);
6. compare the best against explicit no-trade (spec 27).

Step 6 is the one that matters most. Spec 45: the system's most important
competency is rejecting trades whose apparent edge does not survive costs,
uncertainty, and liquidity — so no-trade wins by default and must be beaten by
a margin, not merely tied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from spy_der.domain.enums import CONTRACT_MULTIPLIER, MarketState, StrategyFamily
from spy_der.domain.forecast import ForecastBundle
from spy_der.domain.market import MarketSnapshot, OptionQuote
from spy_der.domain.strategy import (
    ExitPlan,
    Strategy,
    StrategyCandidate,
)
from spy_der.execution.fills import (
    CostModel,
    entry_slippage,
    fill_probability,
    liquidity_profile,
    round_trip_cost,
)
from spy_der.features.builder import MarketFeatures
from spy_der.optimizer.family_filter import eligible_families, families_for_state
from spy_der.optimizer.utility import (
    NoTradeOption,
    PenaltyScales,
    UtilityWeights,
    compute_utility,
    normalized_score,
)
from spy_der.simulation.exits import value_candidate
from spy_der.simulation.price_paths import PathBundle
from spy_der.strategies.generators.context import GenerationContext
from spy_der.strategies.generators.families import generate
from spy_der.strategies.metrics import resolved_profit, structure_greeks
from spy_der.strategies.registry import AdmittedStrategy, StrategyRegistry

MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0

#: Per-family exit defaults (spec section 30).
#: Credit structures take profit early because the last increments of a credit
#: are the slowest and riskiest to collect; long-volatility structures run
#: further because their whole value is in the tail.
EXIT_DEFAULTS: dict[StrategyFamily, tuple[float, float]] = {
    StrategyFamily.BULL_PUT_CREDIT_SPREAD: (0.50, 2.00),
    StrategyFamily.BEAR_CALL_CREDIT_SPREAD: (0.50, 2.00),
    StrategyFamily.IRON_CONDOR: (0.50, 1.80),
    StrategyFamily.ASYMMETRIC_CONDOR: (0.50, 1.80),
    StrategyFamily.BROKEN_WING_CONDOR: (0.50, 1.80),
    StrategyFamily.IRON_BUTTERFLY: (0.35, 1.50),
    StrategyFamily.CALL_BUTTERFLY: (0.55, 0.80),
    StrategyFamily.PUT_BUTTERFLY: (0.55, 0.80),
    StrategyFamily.BROKEN_WING_CALL_BUTTERFLY: (0.50, 1.00),
    StrategyFamily.BROKEN_WING_PUT_BUTTERFLY: (0.50, 1.00),
    StrategyFamily.BULL_CALL_DEBIT_SPREAD: (0.65, 0.60),
    StrategyFamily.BEAR_PUT_DEBIT_SPREAD: (0.65, 0.60),
    StrategyFamily.LONG_STRADDLE: (0.80, 0.50),
    StrategyFamily.LONG_STRANGLE: (0.80, 0.50),
    StrategyFamily.CALL_BACKSPREAD: (0.75, 0.60),
    StrategyFamily.PUT_BACKSPREAD: (0.75, 0.60),
    StrategyFamily.BULLISH_CALENDAR: (0.45, 0.70),
    StrategyFamily.BEARISH_CALENDAR: (0.45, 0.70),
    StrategyFamily.DOUBLE_CALENDAR: (0.45, 0.70),
    StrategyFamily.BULLISH_DIAGONAL: (0.50, 0.70),
    StrategyFamily.BEARISH_DIAGONAL: (0.50, 0.70),
}
DEFAULT_EXIT = (0.55, 0.80)


@dataclass
class OptimizerConfig:
    weights: UtilityWeights = field(default_factory=UtilityWeights)
    scales: PenaltyScales = field(default_factory=PenaltyScales)
    no_trade: NoTradeOption = field(default_factory=NoTradeOption)
    costs: CostModel = field(default_factory=CostModel)

    valuation_paths: int = 600
    valuation_steps: int = 48
    family_probability_threshold: float = 0.15
    max_candidates_per_family: int = 24
    max_total_candidates: int = 240

    min_expected_value: float = 0.0
    min_return_on_risk: float = 0.04
    max_risk_per_contract: float | None = None
    """Largest affordable per-contract loss, supplied by the risk engine.

    Candidates above this cannot be sized to a single contract, so ranking them
    would produce a no-trade even when an affordable alternative had real edge.
    ``None`` disables the gate."""

    min_fill_probability: float = 0.25
    max_spread_ratio: float = 0.20
    horizon: str = "eod"


@dataclass(frozen=True)
class OptimizationResult:
    """Ranked candidates plus the no-trade comparison (spec section 27)."""

    candidates: tuple[StrategyCandidate, ...]
    rejected: tuple[tuple[str, tuple[str, ...]], ...]
    no_trade_utility: float
    no_trade_margin: float
    eligible_families: frozenset[StrategyFamily]
    registry_rejections: dict[str, int]
    evaluated_count: int

    @property
    def best(self) -> StrategyCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def trade_selected(self) -> bool:
        """True only when a candidate beats no-trade by the required margin."""
        best = self.best
        if best is None:
            return False
        return best.utility > self.no_trade_utility + self.no_trade_margin

    @property
    def decision(self) -> str:
        best = self.best
        if best is None or not self.trade_selected:
            return "no_trade"
        return best.strategy_id

    def top(self, n: int = 5) -> tuple[StrategyCandidate, ...]:
        return self.candidates[:n]


@dataclass
class StrategyOptimizer:
    """Generates, evaluates, and ranks defined-risk candidates."""

    config: OptimizerConfig = field(default_factory=OptimizerConfig)

    def optimize(
        self,
        *,
        snapshot: MarketSnapshot,
        features: MarketFeatures,
        forecast: ForecastBundle,
        paths: PathBundle,
        max_risk_per_contract: float | None = None,
    ) -> OptimizationResult:
        cfg = self.config
        risk_cap = (
            max_risk_per_contract
            if max_risk_per_contract is not None
            else cfg.max_risk_per_contract
        )
        families = eligible_families(
            forecast.state_probabilities, threshold=cfg.family_probability_threshold
        )
        registry = StrategyRegistry()

        if not families:
            # Transition or Unstable dominates: no family is eligible and the
            # decision is no-trade without any candidate being constructed.
            return OptimizationResult(
                candidates=(),
                rejected=(),
                no_trade_utility=cfg.no_trade.utility,
                no_trade_margin=cfg.no_trade.margin,
                eligible_families=families,
                registry_rejections={},
                evaluated_count=0,
            )

        near, far = self._expirations(snapshot)
        if near is None:
            return OptimizationResult(
                candidates=(),
                rejected=(("no_expiration", ("no tradeable expiration in the chain",)),),
                no_trade_utility=cfg.no_trade.utility,
                no_trade_margin=cfg.no_trade.margin,
                eligible_families=families,
                registry_rejections={},
                evaluated_count=0,
            )

        ctx = GenerationContext(
            snapshot=snapshot,
            features=features,
            forecast=forecast,
            expiration=near,
            far_expiration=far,
            aggressiveness=cfg.costs.spread_capture,
            horizon=cfg.horizon,
        )

        structures = self._generate(ctx, families)
        admitted = registry.submit_all(structures)

        valuation_paths = paths.subsample(
            n_paths=cfg.valuation_paths, max_steps=cfg.valuation_steps
        )
        modal_families = families_for_state(forecast.dominant_state)

        evaluated: list[StrategyCandidate] = []
        rejected: list[tuple[str, tuple[str, ...]]] = []
        for index, entry in enumerate(admitted[: cfg.max_total_candidates]):
            # Screen affordability before paying for a path valuation, but
            # record the refusal rather than dropping it: spec 32 wants the
            # rejected alternatives and their reasons in the audit record.
            if risk_cap is not None and entry.profile.max_loss > risk_cap:
                rejected.append(
                    (
                        self._strategy_id(entry.strategy, index),
                        (
                            f"per-contract risk {entry.profile.max_loss:.0f} exceeds the "
                            f"affordable budget of {risk_cap:.0f}",
                        ),
                    )
                )
                continue

            candidate = self._evaluate(
                entry,
                index=index,
                snapshot=snapshot,
                features=features,
                forecast=forecast,
                paths=valuation_paths,
                modal_families=modal_families,
                risk_cap=risk_cap,
            )
            if candidate is None:
                rejected.append(
                    (
                        self._strategy_id(entry.strategy, index),
                        ("one or more legs are no longer quotable",),
                    )
                )
                continue
            if candidate.rejection_reasons:
                rejected.append((candidate.strategy_id, candidate.rejection_reasons))
            else:
                evaluated.append(candidate)

        evaluated.sort(key=lambda c: c.utility, reverse=True)

        return OptimizationResult(
            candidates=tuple(evaluated),
            rejected=tuple(rejected),
            no_trade_utility=cfg.no_trade.utility,
            no_trade_margin=cfg.no_trade.margin,
            eligible_families=families,
            registry_rejections=registry.rejection_summary(),
            evaluated_count=len(admitted),
        )

    # -- internals ---------------------------------------------------------

    def _expirations(
        self, snapshot: MarketSnapshot
    ) -> tuple[datetime | None, datetime | None]:
        """Near expiration for the structure, far expiration for calendars.

        The near expiration must have enough life left to be worth entering;
        a contract minutes from settlement cannot carry a managed exit plan.
        """
        near = snapshot.nearest_expiration(min_seconds=1_800.0)
        if near is None:
            return None, None
        later = [e for e in snapshot.expirations if e > near]
        return near, (later[0] if later else None)

    def _generate(
        self, ctx: GenerationContext, families: frozenset[StrategyFamily]
    ) -> list[Strategy]:
        out: list[Strategy] = []
        for family in sorted(families, key=lambda f: f.value):
            produced = generate(family, ctx)
            out.extend(produced[: self.config.max_candidates_per_family])
        return out

    def _evaluate(
        self,
        entry: AdmittedStrategy,
        *,
        index: int,
        snapshot: MarketSnapshot,
        features: MarketFeatures,
        forecast: ForecastBundle,
        paths: PathBundle,
        modal_families: frozenset[StrategyFamily],
        risk_cap: float | None = None,
    ) -> StrategyCandidate | None:
        cfg = self.config
        strategy = entry.strategy
        profile = entry.profile

        # Skip unaffordable structures before paying for a path valuation.
        if risk_cap is not None and profile.max_loss > risk_cap:
            return None

        quotes = self._quotes_for(snapshot, strategy)
        if quotes is None:
            return None

        liquidity = liquidity_profile(quotes)
        strategy_id = self._strategy_id(strategy, index)

        max_loss = profile.max_loss
        max_profit = resolved_profit(
            strategy,
            profile,
            spot=snapshot.spot,
            rate=snapshot.risk_free_rate,
            div_yield=snapshot.dividend_yield,
        )

        exit_plan = self._exit_plan(strategy, features, forecast)
        slippage = entry_slippage(strategy.legs, quotes, cfg.costs)
        costs = round_trip_cost(strategy.legs, quotes, cfg.costs)
        fill = fill_probability(
            quotes,
            leg_count=strategy.leg_count,
            contracts=1,
            model=cfg.costs,
            minutes_to_close=features.minutes_to_close,
        )

        valuation = value_candidate(
            strategy,
            paths,
            exit_plan,
            entry_time=snapshot.timestamp,
            max_profit=max_profit,
            max_loss=max_loss,
            round_trip_cost=costs,
            rate=snapshot.risk_free_rate,
            div_yield=snapshot.dividend_yield,
        )

        capital = self._capital_usage(strategy, max_loss)
        breakdown = compute_utility(
            expected_value=valuation.expected_value,
            cvar=valuation.cvar,
            max_loss=max_loss,
            expected_slippage=slippage,
            fill_probability=fill,
            confidence=forecast.confidence,
            liquidity=liquidity,
            leg_count=strategy.leg_count,
            capital_usage=capital,
            holding_minutes=valuation.mean_holding_minutes,
            weights=cfg.weights,
            scales=cfg.scales,
        )

        eror = valuation.expected_value / max_loss if max_loss > 0.0 else 0.0
        regime_fit = 1.0 if strategy.family in modal_families else 0.6

        candidate = StrategyCandidate(
            strategy_id=strategy_id,
            family=strategy.family,
            strategy=strategy,
            entry_price=strategy.net_entry_cost / CONTRACT_MULTIPLIER,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=profile.breakevens,
            greeks=structure_greeks(
                strategy,
                snapshot.spot,
                snapshot.timestamp,
                rate=snapshot.risk_free_rate,
                div_yield=snapshot.dividend_yield,
            ),
            liquidity=liquidity,
            fill_probability=fill,
            expected_slippage=slippage,
            expected_value=valuation.expected_value,
            expected_return_on_risk=eror,
            cvar=valuation.cvar,
            utility=breakdown.utility,
            assignment_risk=self._assignment_risk(strategy, snapshot, forecast),
            exit_plan=exit_plan,
            probability_of_profit=valuation.win_rate,
            capital_usage=capital,
            complexity_score=float(max(0, strategy.leg_count - 2)),
            diagnostics={
                **breakdown.as_dict(),
                "regime_fit": regime_fit,
                "round_trip_cost": costs,
                "mean_holding_minutes": valuation.mean_holding_minutes,
                "expected_mfe": valuation.expected_mfe,
                "expected_mae": valuation.expected_mae,
                "normalized_score": normalized_score(
                    expected_value=valuation.expected_value,
                    max_loss=max_loss,
                    confidence=forecast.confidence,
                    liquidity=liquidity.score,
                    regime_fit=regime_fit,
                    fill_probability=fill,
                ),
            },
        )

        gates = self._entry_gates(candidate)
        return candidate.with_rejections(gates) if gates else candidate

    def _entry_gates(self, candidate: StrategyCandidate) -> tuple[str, ...]:
        """Optimizer-level eligibility (spec section 29).

        These are screening thresholds, not risk vetoes. The risk engine holds
        absolute veto authority and runs afterwards on whatever survives here.
        """
        cfg = self.config
        reasons: list[str] = []
        if candidate.expected_value <= cfg.min_expected_value:
            reasons.append(
                f"expected value {candidate.expected_value:.2f} at or below minimum"
            )
        if candidate.expected_return_on_risk < cfg.min_return_on_risk:
            reasons.append(
                f"return on risk {candidate.expected_return_on_risk:.3f} below "
                f"{cfg.min_return_on_risk:.3f}"
            )
        if candidate.fill_probability < cfg.min_fill_probability:
            reasons.append(f"fill probability {candidate.fill_probability:.2f} too low")
        if candidate.liquidity.worst_spread_ratio > cfg.max_spread_ratio:
            reasons.append(
                f"worst leg spread ratio {candidate.liquidity.worst_spread_ratio:.3f} "
                f"exceeds {cfg.max_spread_ratio:.3f}"
            )
        return tuple(reasons)

    @staticmethod
    def _quotes_for(
        snapshot: MarketSnapshot, strategy: Strategy
    ) -> tuple[OptionQuote, ...] | None:
        quotes: list[OptionQuote] = []
        for leg in strategy.legs:
            quote = snapshot.quote_for(leg.expiry, leg.strike, leg.right)
            if quote is None:
                return None
            quotes.append(quote)
        return tuple(quotes)

    @staticmethod
    def _strategy_id(strategy: Strategy, index: int) -> str:
        strikes = "-".join(f"{k:g}" for k in strategy.strikes)
        return f"{strategy.family.value}:{strikes}:{index}"

    def _exit_plan(
        self, strategy: Strategy, features: MarketFeatures, forecast: ForecastBundle
    ) -> ExitPlan:
        target, stop = EXIT_DEFAULTS.get(strategy.family, DEFAULT_EXIT)
        seconds_to_expiry = max(
            0.0, (strategy.near_expiry - forecast.timestamp).total_seconds()
        )
        # Hold no longer than the session, the forecast horizon, or the life of
        # the near leg minus the assignment buffer, whichever binds first.
        holding = min(
            max(15.0, features.minutes_to_close),
            max(15.0, seconds_to_expiry / 60.0 - 30.0),
            390.0,
        )
        invalidation = tuple(
            state.value
            for state in (MarketState.TRANSITION, MarketState.UNSTABLE)
        )
        return ExitPlan(
            profit_target_fraction=min(1.0, target),
            stop_loss_fraction=min(1.0, stop),
            max_holding_minutes=holding,
            invalidation_states=invalidation,
            close_before_expiry_minutes=30.0,
            min_forecast_probability=0.25,
        )

    @staticmethod
    def _capital_usage(strategy: Strategy, max_loss: float) -> float:
        """Capital committed: the debit paid, or the margin a credit requires.

        For a defined-risk credit structure the broker holds the maximum loss,
        so that is the capital at work regardless of the credit received.
        """
        net = strategy.net_entry_cost
        return max(net, max_loss) if net > 0.0 else max_loss

    @staticmethod
    def _assignment_risk(
        strategy: Strategy, snapshot: MarketSnapshot, forecast: ForecastBundle
    ) -> float:
        """Probability that a short leg finishes in the money (spec 30.5).

        American-style SPY options can be exercised early, so any short leg that
        finishes in the money is an assignment exposure. The system never plans
        to be assigned; this number feeds the risk veto and the exit plan.
        """
        shorts = [leg for leg in strategy.legs if leg.is_short]
        if not shorts:
            return 0.0

        horizon = forecast.horizons.get("expiration") or next(
            iter(forecast.horizons.values())
        )
        worst = 0.0
        for leg in shorts:
            finish = horizon.finish_probabilities.get(leg.strike)
            if finish is None:
                below = horizon.probability_below(leg.strike)
                finish = 1.0 - below
            itm = finish if leg.right.value == "call" else 1.0 - finish
            worst = max(worst, float(itm))
        return worst
