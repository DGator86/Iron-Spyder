"""Strategy domain types (spec sections 23 and 34.3).

A leg carries a *signed* quantity rather than a side enum. Every downstream
calculation — payoff, exposure, net cost, tail slope — is then plain arithmetic
over ``q_i``, which removes an entire class of sign-handling bugs from the
defined-risk proof in :mod:`spy_der.strategies.validation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from spy_der.domain.enums import (
    CONTRACT_MULTIPLIER,
    ExitReason,
    OptionRight,
    StrategyFamily,
)


@dataclass(frozen=True, slots=True, order=True)
class OptionLeg:
    """``L_i = (q_i, K_i, T_i, chi_i)`` from spec section 23.

    ``quantity`` is signed: positive is long, negative is short, and zero is
    rejected because a zero-quantity leg silently changes a structure's shape
    without changing its arithmetic.
    """

    expiry: datetime
    strike: float
    right: OptionRight
    quantity: int
    entry_price: float = 0.0
    entry_iv: float = 0.0
    """Implied volatility at construction, used to mark the leg along a path.

    Carried on the leg rather than looked up later because a path simulation
    reprices the structure thousands of times and must not depend on the chain
    still being reachable.
    """

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError("OptionLeg.quantity must be non-zero")
        if self.strike <= 0.0:
            raise ValueError("OptionLeg.strike must be positive")
        if self.entry_price < 0.0:
            raise ValueError("OptionLeg.entry_price must be non-negative")

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def intrinsic(self, spot: float) -> float:
        """Per-share intrinsic value at ``spot``."""
        if self.right is OptionRight.CALL:
            return max(0.0, spot - self.strike)
        return max(0.0, self.strike - spot)

    def scaled(self, factor: int) -> OptionLeg:
        """Return this leg with its quantity multiplied by ``factor``."""
        return OptionLeg(
            expiry=self.expiry,
            strike=self.strike,
            right=self.right,
            quantity=self.quantity * factor,
            entry_price=self.entry_price,
            entry_iv=self.entry_iv,
        )

    @property
    def signed_cost(self) -> float:
        """Cash outlay for this leg: positive is a debit paid, negative a credit."""
        return self.quantity * CONTRACT_MULTIPLIER * self.entry_price


@dataclass(frozen=True, slots=True)
class StrategyGreeks:
    """Position-level Greeks, expressed per one unit of the structure."""

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    vanna: float = 0.0
    charm: float = 0.0

    def scaled(self, factor: float) -> StrategyGreeks:
        return StrategyGreeks(
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            theta=self.theta * factor,
            vega=self.vega * factor,
            vanna=self.vanna * factor,
            charm=self.charm * factor,
        )


@dataclass(frozen=True, slots=True)
class ExitPlan:
    """Mandatory exit contract for every open strategy (spec section 30).

    Every field is required to be meaningful: the spec forbids an open position
    without a profit target, a risk stop, a time stop, a state-invalidation
    rule and an end-of-session rule.
    """

    profit_target_fraction: float
    stop_loss_fraction: float
    max_holding_minutes: float
    invalidation_states: tuple[str, ...] = ()
    close_before_expiry_minutes: float = 30.0
    min_forecast_probability: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.profit_target_fraction <= 1.0:
            raise ValueError("profit_target_fraction must be in (0, 1]")
        if not 0.0 < self.stop_loss_fraction <= 1.0:
            raise ValueError("stop_loss_fraction must be in (0, 1]")
        if self.max_holding_minutes <= 0.0:
            raise ValueError("max_holding_minutes must be positive")

    @property
    def reasons_covered(self) -> frozenset[ExitReason]:
        return frozenset(
            {
                ExitReason.PROFIT_TARGET,
                ExitReason.RISK_STOP,
                ExitReason.TIME_STOP,
                ExitReason.STATE_INVALIDATION,
                ExitReason.END_OF_SESSION,
                ExitReason.ASSIGNMENT_RISK,
            }
        )


@dataclass(frozen=True, slots=True)
class Strategy:
    """A structural strategy: a family plus its legs, before evaluation."""

    family: StrategyFamily
    legs: tuple[OptionLeg, ...]

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("Strategy must have at least one leg")

    @property
    def expiries(self) -> tuple[datetime, ...]:
        return tuple(sorted({leg.expiry for leg in self.legs}))

    @property
    def near_expiry(self) -> datetime:
        return self.expiries[0]

    @property
    def far_expiry(self) -> datetime:
        return self.expiries[-1]

    @property
    def strikes(self) -> tuple[float, ...]:
        return tuple(sorted({leg.strike for leg in self.legs}))

    @property
    def is_multi_expiry(self) -> bool:
        return len(self.expiries) > 1

    @property
    def net_entry_cost(self) -> float:
        """Total cash outlay: positive is a net debit, negative a net credit."""
        return sum(leg.signed_cost for leg in self.legs)

    @property
    def contract_count(self) -> int:
        """Total contracts traded, used for commissions and complexity scoring."""
        return sum(abs(leg.quantity) for leg in self.legs)

    @property
    def leg_count(self) -> int:
        return len(self.legs)

    def scaled(self, contracts: int) -> Strategy:
        if contracts < 1:
            raise ValueError("contracts must be >= 1")
        return Strategy(
            family=self.family, legs=tuple(leg.scaled(contracts) for leg in self.legs)
        )


@dataclass(frozen=True, slots=True)
class LiquidityProfile:
    """Execution-quality inputs kept separate from forecast quality."""

    score: float
    worst_spread_ratio: float
    total_open_interest: int
    total_volume: int
    min_quote_size: int

    @property
    def is_tradeable(self) -> bool:
        return self.score > 0.0 and self.worst_spread_ratio < float("inf")


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """A fully evaluated, rankable strategy (spec 34.3)."""

    strategy_id: str
    family: StrategyFamily
    strategy: Strategy
    entry_price: float
    max_profit: float
    max_loss: float
    breakevens: tuple[float, ...]
    greeks: StrategyGreeks
    liquidity: LiquidityProfile
    fill_probability: float
    expected_slippage: float
    expected_value: float
    expected_return_on_risk: float
    cvar: float
    utility: float
    assignment_risk: float
    exit_plan: ExitPlan
    probability_of_profit: float = 0.0
    capital_usage: float = 0.0
    complexity_score: float = 0.0
    contracts: int = 1
    diagnostics: dict[str, float] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()

    @property
    def legs(self) -> tuple[OptionLeg, ...]:
        return self.strategy.legs

    @property
    def is_eligible(self) -> bool:
        return not self.rejection_reasons

    @property
    def total_max_loss(self) -> float:
        """Maximum loss for the sized position, in dollars."""
        return self.max_loss * self.contracts

    def with_rejections(self, reasons: tuple[str, ...]) -> StrategyCandidate:
        from dataclasses import replace

        return replace(self, rejection_reasons=self.rejection_reasons + reasons)


#: Sentinel identifier for the explicit no-trade competitor (spec section 27).
NO_TRADE_ID = "NO_TRADE"
