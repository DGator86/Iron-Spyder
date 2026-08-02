"""Risk limits and kill switches (spec sections 28 and 42)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from spy_der.domain.enums import KillSwitchReason


@dataclass(frozen=True)
class TradeLimits:
    """Per-trade controls (spec 28.1)."""

    max_account_fraction_at_risk: float = 0.005
    max_dollar_risk: float = 750.0
    max_contracts: int = 10
    max_debit_per_contract: float = 500.0
    max_spread_width: float = 25.0
    min_return_on_risk: float = 0.04
    min_liquidity_score: float = 0.35
    max_spread_ratio: float = 0.20
    max_expected_slippage_fraction: float = 0.20
    """Entry slippage as a fraction of maximum loss."""

    max_legs: int = 4
    max_gamma: float = 5_000.0
    max_vega: float = 2_500.0
    max_theta: float = 2_500.0
    max_assignment_risk: float = 0.20
    min_confidence: float = 0.20
    max_model_disagreement: float = 0.35
    min_data_quality: float = 0.70


@dataclass(frozen=True)
class PortfolioLimits:
    """Portfolio controls (spec 28.2).

    ``max_open_strategies`` defaults to one. Spec 28.2 is explicit that this may
    be relaxed only after portfolio-level interaction has been validated, and
    nothing in this system has validated it yet.
    """

    max_open_strategies: int = 1
    max_total_open_risk: float = 1_500.0
    max_total_risk_fraction: float = 0.02
    max_same_expiration_risk: float = 1_000.0
    max_zero_dte_risk: float = 500.0
    max_directional_risk: float = 1_000.0
    max_aggregate_delta: float = 2_000.0
    max_aggregate_gamma: float = 8_000.0
    max_aggregate_vega: float = 4_000.0
    max_capital_fraction: float = 0.25


@dataclass(frozen=True)
class SessionLimits:
    """Session controls (spec 28.3)."""

    daily_loss_limit: float = 1_500.0
    daily_loss_fraction: float = 0.02
    max_consecutive_losses: int = 3
    max_drawdown_fraction: float = 0.06
    max_trades_per_session: int = 6
    min_minutes_to_close: float = 20.0
    """No new positions inside this window; there is not enough time left to
    manage an exit, and complex-order liquidity thins out."""

    block_on_scheduled_event: bool = True


@dataclass(frozen=True)
class RiskConfig:
    trade: TradeLimits = field(default_factory=TradeLimits)
    portfolio: PortfolioLimits = field(default_factory=PortfolioLimits)
    session: SessionLimits = field(default_factory=SessionLimits)
    base_risk_fraction: float = 0.005
    """``f`` in ``R_base = Equity * f``; spec 28.4 gives 0.25%-0.50%.

    Set at the top of that range because the sizing formula then multiplies by
    four quality terms — confidence, liquidity, stability, calibration — whose
    product is typically 0.4 to 0.6 even in good conditions. Effective risk
    lands near 0.2%-0.3% of equity, and a lower ``f`` would size almost every
    admissible SPY structure to zero contracts on a retail-sized account.
    """

    max_slippage_ratio: float = 3.0
    """Realized-to-expected slippage ratio that trips the slippage kill switch."""

    min_calibration_quality: float = 0.35


@dataclass
class SessionState:
    """Mutable per-session accounting consumed by the session limits."""

    trading_day: date | None = None
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    slippage_samples: list[tuple[float, float]] = field(default_factory=list)
    """``(expected, realized)`` slippage pairs for drift monitoring."""

    def roll_to(self, day: date, equity: float) -> None:
        """Reset daily counters when the session changes.

        Peak equity deliberately survives the roll: drawdown is measured from
        the high-water mark, not from each morning's open.
        """
        if self.trading_day == day:
            return
        self.trading_day = day
        self.realized_pnl = 0.0
        self.trades_today = 0
        self.peak_equity = max(self.peak_equity, equity)

    def record_trade(self, pnl: float, equity: float) -> None:
        self.realized_pnl += pnl
        self.trades_today += 1
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0.0 else 0
        self.peak_equity = max(self.peak_equity, equity)

    def record_slippage(self, expected: float, realized: float) -> None:
        self.slippage_samples.append((expected, realized))
        if len(self.slippage_samples) > 200:
            self.slippage_samples.pop(0)

    def drawdown_fraction(self, equity: float) -> float:
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    @property
    def slippage_ratio(self) -> float:
        """Realized over expected slippage across recent fills."""
        if not self.slippage_samples:
            return 1.0
        expected = sum(e for e, _ in self.slippage_samples)
        realized = sum(r for _, r in self.slippage_samples)
        if expected <= 0.0:
            return 1.0
        return realized / expected


@dataclass
class KillSwitch:
    """Latching kill switches (spec section 42).

    Switches latch: once tripped they stay tripped until explicitly reset. A
    condition that clears on its own — a data feed that recovers for one
    snapshot — must not silently re-enable trading, because the operator has
    not yet seen why it tripped.
    """

    active: dict[KillSwitchReason, str] = field(default_factory=dict)

    def trip(self, reason: KillSwitchReason, detail: str = "") -> None:
        self.active.setdefault(reason, detail or reason.value)

    def reset(self, reason: KillSwitchReason | None = None) -> None:
        if reason is None:
            self.active.clear()
        else:
            self.active.pop(reason, None)

    @property
    def is_tripped(self) -> bool:
        return bool(self.active)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(f"kill switch: {r.value} ({d})" for r, d in sorted(self.active.items()))

    def __contains__(self, reason: object) -> bool:
        return reason in self.active
