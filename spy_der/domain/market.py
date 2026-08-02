"""Market data domain types (spec sections 5 and 34.1)."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import cached_property

from spy_der.domain.enums import ExerciseStyle, OptionRight, TradeDirection

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True, slots=True)
class SpyQuote:
    """Top-of-book SPY quote."""

    timestamp: datetime
    bid: float
    ask: float
    last: float

    @property
    def mid(self) -> float:
        if self.bid > 0.0 and self.ask > 0.0:
            return 0.5 * (self.bid + self.ask)
        return self.last

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)


@dataclass(frozen=True, slots=True)
class PriceBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """A single SPY option contract snapshot (spec 5.2)."""

    contract_symbol: str
    timestamp: datetime
    expiration: datetime
    strike: float
    right: OptionRight
    bid: float
    ask: float
    last: float
    bid_size: int
    ask_size: int
    volume: int
    open_interest: int
    implied_volatility: float
    underlying_price: float
    exercise_style: ExerciseStyle = ExerciseStyle.AMERICAN
    multiplier: int = 100
    source: str = "synthetic"

    @property
    def mid(self) -> float:
        """Midpoint ``M_i = (bid + ask) / 2`` (spec 5.2)."""
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_ratio(self) -> float:
        """Relative spread ``(ask - bid) / mid``; ``inf`` when the mid is degenerate."""
        mid = self.mid
        if mid <= 0.0:
            return float("inf")
        return self.spread / mid

    @property
    def seconds_to_expiry(self) -> float:
        return max(0.0, (self.expiration - self.timestamp).total_seconds())

    @property
    def tau(self) -> float:
        """Time to expiration in years, floored at zero."""
        return self.seconds_to_expiry / SECONDS_PER_YEAR

    @property
    def days_to_expiry(self) -> float:
        return self.seconds_to_expiry / 86400.0

    @property
    def is_zero_dte(self) -> bool:
        return self.expiration.date() == self.timestamp.date()

    @property
    def quote_is_tradeable(self) -> bool:
        """Loose sanity gate; the full rule set lives in the data-quality engine."""
        return self.bid >= 0.0 and self.ask > 0.0 and self.ask >= self.bid


@dataclass(frozen=True, slots=True)
class OptionTrade:
    """An executed option trade with the quote prevailing at execution (spec 5.3)."""

    contract_symbol: str
    timestamp: datetime
    price: float
    size: int
    bid_at_trade: float
    ask_at_trade: float
    expiration: datetime
    strike: float
    right: OptionRight
    exchange: str = ""
    is_sweep: bool = False
    multileg_id: str | None = None

    @property
    def direction(self) -> TradeDirection:
        """Aggressor inference (spec 5.3).

        A trade at or above the ask is buyer-initiated, at or below the bid is
        seller-initiated, and anything inside the spread is left unknown rather
        than guessed.
        """
        spread = self.ask_at_trade - self.bid_at_trade
        if spread <= 0.0:
            return TradeDirection.UNKNOWN
        tolerance = 0.25 * spread
        if self.price >= self.ask_at_trade - tolerance:
            return TradeDirection.BUY
        if self.price <= self.bid_at_trade + tolerance:
            return TradeDirection.SELL
        return TradeDirection.UNKNOWN

    @property
    def signed_premium(self) -> float:
        """``Direction * price * contracts * 100`` (spec 5.3)."""
        return float(self.direction.value) * self.price * self.size * 100.0


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Contextual, non-tradeable features (spec 5.4)."""

    vix: float | None = None
    vvix: float | None = None
    spx: float | None = None
    es: float | None = None
    advance_decline: float | None = None
    up_down_volume_ratio: float | None = None
    ten_year_yield: float | None = None
    is_monthly_expiration: bool = False
    is_quarterly_expiration: bool = False
    is_early_close: bool = False
    minutes_to_close: float = 390.0
    scheduled_events: tuple[str, ...] = ()

    @property
    def has_blocking_event(self) -> bool:
        return bool(self.scheduled_events)


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything the decision pipeline is allowed to see at one instant.

    Deliberately not ``slots=True``: the cached properties below re-sort
    thousands of contracts and are read many times per decision, and
    ``cached_property`` needs an instance ``__dict__``.

    This is the point-in-time boundary: a backtest constructs snapshots from
    historical data only, which is what makes ``Decision_t = f(Data_{<=t})``
    (spec 37.4) structurally enforceable rather than merely intended.
    """

    timestamp: datetime
    spy_quote: SpyQuote
    spy_bars: tuple[PriceBar, ...]
    option_chain: tuple[OptionQuote, ...]
    option_trades: tuple[OptionTrade, ...] = ()
    context: ContextSnapshot = field(default_factory=ContextSnapshot)
    data_quality_score: float = 1.0
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.013

    @property
    def spot(self) -> float:
        return self.spy_quote.mid

    @cached_property
    def expirations(self) -> tuple[datetime, ...]:
        return tuple(sorted({q.expiration for q in self.option_chain}))

    @cached_property
    def expiration_dates(self) -> tuple[date, ...]:
        return tuple(sorted({q.expiration.date() for q in self.option_chain}))

    def chain_for(self, expiration: datetime) -> tuple[OptionQuote, ...]:
        return tuple(q for q in self.option_chain if q.expiration == expiration)

    def strikes_for(self, expiration: datetime) -> tuple[float, ...]:
        return tuple(sorted({q.strike for q in self.chain_for(expiration)}))

    def quote_for(
        self, expiration: datetime, strike: float, right: OptionRight
    ) -> OptionQuote | None:
        for q in self.option_chain:
            if q.expiration == expiration and q.strike == strike and q.right is right:
                return q
        return None

    def nearest_expiration(self, *, min_seconds: float = 0.0) -> datetime | None:
        """Earliest expiration with at least ``min_seconds`` of life remaining."""
        for exp in self.expirations:
            if (exp - self.timestamp).total_seconds() >= min_seconds:
                return exp
        return None

    def atm_strike(self, expiration: datetime) -> float | None:
        strikes = self.strikes_for(expiration)
        if not strikes:
            return None
        return nearest_value(strikes, self.spot)


def nearest_value(sorted_values: Sequence[float], target: float) -> float:
    """Return the element of ``sorted_values`` closest to ``target``."""
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    idx = bisect_left(sorted_values, target)
    if idx == 0:
        return sorted_values[0]
    if idx == len(sorted_values):
        return sorted_values[-1]
    lo, hi = sorted_values[idx - 1], sorted_values[idx]
    return lo if (target - lo) <= (hi - target) else hi


def realized_volatility(bars: Iterable[PriceBar], *, periods_per_year: float = 98_280.0) -> float:
    """Close-to-close realized volatility, annualized.

    The default ``periods_per_year`` corresponds to one-minute bars over a
    252-day year of 390-minute sessions.
    """
    closes = [b.close for b in bars if b.close > 0.0]
    if len(closes) < 3:
        return 0.0
    import math

    rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0.0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(max(0.0, var) * periods_per_year)
