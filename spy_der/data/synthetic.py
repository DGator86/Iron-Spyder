"""Synthetic SPY chain and scenario generation (spec section 37.3).

Every scenario has a *predefined* expected market state and expected risk
decision, which is what makes these useful as tests rather than as demos: the
state engine is asserted against a ground truth the generator knows.

Prices are produced by the same Black-Scholes model the analytics layer uses.
That is intentional for structural tests (walls, flip points, payoff geometry
are all exercised faithfully) but it also means these scenarios cannot validate
the pricing model itself — for that, see the finite-difference and put-call
parity tests, which do not use this generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta

import numpy as np

from spy_der.analytics import pricing
from spy_der.domain.enums import MarketState, OptionRight
from spy_der.domain.market import (
    ContextSnapshot,
    MarketSnapshot,
    OptionQuote,
    OptionTrade,
    PriceBar,
    SpyQuote,
)

MARKET_CLOSE = time(20, 0)  # 16:00 America/New_York in UTC during EDT


def _front_expiry_date(now: datetime) -> date:
    """Calendar date of the nearest cash close still strictly after ``now``.

    Synthetic scenarios include a 0DTE tenor. When the caller passes wall-clock
    time after the 20:00 UTC close (or on a weekend evening), ``today + 0DTE``
    would already be expired and trip a fatal data-quality lockout. Roll the
    front month forward so demos and API integration tests stay usable outside
    the cash session while preserving relative ``expiry_days`` spacing.
    """
    today_close = datetime.combine(now.date(), MARKET_CLOSE, tzinfo=now.tzinfo or UTC)
    if now < today_close:
        return now.date()
    return now.date() + timedelta(days=1)


def _expiration_for(now: datetime, days: float) -> datetime:
    """Settlement timestamp ``days`` after the live front-month close."""
    front = _front_expiry_date(now)
    expiry_date = front + timedelta(days=int(days))
    return datetime.combine(expiry_date, MARKET_CLOSE, tzinfo=now.tzinfo or UTC)


@dataclass(frozen=True)
class ScenarioSpec:
    """Knobs for generating a controlled option chain."""

    name: str
    expected_state: MarketState
    spot: float = 550.0
    atm_iv: float = 0.14
    put_skew: float = 0.35
    """Additional IV per unit of downside log-moneyness."""

    call_skew: float = 0.10
    term_slope: float = 0.01
    """IV difference between the one-week tenor and 0DTE.

    Expressed against a one-week reference rather than per year, because the
    tenors this system trades span hours to days: a per-year slope is invisible
    across a 0DTE-to-weekly curve.
    """

    expiry_days: tuple[float, ...] = (0.0, 1.0, 7.0)
    strike_step: float = 1.0
    strike_width_pct: float = 0.06

    pin_strike: float | None = None
    pin_oi_multiplier: float = 1.0
    call_wall_offset: float | None = None
    put_wall_offset: float | None = None
    wall_oi_multiplier: float = 1.0

    base_open_interest: float = 4_000.0
    call_oi_bias: float = 1.0
    """Multiplier on call open interest, which positions the gamma flip.

    The flip sits where gamma-weighted call and put open interest balance. With
    a symmetric chain that point coincides with spot and the positive/negative
    gamma read becomes a coin toss, so scenarios that mean to express a
    definite gamma regime set this away from 1.0.
    """

    base_volume: float = 900.0
    spread_ratio: float = 0.02
    min_spread: float = 0.01

    realized_vol: float = 0.10
    drift: float = 0.0
    session_open_offset: float = 0.0
    """Session open relative to current spot, in points."""

    quote_age_seconds: float = 0.0
    context: ContextSnapshot = field(default_factory=ContextSnapshot)
    corruptions: tuple[str, ...] = ()
    """Injected defects for the data-quality tests; see :func:`_corrupt`."""

    expected_tradeable: bool = True
    seed: int = 7

    @property
    def vol_edge(self) -> float:
        """Realized minus implied at the money: the edge built into this chain.

        The chain is priced off :attr:`atm_iv` while paths are drawn at
        :attr:`realized_vol`, so this difference *is* the mispricing, and it is
        the only source of genuine edge in the synthetic world. Everything else
        about the chain is arbitrage-free by construction.

        Positive favours long volatility, negative favours short. A scenario at
        zero has nothing to find, and no-trade is the correct answer there —
        which makes this the quantity to test edge detection against.
        """
        return self.realized_vol - self.atm_iv


#: Reference tenor for the term-structure slope: one week, in years.
_TERM_REFERENCE_TAU = 7.0 / 365.0


def _iv_for(spec: ScenarioSpec, strike: float, tau: float) -> float:
    """Smile plus term structure, floored to stay invertible."""
    log_moneyness = float(np.log(max(strike, 1e-9) / spec.spot))
    smile = (
        spec.put_skew * max(0.0, -log_moneyness) + spec.call_skew * max(0.0, log_moneyness)
    )
    term = spec.term_slope * (float(np.sqrt(max(tau, 0.0) / _TERM_REFERENCE_TAU)) - 1.0)
    iv = spec.atm_iv + smile + term
    return float(max(0.02, iv))


def _open_interest(
    spec: ScenarioSpec,
    strike: float,
    tau: float,
    right: OptionRight,
    rng: np.random.Generator,
) -> int:
    """Bell-shaped OI around spot with scenario-specific concentrations.

    Call and put open interest are shaped separately. Gamma is identical for a
    call and a put at the same strike, so if both rights carried the same open
    interest the call wall and put wall would always land on the same strike.
    Real chains skew call OI above spot and put OI below it, and that asymmetry
    is what makes the two walls distinct.
    """
    width = max(spec.spot * 0.02, 1.0)
    base = spec.base_open_interest * float(np.exp(-0.5 * ((strike - spec.spot) / width) ** 2))
    base *= 1.0 / (1.0 + 2.0 * tau)  # near-dated expiries hold more OI

    offset = strike - spec.spot
    if right is OptionRight.CALL:
        base *= (1.0 + 0.9 * float(np.tanh(offset / width))) * spec.call_oi_bias
        wall_offset = spec.call_wall_offset
    else:
        base *= 1.0 - 0.9 * float(np.tanh(offset / width))
        wall_offset = spec.put_wall_offset

    if spec.pin_strike is not None and abs(strike - spec.pin_strike) < 1e-9:
        base *= spec.pin_oi_multiplier
    if wall_offset is not None and abs(strike - (spec.spot + wall_offset)) < 1e-9:
        base *= spec.wall_oi_multiplier

    noise = float(rng.uniform(0.85, 1.15))
    return int(max(1.0, base * noise))


def build_snapshot(spec: ScenarioSpec, *, timestamp: datetime | None = None) -> MarketSnapshot:
    """Generate a complete :class:`MarketSnapshot` for ``spec``."""
    rng = np.random.default_rng(spec.seed)
    now = timestamp or datetime(2026, 8, 3, 15, 30, tzinfo=UTC)

    quotes: list[OptionQuote] = []
    for days in spec.expiry_days:
        expiration = _expiration_for(now, days)
        tau = max(1e-6, (expiration - now).total_seconds()) / (365.0 * 24.0 * 3600.0)
        span = spec.spot * spec.strike_width_pct
        lo = spec.strike_step * np.floor((spec.spot - span) / spec.strike_step)
        hi = spec.strike_step * np.ceil((spec.spot + span) / spec.strike_step)
        strikes = np.arange(lo, hi + spec.strike_step * 0.5, spec.strike_step)

        for strike in strikes.tolist():
            iv = _iv_for(spec, strike, tau)
            for right in (OptionRight.CALL, OptionRight.PUT):
                oi = _open_interest(spec, strike, tau, right, rng)
                theo = float(
                    pricing.price(spec.spot, strike, tau, 0.04, 0.013, iv, right)
                )
                theo = max(theo, 0.01)
                half = max(spec.min_spread, theo * spec.spread_ratio) * 0.5
                bid = max(0.01, theo - half)
                ask = theo + half
                moneyness = abs(strike - spec.spot) / spec.spot
                volume = int(
                    max(0.0, spec.base_volume * float(np.exp(-40.0 * moneyness * moneyness)))
                )
                quotes.append(
                    OptionQuote(
                        contract_symbol=_symbol(expiration, strike, right),
                        timestamp=now - timedelta(seconds=spec.quote_age_seconds),
                        expiration=expiration,
                        strike=float(strike),
                        right=right,
                        bid=round(bid, 2),
                        ask=round(ask, 2),
                        last=round(theo, 2),
                        bid_size=int(rng.integers(5, 200)),
                        ask_size=int(rng.integers(5, 200)),
                        volume=volume,
                        open_interest=oi,
                        implied_volatility=iv,
                        underlying_price=spec.spot,
                    )
                )

    bars = _synthetic_bars(spec, now, rng)
    spy_quote = SpyQuote(
        timestamp=now, bid=spec.spot - 0.01, ask=spec.spot + 0.01, last=spec.spot
    )

    snapshot = MarketSnapshot(
        timestamp=now,
        spy_quote=spy_quote,
        spy_bars=tuple(bars),
        option_chain=tuple(quotes),
        option_trades=tuple(_synthetic_trades(spec, now, quotes, rng)),
        context=spec.context,
    )
    return _corrupt(snapshot, spec)


def _symbol(expiration: datetime, strike: float, right: OptionRight) -> str:
    code = "C" if right is OptionRight.CALL else "P"
    return f"SPY{expiration:%y%m%d}{code}{int(round(strike * 1000)):08d}"


def _synthetic_bars(
    spec: ScenarioSpec, now: datetime, rng: np.random.Generator, count: int = 90
) -> list[PriceBar]:
    """One-minute bars ending at ``now``, consistent with the scenario's vol and drift."""
    dt = 1.0 / (252.0 * 390.0)
    sigma = spec.realized_vol * float(np.sqrt(dt))
    mu = spec.drift / 390.0

    shocks = rng.normal(mu, sigma, size=count)
    # Anchor the path so its final close is exactly the scenario spot.
    log_path = np.cumsum(shocks)
    log_path = log_path - log_path[-1]
    closes = spec.spot * np.exp(log_path)
    opens = np.concatenate(([spec.spot + spec.session_open_offset], closes[:-1]))

    bars: list[PriceBar] = []
    for i in range(count):
        o, c = float(opens[i]), float(closes[i])
        wiggle = abs(float(rng.normal(0.0, spec.spot * sigma)))
        bars.append(
            PriceBar(
                timestamp=now - timedelta(minutes=count - i),
                open=o,
                high=max(o, c) + wiggle,
                low=min(o, c) - wiggle,
                close=c,
                volume=float(rng.integers(200_000, 900_000)),
                vwap=0.5 * (o + c),
            )
        )
    return bars


def _synthetic_trades(
    spec: ScenarioSpec,
    now: datetime,
    quotes: list[OptionQuote],
    rng: np.random.Generator,
    count: int = 40,
) -> list[OptionTrade]:
    """Aggressor-signed trades biased by the scenario's drift."""
    if not quotes:
        return []
    liquid = [q for q in quotes if q.volume > 0] or quotes
    trades: list[OptionTrade] = []
    buy_bias = 0.5 + float(np.clip(spec.drift * 4.0, -0.35, 0.35))

    for _ in range(count):
        quote = liquid[int(rng.integers(0, len(liquid)))]
        is_buy = rng.random() < buy_bias
        price = quote.ask if is_buy else quote.bid
        trades.append(
            OptionTrade(
                contract_symbol=quote.contract_symbol,
                timestamp=now - timedelta(seconds=int(rng.integers(1, 900))),
                price=price,
                size=int(rng.integers(1, 250)),
                bid_at_trade=quote.bid,
                ask_at_trade=quote.ask,
                expiration=quote.expiration,
                strike=quote.strike,
                right=quote.right,
                is_sweep=bool(rng.random() < 0.15),
            )
        )
    return trades


def _corrupt(snapshot: MarketSnapshot, spec: ScenarioSpec) -> MarketSnapshot:
    """Inject the defects named in ``spec.corruptions`` for data-quality tests."""
    if not spec.corruptions:
        return snapshot

    chain = list(snapshot.option_chain)
    for defect in spec.corruptions:
        if defect == "crossed_market" and chain:
            chain[0] = replace(chain[0], bid=chain[0].ask + 0.10)
        elif defect == "negative_bid" and len(chain) > 1:
            chain[1] = replace(chain[1], bid=-0.05)
        elif defect == "zero_mid" and len(chain) > 2:
            chain[2] = replace(chain[2], bid=0.0, ask=0.0)
        elif defect == "invalid_iv" and len(chain) > 3:
            chain[3] = replace(chain[3], implied_volatility=-1.0)
        elif defect == "duplicate" and chain:
            chain.append(chain[0])
        elif defect == "expired_contract" and chain:
            chain.append(replace(chain[0], expiration=snapshot.timestamp - timedelta(days=1)))
        elif defect == "stale_quotes":
            chain = [
                replace(q, timestamp=snapshot.timestamp - timedelta(minutes=20)) for q in chain
            ]
        elif defect == "wide_spreads":
            chain = [replace(q, bid=max(0.01, q.mid * 0.4), ask=q.mid * 1.6) for q in chain]
        elif defect == "missing_same_day":
            today = snapshot.timestamp.date()
            chain = [q for q in chain if q.expiration.date() != today]

    return replace(snapshot, option_chain=tuple(chain))


# --------------------------------------------------------------------------
# Named scenarios with predefined expected states (spec 37.3)
# --------------------------------------------------------------------------

SCENARIOS: dict[str, ScenarioSpec] = {
    "strong_pin": ScenarioSpec(
        name="strong_pin",
        expected_state=MarketState.STRONG_PIN,
        atm_iv=0.10,
        pin_strike=550.0,
        pin_oi_multiplier=14.0,
        call_oi_bias=1.6,
        realized_vol=0.05,
        drift=0.0,
        expiry_days=(0.0, 1.0, 7.0),
    ),
    "broad_range": ScenarioSpec(
        name="broad_range",
        expected_state=MarketState.BROAD_RANGE,
        atm_iv=0.13,
        call_wall_offset=8.0,
        put_wall_offset=-8.0,
        wall_oi_multiplier=30.0,
        call_oi_bias=1.5,
        realized_vol=0.105,
        drift=0.0,
    ),
    "bull_grind": ScenarioSpec(
        name="bull_grind",
        expected_state=MarketState.BULL_GRIND,
        atm_iv=0.12,
        put_wall_offset=-5.0,
        wall_oi_multiplier=26.0,
        call_oi_bias=1.45,
        realized_vol=0.09,
        drift=0.06,
        session_open_offset=-2.5,
    ),
    "bear_grind": ScenarioSpec(
        name="bear_grind",
        expected_state=MarketState.BEAR_GRIND,
        atm_iv=0.125,
        put_skew=0.55,
        call_wall_offset=5.0,
        put_wall_offset=-7.0,
        wall_oi_multiplier=26.0,
        call_oi_bias=1.35,
        realized_vol=0.095,
        drift=-0.06,
        session_open_offset=2.5,
    ),
    "bull_breakout": ScenarioSpec(
        name="bull_breakout",
        expected_state=MarketState.BULL_BREAKOUT,
        atm_iv=0.19,
        call_skew=0.22,
        call_oi_bias=0.6,
        realized_vol=0.28,
        drift=0.30,
        session_open_offset=-6.0,
        pin_oi_multiplier=1.0,
    ),
    "bear_breakdown": ScenarioSpec(
        name="bear_breakdown",
        expected_state=MarketState.BEAR_BREAKDOWN,
        atm_iv=0.24,
        put_skew=0.80,
        call_oi_bias=0.6,
        realized_vol=0.34,
        drift=-0.34,
        session_open_offset=7.0,
    ),
    "vol_expansion": ScenarioSpec(
        name="vol_expansion",
        expected_state=MarketState.VOL_EXPANSION,
        atm_iv=0.30,
        term_slope=-0.06,
        call_oi_bias=0.55,
        realized_vol=0.42,
        drift=0.0,
        session_open_offset=-1.0,
    ),
    "vol_contraction": ScenarioSpec(
        name="vol_contraction",
        expected_state=MarketState.VOL_CONTRACTION,
        atm_iv=0.13,
        term_slope=0.05,
        call_oi_bias=1.55,
        realized_vol=0.05,
        call_wall_offset=7.0,
        put_wall_offset=-7.0,
        wall_oi_multiplier=30.0,
        drift=0.0,
    ),
    "directional_range": ScenarioSpec(
        name="directional_range",
        expected_state=MarketState.DIRECTIONAL_RANGE,
        atm_iv=0.14,
        put_skew=0.60,
        put_wall_offset=-6.0,
        call_wall_offset=11.0,
        wall_oi_multiplier=34.0,
        call_oi_bias=1.5,
        realized_vol=0.10,
        drift=0.02,
    ),
    # --- adverse conditions: trading must be disabled -----------------------
    "poor_liquidity": ScenarioSpec(
        name="poor_liquidity",
        expected_state=MarketState.UNSTABLE,
        spread_ratio=0.60,
        min_spread=0.35,
        base_open_interest=40.0,
        base_volume=3.0,
        expected_tradeable=False,
    ),
    "wide_spreads": ScenarioSpec(
        name="wide_spreads",
        expected_state=MarketState.UNSTABLE,
        corruptions=("wide_spreads",),
        expected_tradeable=False,
    ),
    "stale_data": ScenarioSpec(
        name="stale_data",
        expected_state=MarketState.UNSTABLE,
        corruptions=("stale_quotes",),
        quote_age_seconds=1_200.0,
        expected_tradeable=False,
    ),
    "corrupt_quotes": ScenarioSpec(
        name="corrupt_quotes",
        expected_state=MarketState.UNSTABLE,
        corruptions=("crossed_market", "negative_bid", "zero_mid", "invalid_iv", "duplicate"),
        expected_tradeable=False,
    ),
    "missing_same_day": ScenarioSpec(
        name="missing_same_day",
        expected_state=MarketState.UNSTABLE,
        corruptions=("missing_same_day",),
        expected_tradeable=False,
    ),
    # Structurally an ordinary range; only the event calendar blocks trading,
    # so the expected state is normal and the veto must come from the risk engine.
    "event_lockout": ScenarioSpec(
        name="event_lockout",
        expected_state=MarketState.BROAD_RANGE,
        atm_iv=0.13,
        call_wall_offset=8.0,
        put_wall_offset=-8.0,
        wall_oi_multiplier=30.0,
        call_oi_bias=1.5,
        realized_vol=0.09,
        context=ContextSnapshot(scheduled_events=("FOMC",), minutes_to_close=120.0),
        expected_tradeable=False,
    ),
}


def scenario(name: str, **overrides: object) -> ScenarioSpec:
    """Fetch a named scenario, optionally overriding fields."""
    try:
        spec = SCENARIOS[name]
    except KeyError as exc:
        raise KeyError(f"unknown scenario {name!r}; have {sorted(SCENARIOS)}") from exc
    return replace(spec, **overrides) if overrides else spec  # type: ignore[arg-type]


def build_scenario(name: str, *, timestamp: datetime | None = None, **overrides: object) -> MarketSnapshot:
    return build_snapshot(scenario(name, **overrides), timestamp=timestamp)
