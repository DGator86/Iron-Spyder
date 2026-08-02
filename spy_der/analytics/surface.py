"""Expected move, skew, and term structure (spec sections 9.1-9.5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from spy_der.analytics.chain import ChainArrays, build_chain_arrays
from spy_der.domain.enums import OptionRight
from spy_der.domain.market import MarketSnapshot, nearest_value


@dataclass(frozen=True)
class ExpectedMove:
    """ATM straddle and volatility-implied expected move (spec 9.1, 9.2)."""

    straddle_move: float
    vol_move: float
    spot: float
    atm_strike: float
    atm_iv: float
    tau: float
    session_open: float

    @property
    def move(self) -> float:
        """Preferred estimate: the straddle when quotable, else the vol formula."""
        return self.straddle_move if self.straddle_move > 0.0 else self.vol_move

    @property
    def upper(self) -> float:
        return self.spot + self.move

    @property
    def lower(self) -> float:
        return self.spot - self.move

    @property
    def move_pct(self) -> float:
        return self.move / max(self.spot, 1e-9)

    @property
    def utilization(self) -> float:
        """``EMU_t = |S_t - S_open| / EM`` (spec 9.1)."""
        if self.move <= 0.0:
            return 0.0
        return abs(self.spot - self.session_open) / self.move

    @property
    def remaining(self) -> float:
        """``REM_t = EM - |S_t - S_open|`` (spec 9.1)."""
        return self.move - abs(self.spot - self.session_open)

    @property
    def remaining_fraction(self) -> float:
        if self.move <= 0.0:
            return 0.0
        return float(np.clip(self.remaining / self.move, 0.0, 1.0))


def compute_expected_move(
    snapshot: MarketSnapshot, expiration: datetime, *, session_open: float | None = None
) -> ExpectedMove | None:
    """ATM straddle expected move for one expiration."""
    chain = snapshot.chain_for(expiration)
    if not chain:
        return None

    strikes = sorted({q.strike for q in chain})
    atm = nearest_value(strikes, snapshot.spot)
    call = snapshot.quote_for(expiration, atm, OptionRight.CALL)
    put = snapshot.quote_for(expiration, atm, OptionRight.PUT)
    if call is None or put is None:
        return None

    straddle = call.mid + put.mid if call.mid > 0.0 and put.mid > 0.0 else 0.0
    atm_iv = 0.5 * (call.implied_volatility + put.implied_volatility)
    tau = call.tau
    vol_move = snapshot.spot * atm_iv * float(np.sqrt(max(tau, 0.0)))

    opening = session_open
    if opening is None:
        opening = snapshot.spy_bars[0].open if snapshot.spy_bars else snapshot.spot

    return ExpectedMove(
        straddle_move=straddle,
        vol_move=vol_move,
        spot=snapshot.spot,
        atm_strike=atm,
        atm_iv=atm_iv,
        tau=tau,
        session_open=opening,
    )


@dataclass(frozen=True)
class SkewMetrics:
    """25-delta skew geometry (spec 9.3)."""

    atm_iv: float
    call_25d_iv: float
    put_25d_iv: float
    call_25d_strike: float
    put_25d_strike: float

    @property
    def put_skew(self) -> float:
        """``IV_25dPut - IV_ATM``."""
        return self.put_25d_iv - self.atm_iv

    @property
    def call_skew(self) -> float:
        """``IV_25dCall - IV_ATM``."""
        return self.call_25d_iv - self.atm_iv

    @property
    def risk_reversal(self) -> float:
        """``RR_25 = IV_25dCall - IV_25dPut``."""
        return self.call_25d_iv - self.put_25d_iv

    @property
    def butterfly(self) -> float:
        """``BF_25 = (IV_25dCall + IV_25dPut)/2 - IV_ATM``."""
        return 0.5 * (self.call_25d_iv + self.put_25d_iv) - self.atm_iv


def compute_skew(arrays: ChainArrays, *, target_delta: float = 0.25) -> SkewMetrics | None:
    """Interpolate IV at the 25-delta wings and at the money.

    Interpolating in delta space rather than picking the nearest listed strike
    keeps the metric stable when the strike grid is coarse, which matters most
    on short-dated expirations where a single strike step is several deltas.

    ``arrays`` must cover exactly one expiration. Mixing tenors puts near-binary
    0DTE deltas next to smooth weekly ones, and the interpolation then reports a
    risk reversal near zero regardless of how steep the smile actually is.
    """
    distinct_taus = np.unique(np.round(arrays.tau, 12))
    if distinct_taus.size > 1:
        raise ValueError(
            f"compute_skew requires a single expiration, got {distinct_taus.size}; "
            "use build_expiration_arrays() to slice the chain first"
        )

    call_mask = arrays.call_mask
    put_mask = arrays.put_mask
    if not np.any(call_mask) or not np.any(put_mask):
        return None

    atm_index = int(np.argmin(np.abs(arrays.strike - arrays.spot)))
    atm_iv = float(arrays.iv[atm_index])

    call_iv, call_strike = _interpolate_at_delta(
        arrays.delta[call_mask], arrays.iv[call_mask], arrays.strike[call_mask], target_delta
    )
    put_iv, put_strike = _interpolate_at_delta(
        np.abs(arrays.delta[put_mask]), arrays.iv[put_mask], arrays.strike[put_mask], target_delta
    )
    if call_iv is None or put_iv is None:
        return None

    return SkewMetrics(
        atm_iv=atm_iv,
        call_25d_iv=call_iv,
        put_25d_iv=put_iv,
        call_25d_strike=call_strike,
        put_25d_strike=put_strike,
    )


def _interpolate_at_delta(
    deltas: np.ndarray, ivs: np.ndarray, strikes: np.ndarray, target: float
) -> tuple[float | None, float]:
    """Linear interpolation of IV and strike at ``|delta| = target``."""
    order = np.argsort(deltas)
    d_sorted = deltas[order]
    iv_sorted = ivs[order]
    k_sorted = strikes[order]
    if len(d_sorted) < 2 or target < d_sorted[0] or target > d_sorted[-1]:
        # Outside the quoted delta range; fall back to the closest contract.
        index = int(np.argmin(np.abs(deltas - target)))
        return float(ivs[index]), float(strikes[index])
    return (
        float(np.interp(target, d_sorted, iv_sorted)),
        float(np.interp(target, d_sorted, k_sorted)),
    )


@dataclass(frozen=True)
class TermStructure:
    """ATM implied volatility by expiration (spec 9.4)."""

    tenors: tuple[float, ...]
    """Time to expiration in years, ascending."""

    atm_ivs: tuple[float, ...]
    expirations: tuple[datetime, ...]

    @property
    def front_iv(self) -> float:
        return self.atm_ivs[0] if self.atm_ivs else 0.0

    def slope(self, near_index: int = 0, far_index: int = 1) -> float:
        """``TermSlope = IV(T2) - IV(T1)``."""
        if len(self.atm_ivs) <= max(near_index, far_index):
            return 0.0
        return self.atm_ivs[far_index] - self.atm_ivs[near_index]

    @property
    def curvature(self) -> float:
        """Second difference across the first three tenors."""
        if len(self.atm_ivs) < 3:
            return 0.0
        return self.atm_ivs[2] - 2.0 * self.atm_ivs[1] + self.atm_ivs[0]

    @property
    def is_inverted(self) -> bool:
        """Front IV above the next tenor — typical of event or stress pricing."""
        return len(self.atm_ivs) >= 2 and self.atm_ivs[0] > self.atm_ivs[1]


def compute_term_structure(snapshot: MarketSnapshot) -> TermStructure:
    """ATM IV for each expiration present in the chain."""
    tenors: list[float] = []
    ivs: list[float] = []
    expirations: list[datetime] = []

    for expiration in snapshot.expirations:
        chain = snapshot.chain_for(expiration)
        if not chain:
            continue
        strikes = sorted({q.strike for q in chain})
        atm = nearest_value(strikes, snapshot.spot)
        quotes = [q for q in chain if q.strike == atm]
        if not quotes:
            continue
        tenors.append(quotes[0].tau)
        ivs.append(float(np.mean([q.implied_volatility for q in quotes])))
        expirations.append(expiration)

    return TermStructure(
        tenors=tuple(tenors), atm_ivs=tuple(ivs), expirations=tuple(expirations)
    )


@dataclass(frozen=True)
class VolatilityEdge:
    """Implied versus realized volatility (spec 9.5)."""

    atm_iv: float
    realized_vol: float
    forecast_realized_vol: float

    @property
    def iv_rv_spread(self) -> float:
        """``IVRVSpread = IV_ATM - RV``."""
        return self.atm_iv - self.realized_vol

    @property
    def vol_edge(self) -> float:
        """``VolEdge = E[RV_future] - IV_current``.

        Positive favours long-volatility structures; negative favours
        defined-risk short-volatility structures.
        """
        return self.forecast_realized_vol - self.atm_iv

    @property
    def favors_long_vol(self) -> bool:
        return self.vol_edge > 0.0


def compute_volatility_edge(
    atm_iv: float, realized_vol: float, *, mean_reversion: float = 0.5, anchor: float | None = None
) -> VolatilityEdge:
    """Forecast near-term realized volatility by reverting toward implied.

    A deliberately simple estimator: realized volatility is pulled toward the
    implied level (or an explicit ``anchor``) at rate ``mean_reversion``. It is
    a baseline the nonlinear specialists must beat, not the final word.
    """
    target = anchor if anchor is not None else atm_iv
    forecast = realized_vol + mean_reversion * (target - realized_vol)
    return VolatilityEdge(
        atm_iv=atm_iv, realized_vol=realized_vol, forecast_realized_vol=forecast
    )


def atm_implied_volatility(snapshot: MarketSnapshot, expiration: datetime) -> float:
    """Average of the ATM call and put implied volatilities."""
    chain = snapshot.chain_for(expiration)
    if not chain:
        return 0.0
    strikes = sorted({q.strike for q in chain})
    atm = nearest_value(strikes, snapshot.spot)
    quotes = [q for q in chain if q.strike == atm]
    if not quotes:
        return 0.0
    return float(np.mean([q.implied_volatility for q in quotes]))


def build_expiration_arrays(
    snapshot: MarketSnapshot, expiration: datetime
) -> ChainArrays | None:
    quotes = snapshot.chain_for(expiration)
    if not quotes:
        return None
    return build_chain_arrays(
        quotes,
        spot=snapshot.spot,
        rate=snapshot.risk_free_rate,
        div_yield=snapshot.dividend_yield,
    )
