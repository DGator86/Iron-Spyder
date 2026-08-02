"""Black-Scholes-Merton pricing and the full Greek set (spec section 7).

Conventions used throughout:

* ``tau`` is time to expiration in **years**.
* ``rate`` and ``div_yield`` are continuously compounded annual rates.
* Greeks are per one unit of underlying, per ``1.0`` of volatility (not per
  vol point), and per **year** of calendar time.
* ``charm`` and ``color`` are derivatives with respect to *calendar time*
  ``t``, i.e. ``d/dt = -d/dtau``, matching spec 7.2. A long option's gamma and
  delta therefore decay in the direction these Greeks report.

All functions accept scalars or NumPy arrays. The array paths matter: the
gamma-profile engine reprices the entire chain across a spot grid, which is the
hottest loop in the analytics layer.
"""

from __future__ import annotations

from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import ndtr

from spy_der.domain.enums import OptionRight

FloatArray: TypeAlias = NDArray[np.float64]

#: Below these thresholds the model degenerates and we fall back to intrinsic.
MIN_TAU: Final[float] = 1e-9
MIN_VOL: Final[float] = 1e-8

_SQRT_2PI: Final[float] = float(np.sqrt(2.0 * np.pi))


def _phi(x: FloatArray) -> FloatArray:
    """Standard normal PDF."""
    return np.asarray(np.exp(-0.5 * x * x) / _SQRT_2PI, dtype=np.float64)


def _N(x: FloatArray) -> FloatArray:
    """Standard normal CDF.

    Uses ``scipy.special.ndtr`` rather than ``scipy.stats.norm.cdf``: it is the
    same function without the distribution-object dispatch overhead, and this
    is the hot path in path simulation, which reprices structures at every step
    of every path.
    """
    return np.asarray(ndtr(x), dtype=np.float64)


def _as_array(value: float | FloatArray) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


def d1_d2(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(d1, d2)`` with degenerate inputs clamped rather than raising.

    Clamping instead of raising is deliberate: a live 0DTE chain contains
    contracts with seconds of life and quotes that imply zero volatility, and
    an exception there would take down the whole analytics pass.
    """
    s = _as_array(spot)
    k = _as_array(strike)
    t = np.maximum(_as_array(tau), MIN_TAU)
    v = np.maximum(_as_array(vol), MIN_VOL)
    r = _as_array(rate)
    q = _as_array(div_yield)

    vol_sqrt_t = v * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return np.asarray(d1, dtype=np.float64), np.asarray(d2, dtype=np.float64)


def _is_call(right: OptionRight | FloatArray) -> FloatArray:
    """Broadcastable ``+1`` for calls, ``-1`` for puts."""
    if isinstance(right, OptionRight):
        return _as_array(1.0 if right is OptionRight.CALL else -1.0)
    return _as_array(right)


def intrinsic_value(
    spot: float | FloatArray, strike: float | FloatArray, right: OptionRight | FloatArray
) -> FloatArray:
    sign = _is_call(right)
    return np.asarray(np.maximum(0.0, sign * (_as_array(spot) - _as_array(strike))), dtype=np.float64)


def price(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
    right: OptionRight | FloatArray,
) -> FloatArray:
    """Black-Scholes-Merton price with continuous dividend yield (spec 7.1)."""
    s = _as_array(spot)
    k = _as_array(strike)
    t = _as_array(tau)
    r = _as_array(rate)
    q = _as_array(div_yield)
    v = _as_array(vol)
    sign = _is_call(right)

    d_1, d_2 = d1_d2(s, k, t, r, q, v)
    disc_s = s * np.exp(-q * np.maximum(t, 0.0))
    disc_k = k * np.exp(-r * np.maximum(t, 0.0))
    value = sign * (disc_s * _N(sign * d_1) - disc_k * _N(sign * d_2))

    expired = (_as_array(t) <= MIN_TAU) | (_as_array(v) <= MIN_VOL)
    return np.asarray(np.where(expired, intrinsic_value(s, k, right), value), dtype=np.float64)


def delta(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
    right: OptionRight | FloatArray,
) -> FloatArray:
    """``Delta_C = e^{-qT} N(d1)``; ``Delta_P = e^{-qT}(N(d1) - 1)`` (spec 7.2)."""
    t = _as_array(tau)
    q = _as_array(div_yield)
    sign = _is_call(right)
    d_1, _ = d1_d2(spot, strike, tau, rate, div_yield, vol)
    value = sign * np.exp(-q * np.maximum(t, 0.0)) * _N(sign * d_1)

    # At expiry delta is the step function; ties at the strike resolve to zero.
    expired = (t <= MIN_TAU) | (_as_array(vol) <= MIN_VOL)
    itm = sign * (_as_array(spot) - _as_array(strike)) > 0.0
    return np.asarray(np.where(expired, np.where(itm, sign, 0.0), value), dtype=np.float64)


def gamma(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
) -> FloatArray:
    """``Gamma = e^{-qT} phi(d1) / (S sigma sqrt(T))`` (spec 7.2).

    Gamma is right-independent, which the property tests rely on.
    """
    s = _as_array(spot)
    t = _as_array(tau)
    v = _as_array(vol)
    q = _as_array(div_yield)
    d_1, _ = d1_d2(spot, strike, tau, rate, div_yield, vol)
    safe_t = np.maximum(t, MIN_TAU)
    safe_v = np.maximum(v, MIN_VOL)
    value = np.exp(-q * safe_t) * _phi(d_1) / (s * safe_v * np.sqrt(safe_t))
    expired = (t <= MIN_TAU) | (v <= MIN_VOL)
    return np.asarray(np.where(expired, 0.0, value), dtype=np.float64)


def vega(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
) -> FloatArray:
    """``Vega = S e^{-qT} phi(d1) sqrt(T)`` per 1.0 of volatility (spec 7.2)."""
    s = _as_array(spot)
    t = _as_array(tau)
    q = _as_array(div_yield)
    d_1, _ = d1_d2(spot, strike, tau, rate, div_yield, vol)
    safe_t = np.maximum(t, MIN_TAU)
    value = s * np.exp(-q * safe_t) * _phi(d_1) * np.sqrt(safe_t)
    expired = (t <= MIN_TAU) | (_as_array(vol) <= MIN_VOL)
    return np.asarray(np.where(expired, 0.0, value), dtype=np.float64)


def theta(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
    right: OptionRight | FloatArray,
) -> FloatArray:
    """``dV/dt`` per year of calendar time (negative for most long options)."""
    s = _as_array(spot)
    k = _as_array(strike)
    t = _as_array(tau)
    r = _as_array(rate)
    q = _as_array(div_yield)
    v = _as_array(vol)
    sign = _is_call(right)

    d_1, d_2 = d1_d2(spot, strike, tau, rate, div_yield, vol)
    safe_t = np.maximum(t, MIN_TAU)
    safe_v = np.maximum(v, MIN_VOL)

    decay = -(s * np.exp(-q * safe_t) * _phi(d_1) * safe_v) / (2.0 * np.sqrt(safe_t))
    carry = sign * q * s * np.exp(-q * safe_t) * _N(sign * d_1)
    discount = -sign * r * k * np.exp(-r * safe_t) * _N(sign * d_2)
    value = decay + carry + discount

    expired = (t <= MIN_TAU) | (v <= MIN_VOL)
    return np.asarray(np.where(expired, 0.0, value), dtype=np.float64)


def vanna(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
) -> FloatArray:
    """``Vanna = dDelta/dsigma = -e^{-qT} phi(d1) d2 / sigma`` (spec 7.2)."""
    t = _as_array(tau)
    v = _as_array(vol)
    q = _as_array(div_yield)
    d_1, d_2 = d1_d2(spot, strike, tau, rate, div_yield, vol)
    safe_v = np.maximum(v, MIN_VOL)
    value = -np.exp(-q * np.maximum(t, MIN_TAU)) * _phi(d_1) * d_2 / safe_v
    expired = (t <= MIN_TAU) | (v <= MIN_VOL)
    return np.asarray(np.where(expired, 0.0, value), dtype=np.float64)


def vomma(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
) -> FloatArray:
    """``Vomma = dVega/dsigma = Vega * d1 d2 / sigma`` (spec 7.2)."""
    t = _as_array(tau)
    v = _as_array(vol)
    d_1, d_2 = d1_d2(spot, strike, tau, rate, div_yield, vol)
    safe_v = np.maximum(v, MIN_VOL)
    value = vega(spot, strike, tau, rate, div_yield, vol) * d_1 * d_2 / safe_v
    expired = (t <= MIN_TAU) | (v <= MIN_VOL)
    return np.asarray(np.where(expired, 0.0, value), dtype=np.float64)


def charm(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
    right: OptionRight | FloatArray,
) -> FloatArray:
    """``Charm = dDelta/dt = -dDelta/dtau`` (spec 7.2)."""
    t = _as_array(tau)
    r = _as_array(rate)
    q = _as_array(div_yield)
    v = _as_array(vol)
    sign = _is_call(right)

    d_1, d_2 = d1_d2(spot, strike, tau, rate, div_yield, vol)
    safe_t = np.maximum(t, MIN_TAU)
    safe_v = np.maximum(v, MIN_VOL)
    vol_sqrt_t = safe_v * np.sqrt(safe_t)

    shared = -np.exp(-q * safe_t) * _phi(d_1) * (
        2.0 * (r - q) * safe_t - d_2 * vol_sqrt_t
    ) / (2.0 * safe_t * vol_sqrt_t)
    carry = sign * q * np.exp(-q * safe_t) * _N(sign * d_1)
    value = shared + carry

    expired = (t <= MIN_TAU) | (v <= MIN_VOL)
    return np.asarray(np.where(expired, 0.0, value), dtype=np.float64)


def speed(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
) -> FloatArray:
    """``Speed = dGamma/dS = -(Gamma / S)(d1 / (sigma sqrt(T)) + 1)`` (spec 7.2)."""
    s = _as_array(spot)
    t = _as_array(tau)
    v = _as_array(vol)
    d_1, _ = d1_d2(spot, strike, tau, rate, div_yield, vol)
    safe_t = np.maximum(t, MIN_TAU)
    safe_v = np.maximum(v, MIN_VOL)
    g = gamma(spot, strike, tau, rate, div_yield, vol)
    value = -(g / s) * (d_1 / (safe_v * np.sqrt(safe_t)) + 1.0)
    expired = (t <= MIN_TAU) | (v <= MIN_VOL)
    return np.asarray(np.where(expired, 0.0, value), dtype=np.float64)


def color(
    spot: float | FloatArray,
    strike: float | FloatArray,
    tau: float | FloatArray,
    rate: float | FloatArray,
    div_yield: float | FloatArray,
    vol: float | FloatArray,
) -> FloatArray:
    """``Color = dGamma/dt = -dGamma/dtau`` (spec 7.2)."""
    s = _as_array(spot)
    t = _as_array(tau)
    r = _as_array(rate)
    q = _as_array(div_yield)
    v = _as_array(vol)

    d_1, d_2 = d1_d2(spot, strike, tau, rate, div_yield, vol)
    safe_t = np.maximum(t, MIN_TAU)
    safe_v = np.maximum(v, MIN_VOL)
    vol_sqrt_t = safe_v * np.sqrt(safe_t)

    bracket = 2.0 * q * safe_t + 1.0 + d_1 * (
        2.0 * (r - q) * safe_t - d_2 * vol_sqrt_t
    ) / vol_sqrt_t
    value = np.exp(-q * safe_t) * _phi(d_1) / (2.0 * s * safe_t * vol_sqrt_t) * bracket

    expired = (t <= MIN_TAU) | (v <= MIN_VOL)
    return np.asarray(np.where(expired, 0.0, value), dtype=np.float64)


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    div_yield: float,
    right: OptionRight,
    *,
    low: float = 1e-4,
    high: float = 6.0,
    tolerance: float = 1e-8,
) -> float | None:
    """Invert Black-Scholes for volatility, or ``None`` when no root exists.

    Returning ``None`` rather than a clamped guess is intentional. A price
    outside the no-arbitrage band is a data-quality event that the validator
    must see, not something to paper over with a fabricated volatility.
    """
    if tau <= MIN_TAU or market_price <= 0.0:
        return None

    disc_s = spot * np.exp(-div_yield * tau)
    disc_k = strike * np.exp(-rate * tau)
    lower_bound = max(0.0, disc_s - disc_k) if right is OptionRight.CALL else max(0.0, disc_k - disc_s)
    upper_bound = disc_s if right is OptionRight.CALL else disc_k
    if market_price < lower_bound - tolerance or market_price > upper_bound + tolerance:
        return None

    def objective(sigma: float) -> float:
        return float(price(spot, strike, tau, rate, div_yield, sigma, right)) - market_price

    f_low, f_high = objective(low), objective(high)
    if f_low > 0.0:
        return low
    if f_high < 0.0:
        return None
    try:
        return float(brentq(objective, low, high, xtol=tolerance, maxiter=200))
    except (ValueError, RuntimeError):  # pragma: no cover - brentq guards above
        return None


def put_call_parity_gap(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    div_yield: float,
) -> float:
    """``(C - P) - (S e^{-qT} - K e^{-rT})``; zero under no-arbitrage (spec 37.1)."""
    forward = spot * np.exp(-div_yield * tau) - strike * np.exp(-rate * tau)
    return float(call_price - put_price - forward)
