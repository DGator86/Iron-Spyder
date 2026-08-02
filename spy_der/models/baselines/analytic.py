"""Closed-form GBM baselines (spec section 15).

These are exact first-passage and terminal results for geometric Brownian
motion. They serve two purposes:

1. **Priors.** Before any model is fitted, these give the forecast engine
   calibrated-by-construction probabilities, so the system is functional (and
   honest) from the first snapshot.
2. **Oracles.** The Monte Carlo path simulator must reproduce them to within
   sampling error. That is how the simulator gets tested without testing it
   against itself.

Everything is expressed in log space. With ``X_t = ln(S_t / S_0)`` the process
is arithmetic Brownian motion with drift ``mu = m - sigma^2 / 2`` and volatility
``sigma``, and the reflection principle applies directly.
"""

from __future__ import annotations

import math

from scipy.stats import norm

MIN_TAU = 1e-12
MIN_VOL = 1e-9


def _params(tau: float, vol: float, drift: float) -> tuple[float, float, float]:
    """Return ``(mu, sigma, sqrt_t)`` for the log process."""
    t = max(tau, MIN_TAU)
    sigma = max(vol, MIN_VOL)
    mu = drift - 0.5 * sigma * sigma
    return mu, sigma, math.sqrt(t)


def finish_probability_above(
    spot: float, strike: float, tau: float, vol: float, drift: float = 0.0
) -> float:
    """``P(S_T >= K)`` under GBM."""
    if spot <= 0.0 or strike <= 0.0:
        return 0.0
    mu, sigma, sqrt_t = _params(tau, vol, drift)
    b = math.log(strike / spot)
    return float(norm.cdf((mu * max(tau, MIN_TAU) - b) / (sigma * sqrt_t)))


def finish_probability_below(
    spot: float, strike: float, tau: float, vol: float, drift: float = 0.0
) -> float:
    return 1.0 - finish_probability_above(spot, strike, tau, vol, drift)


def touch_probability_up(
    spot: float, barrier: float, tau: float, vol: float, drift: float = 0.0
) -> float:
    """``P(max_{u<=T} S_u >= B)`` for an upper barrier (spec 15.5)."""
    if barrier <= spot:
        return 1.0
    mu, sigma, sqrt_t = _params(tau, vol, drift)
    t = max(tau, MIN_TAU)
    b = math.log(barrier / spot)

    first = norm.cdf((mu * t - b) / (sigma * sqrt_t))
    exponent = 2.0 * mu * b / (sigma * sigma)
    # Guard the reflection term against overflow for distant barriers.
    if exponent > 700.0:  # pragma: no cover - numerically saturated
        return 1.0
    second = math.exp(exponent) * norm.cdf((-b - mu * t) / (sigma * sqrt_t))
    return float(min(1.0, max(0.0, first + second)))


def touch_probability_down(
    spot: float, barrier: float, tau: float, vol: float, drift: float = 0.0
) -> float:
    """``P(min_{u<=T} S_u <= B)`` for a lower barrier (spec 15.5)."""
    if barrier >= spot:
        return 1.0
    mu, sigma, sqrt_t = _params(tau, vol, drift)
    t = max(tau, MIN_TAU)
    b = math.log(barrier / spot)  # negative

    first = norm.cdf((b - mu * t) / (sigma * sqrt_t))
    exponent = 2.0 * mu * b / (sigma * sigma)
    if exponent > 700.0:  # pragma: no cover - numerically saturated
        return 1.0
    second = math.exp(exponent) * norm.cdf((b + mu * t) / (sigma * sqrt_t))
    return float(min(1.0, max(0.0, first + second)))


def no_touch_probability(
    spot: float,
    lower: float,
    upper: float,
    tau: float,
    vol: float,
    drift: float = 0.0,
    *,
    terms: int = 64,
) -> float:
    """``P(lower < S_u < upper for all u <= T)`` — double-barrier survival.

    Solved by the absorbing-boundary eigenfunction expansion of the heat
    equation, which converges geometrically: each term carries
    ``exp(-n^2 pi^2 sigma^2 T / (2 L^2))``, so a handful of terms is already at
    machine precision for realistic barrier separations.

    The expansion is driftless. Over the horizons this system forecasts — five
    minutes to one session — the drift contribution to a barrier probability is
    several orders of magnitude below the diffusion term, and the drift-aware
    number is available from the path simulator, which is what spec 20 uses for
    range probabilities anyway. ``drift`` is accepted for interface symmetry and
    applied only as a first-order shift of the barriers.
    """
    if not 0.0 < lower < spot < upper:
        return 0.0

    mu, sigma, _ = _params(tau, vol, drift)
    t = max(tau, MIN_TAU)

    # First-order drift handling: shift the barriers by the expected log drift.
    a = math.log(lower / spot) - mu * t
    b = math.log(upper / spot) - mu * t
    if not a < 0.0 < b:
        return 0.0

    width = b - a
    start = -a  # distance from the lower barrier
    decay = (math.pi * sigma) ** 2 * t / (2.0 * width * width)

    total = 0.0
    for n in range(1, terms + 1, 2):  # even terms vanish
        exponent = -(n * n) * decay
        if exponent < -700.0:
            break
        total += (4.0 / (n * math.pi)) * math.sin(n * math.pi * start / width) * math.exp(
            exponent
        )
    return float(min(1.0, max(0.0, total)))


def range_containment_probability(
    spot: float, lower: float, upper: float, tau: float, vol: float, drift: float = 0.0
) -> float:
    """``P(K_L <= S_T <= K_U)`` at the terminal date only (spec 15.4).

    This is the *finish* probability, not the no-touch probability: a structure
    held to expiration cares only where price ends, while a stop-out cares
    whether a barrier was ever breached. Use :func:`no_touch_probability` for
    the latter.
    """
    return max(
        0.0,
        finish_probability_above(spot, lower, tau, vol, drift)
        - finish_probability_above(spot, upper, tau, vol, drift),
    )


def pin_probability(
    spot: float, pin: float, epsilon: float, tau: float, vol: float, drift: float = 0.0
) -> float:
    """``P(|S_T - K_pin| <= epsilon)`` (spec 15.3)."""
    if epsilon <= 0.0:
        return 0.0
    return range_containment_probability(spot, pin - epsilon, pin + epsilon, tau, vol, drift)


def return_quantile(tau: float, vol: float, quantile: float, drift: float = 0.0) -> float:
    """Log-return quantile ``Q_tau(r)`` under GBM (spec 15.6)."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    mu, sigma, sqrt_t = _params(tau, vol, drift)
    return float(mu * max(tau, MIN_TAU) + sigma * sqrt_t * norm.ppf(quantile))


def price_quantile(
    spot: float, tau: float, vol: float, quantile: float, drift: float = 0.0
) -> float:
    """Price quantile implied by :func:`return_quantile`."""
    return spot * math.exp(return_quantile(tau, vol, quantile, drift))


def expected_max_excursion(
    spot: float, tau: float, vol: float, drift: float = 0.0, *, upward: bool = True
) -> float:
    """Expected maximum favourable or adverse excursion, in price terms.

    For driftless Brownian motion ``E[max |X|] = sigma sqrt(2T/pi)``; the drift
    term is added linearly, which is a good approximation over the short
    horizons this system forecasts.
    """
    mu, sigma, sqrt_t = _params(tau, vol, drift)
    t = max(tau, MIN_TAU)
    magnitude = sigma * sqrt_t * math.sqrt(2.0 / math.pi)
    shift = mu * t
    log_excursion = magnitude + shift if upward else -magnitude + shift
    return spot * (math.exp(log_excursion) - 1.0)
