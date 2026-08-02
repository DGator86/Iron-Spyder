"""Model-based marking of a structure (spec section 23, "mark-to-market value").

``V_S = sum_i q_i V_i(S_t, IV_t, tau_t)``. Prices are per share of SPY and
signed by leg quantity, so a debit structure marks positive and a credit
structure marks negative. That sign convention lines up with
:attr:`spy_der.domain.strategy.Strategy.net_entry_cost` and with
:attr:`spy_der.domain.execution.Position.unrealized_pnl`, which is what keeps
P&L arithmetic consistent from candidate generation through to the fill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from spy_der.analytics import pricing
from spy_der.domain.enums import CONTRACT_MULTIPLIER
from spy_der.domain.strategy import OptionLeg, Strategy, StrategyGreeks
from spy_der.strategies.payoff import PayoffProfile, payoff_profile

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def leg_tau(leg: OptionLeg, now: datetime) -> float:
    return max(0.0, (leg.expiry - now).total_seconds()) / SECONDS_PER_YEAR


def structure_price(
    strategy: Strategy,
    spot: float,
    now: datetime,
    *,
    iv_scale: float = 1.0,
    iv_shift: float = 0.0,
    rate: float = 0.04,
    div_yield: float = 0.013,
) -> float:
    """Per-share net value of the structure at ``(spot, now)``."""
    total = 0.0
    for leg in strategy.legs:
        tau = leg_tau(leg, now)
        iv = max(1e-4, leg.entry_iv * iv_scale + iv_shift)
        total += leg.quantity * float(
            pricing.price(spot, leg.strike, tau, rate, div_yield, iv, leg.right)
        )
    return total


def entry_price_per_share(strategy: Strategy) -> float:
    """Net entry cost expressed per share, matching :func:`structure_price`."""
    return strategy.net_entry_cost / CONTRACT_MULTIPLIER


def structure_pnl(
    strategy: Strategy,
    spot: float,
    now: datetime,
    *,
    iv_scale: float = 1.0,
    iv_shift: float = 0.0,
    rate: float = 0.04,
    div_yield: float = 0.013,
) -> float:
    """Dollar mark-to-market P&L per contract."""
    value = structure_price(
        strategy, spot, now, iv_scale=iv_scale, iv_shift=iv_shift, rate=rate, div_yield=div_yield
    )
    return (value - entry_price_per_share(strategy)) * CONTRACT_MULTIPLIER


def structure_price_grid(
    strategy: Strategy,
    spots: NDArray[np.float64],
    now: datetime,
    *,
    iv_scales: NDArray[np.float64] | float = 1.0,
    rate: float = 0.04,
    div_yield: float = 0.013,
) -> NDArray[np.float64]:
    """Vectorized marking across a grid of spots and IV scales.

    Used by the path simulator, which marks every structure at every step of
    every path; the scalar path would be prohibitively slow there.
    """
    spots_array = np.asarray(spots, dtype=np.float64)
    scales = np.broadcast_to(np.asarray(iv_scales, dtype=np.float64), spots_array.shape)

    total = np.zeros_like(spots_array)
    for leg in strategy.legs:
        tau = leg_tau(leg, now)
        iv = np.maximum(1e-4, leg.entry_iv * scales)
        total += leg.quantity * pricing.price(
            spots_array, leg.strike, tau, rate, div_yield, iv, leg.right
        )
    return total


#: Calendar days per year, for converting per-year theta to the per-day
#: convention position Greeks are quoted and risk-limited in.
DAYS_PER_YEAR = 365.0

#: Volatility points per unit of volatility, for the per-point vega convention.
VOL_POINTS = 100.0


def structure_greeks(
    strategy: Strategy,
    spot: float,
    now: datetime,
    *,
    rate: float = 0.04,
    div_yield: float = 0.013,
) -> StrategyGreeks:
    """Position Greeks per contract, in **trading conventions**.

    The pricing layer works in per-year, per-unit-of-volatility terms because
    that is what the closed forms and the finite-difference tests require. Risk
    limits and dashboards are stated the way desks quote them, so the
    conversion happens here, once:

    * ``theta`` is dollars per **calendar day**, not per year. A 0DTE straddle
      has an annualized theta in the hundreds of thousands of dollars; compared
      against a per-day limit that number vetoes every short-dated structure
      for no reason.
    * ``vega`` and ``vanna`` are dollars per **volatility point** (1%), not per
      1.0 of volatility.
    * ``delta`` and ``gamma`` are already dollars per point of SPY.
    """
    delta = gamma = theta = vega = vanna = charm = 0.0
    for leg in strategy.legs:
        tau = leg_tau(leg, now)
        iv = max(1e-4, leg.entry_iv)
        scale = leg.quantity * CONTRACT_MULTIPLIER
        delta += scale * float(pricing.delta(spot, leg.strike, tau, rate, div_yield, iv, leg.right))
        gamma += scale * float(pricing.gamma(spot, leg.strike, tau, rate, div_yield, iv))
        theta += scale * float(pricing.theta(spot, leg.strike, tau, rate, div_yield, iv, leg.right))
        vega += scale * float(pricing.vega(spot, leg.strike, tau, rate, div_yield, iv))
        vanna += scale * float(pricing.vanna(spot, leg.strike, tau, rate, div_yield, iv))
        charm += scale * float(pricing.charm(spot, leg.strike, tau, rate, div_yield, iv, leg.right))

    return StrategyGreeks(
        delta=delta,
        gamma=gamma,
        theta=theta / DAYS_PER_YEAR,
        vega=vega / VOL_POINTS,
        vanna=vanna / VOL_POINTS,
        charm=charm / DAYS_PER_YEAR,
    )


@dataclass(frozen=True)
class ModelProfitProfile:
    """Profit geometry for structures whose diagram is model dependent."""

    max_profit: float
    best_spot: float
    evaluated_at: datetime
    breakevens: tuple[float, ...]


def model_profit_profile(
    strategy: Strategy,
    *,
    spot: float,
    rate: float = 0.04,
    div_yield: float = 0.013,
    width: float = 0.08,
    points: int = 161,
) -> ModelProfitProfile:
    """Profit geometry of a calendarized structure at its near expiration.

    :func:`spy_der.strategies.payoff.payoff_profile` leaves ``max_profit`` as
    NaN for multi-expiry structures because valuing a surviving far leg at
    intrinsic is wrong. Here the far legs are priced properly at the moment the
    near leg settles, one second past the near expiration so the expiring legs
    are unambiguously worthless.
    """
    evaluate_at = strategy.near_expiry + timedelta(seconds=1)
    spots = np.linspace(spot * (1.0 - width), spot * (1.0 + width), points)

    values = structure_price_grid(strategy, spots, evaluate_at, rate=rate, div_yield=div_yield)
    pnl = (values - entry_price_per_share(strategy)) * CONTRACT_MULTIPLIER

    best_index = int(np.argmax(pnl))
    return ModelProfitProfile(
        max_profit=float(pnl[best_index]),
        best_spot=float(spots[best_index]),
        evaluated_at=evaluate_at,
        breakevens=_zero_crossings(spots, pnl),
    )


def _zero_crossings(
    spots: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[float, ...]:
    out: list[float] = []
    for i in range(len(spots) - 1):
        v0, v1 = float(values[i]), float(values[i + 1])
        if v0 * v1 < 0.0:
            s0, s1 = float(spots[i]), float(spots[i + 1])
            out.append(s0 + (s1 - s0) * (-v0) / (v1 - v0))
    return tuple(out)


def resolved_profit(
    strategy: Strategy,
    profile: PayoffProfile,
    *,
    spot: float,
    rate: float = 0.04,
    div_yield: float = 0.013,
) -> float:
    """Maximum profit with the calendar case filled in.

    Returns a finite number for every admissible structure: the terminal
    diagram value when it exists, the model value for calendars, and a
    forecast-free practical bound for unbounded structures.
    """
    if profile.profit_is_model_dependent:
        return model_profit_profile(
            strategy, spot=spot, rate=rate, div_yield=div_yield
        ).max_profit
    if math.isinf(profile.max_profit):
        # Unbounded upside: report the value at a practical bound rather than
        # infinity, so ranking and capture ratios stay well defined.
        return profile.max_profit_at_bound
    return profile.max_profit


def full_profile(
    strategy: Strategy, *, spot: float, rate: float = 0.04, div_yield: float = 0.013
) -> tuple[PayoffProfile, float]:
    """Return ``(payoff_profile, resolved_max_profit)`` in one call."""
    profile = payoff_profile(strategy)
    return profile, resolved_profit(
        strategy, profile, spot=spot, rate=rate, div_yield=div_yield
    )
