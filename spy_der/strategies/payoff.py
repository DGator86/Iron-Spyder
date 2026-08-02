"""Deterministic payoff engine (spec section 23).

Two payoff notions are computed and kept strictly separate:

**Terminal payoff** values every leg at intrinsic on the structure's final
expiration. This is the classic payoff diagram and is what breakevens and
maximum profit are read from.

**Conservative payoff** values the structure at an intermediate expiration
checkpoint using worst-case bounds for legs that have not expired yet:

* a long unexpired leg is floored at intrinsic (an option is never worth less);
* a short unexpired leg is capped at ``S`` for a call and ``K`` for a put (an
  option is never worth more).

Because both are one-sided bounds in the direction that hurts the position, the
resulting minimum is a genuine lower bound on P&L, not an estimate. That is what
makes ``max_loss`` safe to hand to the risk engine for a calendar or diagonal,
where a naive "value everything at intrinsic" calculation would understate risk.

The payoff is piecewise linear in ``S`` with kinks only at strikes, so
evaluating at ``{0} + strikes + far_bound`` and inspecting the asymptotic slopes
locates the exact extrema. No sampling grid and no magic large-number probe is
involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from spy_der.domain.enums import CONTRACT_MULTIPLIER, OptionRight
from spy_der.domain.strategy import OptionLeg, Strategy


@dataclass(frozen=True, slots=True)
class PayoffProfile:
    """Complete deterministic risk geometry of a structure, per one contract."""

    max_loss: float
    max_profit: float
    max_profit_at_bound: float
    breakevens: tuple[float, ...]
    upside_slope: float
    downside_slope: float
    net_entry_cost: float
    worst_case_spot: float
    worst_case_expiry: datetime
    profit_is_unbounded: bool
    profit_is_model_dependent: bool = False
    """True for calendarized structures.

    Once a near leg expires, the far leg's worth is a model quantity rather
    than an intrinsic one, so ``max_profit`` and ``breakevens`` are left as NaN
    and empty here and are filled in by
    :func:`spy_der.strategies.metrics.model_profit_profile`, which prices the
    surviving legs. ``max_loss`` remains exact and is unaffected.
    """

    @property
    def is_bounded_loss(self) -> bool:
        return math.isfinite(self.max_loss)

    @property
    def is_credit(self) -> bool:
        return self.net_entry_cost < 0.0

    @property
    def risk_reward(self) -> float:
        if self.max_loss <= 0.0:
            return float("inf")
        return self.max_profit_at_bound / self.max_loss


def leg_value_bound(leg: OptionLeg, spot: float, at_expiry: datetime, *, conservative: bool) -> float:
    """Per-share value of ``leg`` at ``spot`` on ``at_expiry``.

    With ``conservative=True`` the value is biased against the holder of the
    leg, giving a one-sided bound suitable for maximum-loss proofs.
    """
    if leg.expiry <= at_expiry:
        return leg.intrinsic(spot)
    if not conservative or leg.is_long:
        # An unexpired long leg is worth at least its intrinsic value.
        return leg.intrinsic(spot)
    # An unexpired short leg is worth at most the underlying (call) or the
    # strike (put); both are undiscounted, which only strengthens the bound.
    return spot if leg.right is OptionRight.CALL else leg.strike


def payoff_at(
    strategy: Strategy,
    spot: float,
    *,
    at_expiry: datetime | None = None,
    conservative: bool = False,
) -> float:
    """Dollar P&L per contract at ``spot``, net of the entry cost."""
    checkpoint = at_expiry if at_expiry is not None else strategy.far_expiry
    gross = sum(
        leg.quantity
        * CONTRACT_MULTIPLIER
        * leg_value_bound(leg, spot, checkpoint, conservative=conservative)
        for leg in strategy.legs
    )
    return gross - strategy.net_entry_cost


def terminal_payoff(strategy: Strategy, spot: float) -> float:
    """Payoff on the final expiration with all legs at intrinsic."""
    return payoff_at(strategy, spot, at_expiry=strategy.far_expiry, conservative=False)


def tail_slopes(strategy: Strategy) -> tuple[float, float]:
    """Return ``(upside_slope, downside_slope)`` in dollars per point of SPY.

    ``upside_slope`` is the asymptotic slope as ``S -> inf`` and is the single
    quantity that decides whether loss is bounded: every call contributes
    ``+1`` per unit of signed quantity regardless of whether it is valued at
    intrinsic or at its short-leg upper bound ``S``.

    ``downside_slope`` is the slope as ``S -> 0+``. Puts cannot pay more than
    their strike, so the downside is always finite for equity options; the
    slope is reported for diagnostics and for detecting unbounded *profit*.
    """
    upside = sum(
        leg.quantity for leg in strategy.legs if leg.right is OptionRight.CALL
    ) * CONTRACT_MULTIPLIER
    downside = -sum(
        leg.quantity for leg in strategy.legs if leg.right is OptionRight.PUT
    ) * CONTRACT_MULTIPLIER
    return float(upside), float(downside)


def evaluation_spots(strategy: Strategy, *, far_bound: float | None = None) -> tuple[float, ...]:
    """Kink points plus endpoints — the exact extrema candidates."""
    strikes = strategy.strikes
    bound = far_bound if far_bound is not None else max(2.0 * strikes[-1], strikes[-1] + 100.0)
    return (0.0, *strikes, bound)


def payoff_profile(strategy: Strategy, *, far_bound: float | None = None) -> PayoffProfile:
    """Compute the deterministic risk geometry of ``strategy``.

    ``max_loss`` is the worst conservative payoff across every expiration
    checkpoint, so a calendar is measured at the moment its short leg expires
    rather than only at the far leg's expiration.
    """
    upside_slope, downside_slope = tail_slopes(strategy)
    spots = evaluation_spots(strategy, far_bound=far_bound)

    worst = math.inf
    worst_spot = 0.0
    worst_expiry = strategy.near_expiry
    for checkpoint in strategy.expiries:
        for spot in spots:
            value = payoff_at(strategy, spot, at_expiry=checkpoint, conservative=True)
            if value < worst:
                worst, worst_spot, worst_expiry = value, spot, checkpoint

    # A negative upside slope means the conservative payoff falls without limit
    # as SPY rises; no finite evaluation point can capture that.
    max_loss = math.inf if upside_slope < 0.0 else max(0.0, -worst)

    profit_unbounded = upside_slope > 0.0

    if strategy.is_multi_expiry:
        # The terminal diagram is not defined for a structure whose legs settle
        # on different dates; see PayoffProfile.profit_is_model_dependent.
        max_profit = math.nan
        best_at_bound = math.nan
        bes: tuple[float, ...] = ()
    else:
        best_at_bound = max(terminal_payoff(strategy, s) for s in spots)
        max_profit = math.inf if profit_unbounded else best_at_bound
        bes = breakevens(strategy)

    return PayoffProfile(
        max_loss=max_loss,
        max_profit=max_profit,
        max_profit_at_bound=best_at_bound,
        breakevens=bes,
        upside_slope=upside_slope,
        downside_slope=downside_slope,
        net_entry_cost=strategy.net_entry_cost,
        worst_case_spot=worst_spot,
        worst_case_expiry=worst_expiry,
        profit_is_unbounded=profit_unbounded,
        profit_is_model_dependent=strategy.is_multi_expiry,
    )


def breakevens(strategy: Strategy, *, far_bound: float | None = None) -> tuple[float, ...]:
    """Exact terminal breakevens: the boundaries of the profitable region.

    Roots are solved analytically on each linear segment rather than
    approximated by the segment midpoint, so a bull call spread reports its
    true breakeven at ``long strike + debit`` and not the midpoint between
    strikes.

    A kink that merely sits inside a flat zero-P&L region is not a boundary and
    is not reported — a zero-cost backspread breaks even across its entire
    lower range, and listing every strike there as a "breakeven" would be
    noise. Only points where the payoff sign actually changes are returned.
    """
    spots = evaluation_spots(strategy, far_bound=far_bound)
    values = [terminal_payoff(strategy, s) for s in spots]

    def sign_at(spot: float) -> int:
        value = terminal_payoff(strategy, spot)
        if value > 1e-9:
            return 1
        if value < -1e-9:
            return -1
        return 0

    roots: list[float] = []
    for index, ((s0, v0), (s1, v1)) in enumerate(
        zip(zip(spots, values), zip(spots[1:], values[1:]))
    ):
        if v0 * v1 < 0.0:
            roots.append(s0 + (s1 - s0) * (-v0) / (v1 - v0))
            continue
        if abs(v0) <= 1e-9:
            # Report this kink only if the payoff sign genuinely differs across
            # it, probing the interiors of the neighbouring segments.
            left = sign_at(0.5 * (spots[index - 1] + s0)) if index > 0 else None
            right = sign_at(0.5 * (s0 + s1))
            if left is not None and left != right:
                roots.append(s0)

    deduped: list[float] = []
    for r in sorted(roots):
        if not deduped or abs(r - deduped[-1]) > 1e-9:
            deduped.append(r)
    return tuple(deduped)


def payoff_curve(
    strategy: Strategy, low: float, high: float, points: int = 121
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Sampled terminal payoff for charting."""
    if points < 2 or high <= low:
        raise ValueError("require points >= 2 and high > low")
    step = (high - low) / (points - 1)
    xs = tuple(low + i * step for i in range(points))
    return xs, tuple(terminal_payoff(strategy, x) for x in xs)


def probability_of_profit(
    strategy: Strategy, cdf: object, *, far_bound: float | None = None
) -> float:
    """Terminal probability of profit given a price CDF callable.

    ``cdf(x)`` must return ``P(S_T <= x)``. The profitable set of a piecewise
    linear payoff is a union of intervals delimited by the breakevens, so the
    probability is summed exactly over those intervals instead of being
    estimated by sampling.
    """
    if not callable(cdf):
        raise TypeError("cdf must be callable")
    bes = breakevens(strategy, far_bound=far_bound)
    spots = evaluation_spots(strategy, far_bound=far_bound)
    edges = [0.0, *bes, spots[-1]]

    total = 0.0
    for lo, hi in zip(edges, edges[1:]):
        if hi <= lo:
            continue
        midpoint = 0.5 * (lo + hi)
        if terminal_payoff(strategy, midpoint) > 0.0:
            total += float(cdf(hi)) - float(cdf(lo))
    return min(1.0, max(0.0, total))
