"""Path-based exit simulation and expected value (spec sections 20, 26, 30).

Every open strategy must carry a full exit contract, so evaluating a candidate
means evaluating *the strategy plus its exit plan*, not the payoff diagram. Two
candidates with identical terminal payoffs but different stops have different
expected values, and this module is where that difference is priced.

Exits are evaluated in a fixed precedence at each step:

1. assignment / pre-expiry closure (hard, non-negotiable);
2. risk stop;
3. profit target;
4. time stop;
5. horizon end.

The risk stop is checked before the profit target on purpose. When a single
step straddles both thresholds the intra-step path is unknown, and assuming the
favourable one fired first is exactly the optimism that makes a backtest
unreproducible in live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from spy_der.domain.enums import CONTRACT_MULTIPLIER, ExitReason
from spy_der.domain.strategy import ExitPlan, Strategy
from spy_der.simulation.price_paths import PathBundle
from spy_der.strategies.metrics import entry_price_per_share, structure_price_grid

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
FloatArray: TypeAlias = NDArray[np.float64]

#: Order in which exit reasons are resolved when several fire on one step.
REASON_PRECEDENCE: tuple[ExitReason, ...] = (
    ExitReason.ASSIGNMENT_RISK,
    ExitReason.RISK_STOP,
    ExitReason.PROFIT_TARGET,
    ExitReason.TIME_STOP,
    ExitReason.END_OF_SESSION,
)


@dataclass(frozen=True)
class ExitSimulation:
    """Outcome of walking every path under one exit plan."""

    pnl: FloatArray
    """Dollar P&L per contract, per path, net of round-trip costs."""

    exit_step: NDArray[np.int_]
    exit_reasons: tuple[ExitReason, ...]
    """Reason per path, aligned with ``pnl``."""

    holding_minutes: FloatArray
    max_favourable: FloatArray
    max_adverse: FloatArray

    @property
    def expected_value(self) -> float:
        return float(np.mean(self.pnl))

    @property
    def win_rate(self) -> float:
        return float(np.mean(self.pnl > 0.0))

    @property
    def std(self) -> float:
        return float(np.std(self.pnl))

    def cvar(self, alpha: float = 0.95) -> float:
        """``CVaR_alpha = E[Loss | Loss >= VaR_alpha]`` as a positive number."""
        losses = -self.pnl
        if losses.size == 0:
            return 0.0
        threshold = float(np.quantile(losses, alpha))
        tail = losses[losses >= threshold]
        if tail.size == 0:
            return max(0.0, threshold)
        return max(0.0, float(np.mean(tail)))

    def reason_counts(self) -> dict[ExitReason, int]:
        counts: dict[ExitReason, int] = {}
        for reason in self.exit_reasons:
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    @property
    def mean_holding_minutes(self) -> float:
        return float(np.mean(self.holding_minutes))


def simulate_exits(
    strategy: Strategy,
    bundle: PathBundle,
    exit_plan: ExitPlan,
    *,
    entry_time: datetime,
    max_profit: float,
    max_loss: float,
    round_trip_cost: float = 0.0,
    rate: float = 0.04,
    div_yield: float = 0.013,
) -> ExitSimulation:
    """Walk ``bundle`` applying ``exit_plan`` and return per-path outcomes.

    ``max_profit`` and ``max_loss`` are per-contract dollar amounts and set the
    scale for the target and stop fractions. ``round_trip_cost`` is charged once
    per path, covering entry and exit commissions and slippage.
    """
    steps = bundle.n_steps
    entry_per_share = entry_price_per_share(strategy)

    pnl_matrix = np.empty((bundle.n_paths, steps + 1), dtype=np.float64)
    for t in range(steps + 1):
        now = entry_time + timedelta(seconds=float(bundle.times[t]) * SECONDS_PER_YEAR)
        values = structure_price_grid(
            strategy,
            bundle.spot[:, t],
            now,
            iv_scales=bundle.iv_scale[:, t],
            rate=rate,
            div_yield=div_yield,
        )
        pnl_matrix[:, t] = (values - entry_per_share) * CONTRACT_MULTIPLIER

    minutes = bundle.times * 365.0 * 24.0 * 60.0
    target_level = exit_plan.profit_target_fraction * max(max_profit, 1e-9)
    stop_level = -exit_plan.stop_loss_fraction * max(max_loss, 1e-9)

    # Forced-closure step: the earlier of the time stop and the pre-expiry
    # assignment cutoff. Never allow a short leg to run into settlement.
    seconds_to_near_expiry = (strategy.near_expiry - entry_time).total_seconds()
    assignment_cutoff_minutes = (
        seconds_to_near_expiry / 60.0 - exit_plan.close_before_expiry_minutes
    )
    time_stop_minutes = exit_plan.max_holding_minutes

    hit_target = pnl_matrix >= target_level
    hit_stop = pnl_matrix <= stop_level
    # Time-based conditions depend only on the step, not on the path.
    past_assignment = minutes >= assignment_cutoff_minutes
    past_time_stop = minutes >= time_stop_minutes

    triggered = hit_target | hit_stop | past_assignment[None, :] | past_time_stop[None, :]
    triggered[:, -1] = True  # the horizon always closes the position

    exit_step = np.argmax(triggered, axis=1)
    rows = np.arange(bundle.n_paths)
    raw_pnl = pnl_matrix[rows, exit_step]

    reasons = _resolve_reasons(
        exit_step=exit_step,
        rows=rows,
        hit_target=hit_target,
        hit_stop=hit_stop,
        past_assignment=past_assignment,
        past_time_stop=past_time_stop,
        last_step=steps,
    )

    # Excursions are measured only over the realized holding period.
    step_index = np.arange(steps + 1)[None, :]
    held = step_index <= exit_step[:, None]
    masked_gain = np.where(held, pnl_matrix, -np.inf)
    masked_loss = np.where(held, pnl_matrix, np.inf)

    return ExitSimulation(
        pnl=raw_pnl - round_trip_cost,
        exit_step=exit_step,
        exit_reasons=reasons,
        holding_minutes=minutes[exit_step],
        max_favourable=np.max(masked_gain, axis=1),
        max_adverse=np.min(masked_loss, axis=1),
    )


def _resolve_reasons(
    *,
    exit_step: NDArray[np.int_],
    rows: NDArray[np.int_],
    hit_target: NDArray[np.bool_],
    hit_stop: NDArray[np.bool_],
    past_assignment: NDArray[np.bool_],
    past_time_stop: NDArray[np.bool_],
    last_step: int,
) -> tuple[ExitReason, ...]:
    """Attribute each path's exit to the highest-precedence condition that fired.

    ``past_assignment`` and ``past_time_stop`` are indexed by step only; the
    P&L conditions are indexed by ``(path, step)``.
    """
    at_assignment = past_assignment[exit_step]
    at_stop = hit_stop[rows, exit_step]
    at_target = hit_target[rows, exit_step]
    at_time = past_time_stop[exit_step]

    reasons: list[ExitReason] = []
    for index in range(len(exit_step)):
        if at_assignment[index]:
            reasons.append(ExitReason.ASSIGNMENT_RISK)
        elif at_stop[index]:
            reasons.append(ExitReason.RISK_STOP)
        elif at_target[index]:
            reasons.append(ExitReason.PROFIT_TARGET)
        elif at_time[index]:
            reasons.append(ExitReason.TIME_STOP)
        elif exit_step[index] == last_step:
            reasons.append(ExitReason.END_OF_SESSION)
        else:  # pragma: no cover - triggered implies one of the above
            reasons.append(ExitReason.END_OF_SESSION)
    return tuple(reasons)


@dataclass(frozen=True)
class PathValuation:
    """Path-based expected value and risk of a candidate (spec section 26)."""

    expected_value: float
    cvar: float
    win_rate: float
    std: float
    mean_holding_minutes: float
    expected_mfe: float
    expected_mae: float
    reason_counts: dict[ExitReason, int]

    @property
    def sharpe_like(self) -> float:
        """Mean over standard deviation; a scale-free ranking aid, not a Sharpe ratio."""
        return self.expected_value / self.std if self.std > 0.0 else 0.0


def value_candidate(
    strategy: Strategy,
    bundle: PathBundle,
    exit_plan: ExitPlan,
    *,
    entry_time: datetime,
    max_profit: float,
    max_loss: float,
    round_trip_cost: float = 0.0,
    cvar_alpha: float = 0.95,
    rate: float = 0.04,
    div_yield: float = 0.013,
) -> PathValuation:
    """``EV_path(S) = E[PnL(S, Path)] - Costs`` (spec section 20)."""
    outcome = simulate_exits(
        strategy,
        bundle,
        exit_plan,
        entry_time=entry_time,
        max_profit=max_profit,
        max_loss=max_loss,
        round_trip_cost=round_trip_cost,
        rate=rate,
        div_yield=div_yield,
    )
    return PathValuation(
        expected_value=outcome.expected_value,
        cvar=outcome.cvar(cvar_alpha),
        win_rate=outcome.win_rate,
        std=outcome.std,
        mean_holding_minutes=outcome.mean_holding_minutes,
        expected_mfe=float(np.mean(outcome.max_favourable)),
        expected_mae=float(np.mean(outcome.max_adverse)),
        reason_counts=outcome.reason_counts(),
    )
