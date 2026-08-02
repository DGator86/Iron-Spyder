"""Point-in-time backtest engine and cost stress (spec sections 37.4, 37.7).

The backtest drives the *same* :class:`~spy_der.pipeline.DecisionPipeline` the
live path uses. There is no parallel "backtest logic" that could drift from
production behaviour, and the point-in-time guarantee is inherited rather than
re-implemented: the pipeline only ever sees one snapshot.

Option P&L comes from historical option prices throughout — entries, marks, and
exits are all priced off the chain in the snapshot. Nothing here estimates
option P&L from SPY movement, which spec 37.4 explicitly forbids.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np

from spy_der.backtest.metrics import BacktestMetrics, summarize
from spy_der.domain.enums import ExitReason
from spy_der.domain.execution import Position
from spy_der.domain.market import MarketSnapshot
from spy_der.execution.fills import CostModel
from spy_der.pipeline import DecisionPipeline, PipelineResult, force_close_all


@dataclass(frozen=True)
class StressProfile:
    """One cost-stress configuration (spec section 37.7)."""

    name: str
    spread_multiplier: float = 1.0
    adverse_ticks: float = 0.0
    fill_probability_scale: float = 1.0

    def apply(self, costs: CostModel) -> CostModel:
        stressed = costs.stressed(
            spread_multiplier=self.spread_multiplier, adverse_ticks=self.adverse_ticks
        )
        if self.fill_probability_scale != 1.0:
            stressed = replace(
                stressed,
                fill_probability_floor=stressed.fill_probability_floor
                * self.fill_probability_scale,
            )
        return stressed


#: The stress grid required by spec 37.7.
STRESS_PROFILES: tuple[StressProfile, ...] = (
    StressProfile("recorded", 1.00, 0.0),
    StressProfile("spread_1.25x", 1.25, 0.0),
    StressProfile("spread_1.50x", 1.50, 0.0),
    StressProfile("spread_2.00x", 2.00, 0.0),
    StressProfile("one_tick_adverse", 1.00, 1.0),
    StressProfile("two_tick_adverse", 1.00, 2.0),
    StressProfile("spread_1.50x_two_tick", 1.50, 2.0),
    StressProfile("reduced_fill", 1.25, 1.0, fill_probability_scale=0.5),
)


@dataclass
class BacktestResult:
    """Outcome of one backtest run."""

    profile: str
    decisions: list[PipelineResult] = field(default_factory=list)
    closed_positions: list[Position] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def trade_count(self) -> int:
        return len(self.closed_positions)

    @property
    def decision_count(self) -> int:
        return len(self.decisions)

    @property
    def no_trade_rate(self) -> float:
        if not self.decisions:
            return 1.0
        return sum(1 for d in self.decisions if d.is_no_trade) / len(self.decisions)

    @property
    def pnl(self) -> list[float]:
        return [p.realized_pnl for p in self.closed_positions]

    def metrics(self) -> BacktestMetrics:
        return summarize(
            pnl=self.pnl,
            equity_curve=[e for _, e in self.equity_curve],
            no_trade_rate=self.no_trade_rate,
            profile=self.profile,
        )


def run_backtest(
    pipeline: DecisionPipeline,
    snapshots: Iterable[MarketSnapshot],
    *,
    profile: str = "recorded",
    close_at_end: bool = True,
) -> BacktestResult:
    """Replay ``snapshots`` in order through ``pipeline``.

    Snapshots must be chronological. They are consumed one at a time and the
    pipeline is never handed the sequence, so there is no path by which a
    decision can observe a later timestamp.
    """
    result = BacktestResult(profile=profile)
    previous: datetime | None = None
    last: MarketSnapshot | None = None

    for snapshot in snapshots:
        if previous is not None and snapshot.timestamp < previous:
            raise ValueError(
                f"snapshots must be chronological: {snapshot.timestamp} follows {previous}"
            )
        previous = snapshot.timestamp
        last = snapshot

        step = pipeline.step(snapshot)
        result.decisions.append(step)
        result.equity_curve.append((snapshot.timestamp, pipeline.broker.equity))

        for position in pipeline.portfolio.positions:
            if not position.is_open and position not in result.closed_positions:
                result.closed_positions.append(position)

    if close_at_end and last is not None:
        closed = force_close_all(pipeline, last, ExitReason.END_OF_SESSION)
        result.closed_positions.extend(c for c in closed if c not in result.closed_positions)
        result.equity_curve.append((last.timestamp, pipeline.broker.equity))

    return result


def run_cost_stress(
    pipeline_factory: object,
    snapshots: Sequence[MarketSnapshot],
    *,
    profiles: Sequence[StressProfile] = STRESS_PROFILES,
) -> dict[str, BacktestResult]:
    """Run the backtest under each stress profile (spec 37.7).

    ``pipeline_factory`` is a callable taking a :class:`CostModel` and returning
    a fresh :class:`DecisionPipeline`. A factory rather than a single pipeline
    is required: reusing one would carry equity, open positions, and HMM state
    across profiles and make the comparison meaningless.
    """
    if not callable(pipeline_factory):
        raise TypeError("pipeline_factory must be callable")

    results: dict[str, BacktestResult] = {}
    for stress in profiles:
        costs = stress.apply(CostModel())
        pipeline = pipeline_factory(costs)
        results[stress.name] = run_backtest(pipeline, snapshots, profile=stress.name)
    return results


def stress_summary(results: dict[str, BacktestResult]) -> dict[str, dict[str, float]]:
    """Compact comparison across stress profiles."""
    out: dict[str, dict[str, float]] = {}
    for name, result in results.items():
        metrics = result.metrics()
        out[name] = {
            "trades": float(metrics.trade_count),
            "total_pnl": metrics.total_pnl,
            "expectancy": metrics.expectancy,
            "win_rate": metrics.win_rate,
            "max_drawdown": metrics.max_drawdown,
            "no_trade_rate": result.no_trade_rate,
        }
    return out


def survives_stress(
    results: dict[str, BacktestResult], *, require_positive_through: str = "spread_1.50x"
) -> bool:
    """Whether expectancy stays positive through a given stress level.

    Spec 37.7: a strategy that depends on midpoint fills is not production
    ready. Passing means the edge is still there once spreads widen, not merely
    that it existed at recorded prices.
    """
    order = [p.name for p in STRESS_PROFILES]
    if require_positive_through not in order:
        raise ValueError(f"unknown stress profile {require_positive_through!r}")

    limit = order.index(require_positive_through)
    for name in order[: limit + 1]:
        result = results.get(name)
        if result is None:
            return False
        if result.trade_count == 0:
            continue  # no trades is not a failure, it is abstention
        if result.metrics().expectancy <= 0.0:
            return False
    return True


def compare_to_baselines(
    system: BacktestResult, baselines: dict[str, Sequence[float]]
) -> dict[str, dict[str, float]]:
    """Compare system expectancy against baseline strategies (spec 37.10)."""
    system_metrics = system.metrics()
    out: dict[str, dict[str, float]] = {
        "system": {
            "expectancy": system_metrics.expectancy,
            "total_pnl": system_metrics.total_pnl,
            "trades": float(system_metrics.trade_count),
        }
    }
    for name, pnl in baselines.items():
        values = np.asarray(list(pnl), dtype=np.float64)
        expectancy = float(np.mean(values)) if values.size else 0.0
        out[name] = {
            "expectancy": expectancy,
            "total_pnl": float(np.sum(values)) if values.size else 0.0,
            "trades": float(values.size),
            "system_advantage": system_metrics.expectancy - expectancy,
        }
    return out
