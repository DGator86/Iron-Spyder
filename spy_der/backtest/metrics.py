"""Backtest performance metrics (spec sections 37.4, 38)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BacktestMetrics:
    profile: str
    trade_count: int
    total_pnl: float
    expectancy: float
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_fraction: float
    pnl_std: float
    no_trade_rate: float

    @property
    def is_profitable(self) -> bool:
        return self.expectancy > 0.0

    @property
    def sharpe_like(self) -> float:
        """Per-trade mean over standard deviation.

        Not an annualized Sharpe ratio — trades are irregularly spaced and
        overlapping, so annualizing would imply a sampling frequency this
        system does not have. Useful only for comparing runs.
        """
        return self.expectancy / self.pnl_std if self.pnl_std > 0.0 else 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "profile": self.profile,
            "trades": self.trade_count,
            "total_pnl": self.total_pnl,
            "expectancy": self.expectancy,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_fraction": self.max_drawdown_fraction,
            "sharpe_like": self.sharpe_like,
            "no_trade_rate": self.no_trade_rate,
        }


def summarize(
    *,
    pnl: Sequence[float],
    equity_curve: Sequence[float],
    no_trade_rate: float,
    profile: str = "recorded",
) -> BacktestMetrics:
    values = np.asarray(list(pnl), dtype=np.float64)
    wins = values[values > 0.0]
    losses = values[values < 0.0]
    gross_win = float(np.sum(wins)) if wins.size else 0.0
    gross_loss = float(-np.sum(losses)) if losses.size else 0.0

    drawdown, drawdown_fraction = max_drawdown(equity_curve)

    return BacktestMetrics(
        profile=profile,
        trade_count=int(values.size),
        total_pnl=float(np.sum(values)) if values.size else 0.0,
        expectancy=float(np.mean(values)) if values.size else 0.0,
        win_rate=float(wins.size / values.size) if values.size else 0.0,
        average_win=float(np.mean(wins)) if wins.size else 0.0,
        average_loss=float(np.mean(losses)) if losses.size else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0.0 else float("inf"),
        max_drawdown=drawdown,
        max_drawdown_fraction=drawdown_fraction,
        pnl_std=float(np.std(values)) if values.size > 1 else 0.0,
        no_trade_rate=no_trade_rate,
    )


def max_drawdown(equity_curve: Sequence[float]) -> tuple[float, float]:
    """Return ``(peak-to-trough dollars, fraction of peak)``."""
    values = np.asarray(list(equity_curve), dtype=np.float64)
    if values.size < 2:
        return 0.0, 0.0
    running_peak = np.maximum.accumulate(values)
    drawdowns = running_peak - values
    index = int(np.argmax(drawdowns))
    peak = float(running_peak[index])
    dollars = float(drawdowns[index])
    return dollars, (dollars / peak if peak > 0.0 else 0.0)


def parameter_stability(
    results: dict[float, float], *, tolerance: float = 0.35
) -> dict[str, float | bool]:
    """Assess whether performance sits on a plateau (spec 37.8).

    A single-point optimum is a sign of overfitting; a plateau is acceptable.
    "Plateau" here means the neighbours of the best parameter retain at least
    ``1 - tolerance`` of its performance.
    """
    if len(results) < 3:
        return {"is_plateau": False, "best": 0.0, "neighbour_ratio": 0.0}

    ordered = sorted(results.items())
    values = [v for _, v in ordered]
    best_index = int(np.argmax(values))
    best = values[best_index]
    if best <= 0.0:
        return {"is_plateau": False, "best": float(best), "neighbour_ratio": 0.0}

    neighbours = [
        values[i] for i in (best_index - 1, best_index + 1) if 0 <= i < len(values)
    ]
    ratio = float(np.mean(neighbours) / best) if neighbours else 0.0
    return {
        "is_plateau": bool(ratio >= 1.0 - tolerance),
        "best": float(best),
        "best_parameter": float(ordered[best_index][0]),
        "neighbour_ratio": ratio,
    }
