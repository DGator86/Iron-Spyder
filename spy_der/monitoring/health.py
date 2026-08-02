"""Drift, calibration, and execution-health monitoring (spec section 41).

Monitoring exists to trigger the automatic responses of spec 41 — reduce size,
disable complex strategies, revert to baseline models, go paper-only, or lock
out entirely — so each check returns a severity that maps to one of those
actions rather than a bare number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from spy_der.domain.enums import KillSwitchReason
from spy_der.models.calibration.calibrator import expected_calibration_error


class Response(str, Enum):
    """Automatic responses available to the monitor (spec 41)."""

    NONE = "none"
    REDUCE_SIZE = "reduce_size"
    DISABLE_COMPLEX = "disable_complex_strategies"
    REVERT_TO_BASELINE = "revert_to_baseline_models"
    PAPER_ONLY = "paper_only"
    LOCKOUT = "full_trading_lockout"


@dataclass(frozen=True)
class HealthFinding:
    check: str
    response: Response
    detail: str
    value: float
    threshold: float

    @property
    def is_actionable(self) -> bool:
        return self.response is not Response.NONE


@dataclass(frozen=True)
class HealthReport:
    findings: tuple[HealthFinding, ...]

    @property
    def is_healthy(self) -> bool:
        return not any(f.is_actionable for f in self.findings)

    @property
    def response(self) -> Response:
        """The most severe response any check called for."""
        order = list(Response)
        worst = Response.NONE
        for finding in self.findings:
            if order.index(finding.response) > order.index(worst):
                worst = finding.response
        return worst

    @property
    def kill_switch_reasons(self) -> tuple[KillSwitchReason, ...]:
        mapping = {
            "data_freshness": KillSwitchReason.DATA_QUALITY,
            "calibration": KillSwitchReason.CALIBRATION,
            "slippage_drift": KillSwitchReason.SLIPPAGE,
            "drawdown": KillSwitchReason.DRAWDOWN,
        }
        return tuple(
            mapping[f.check]
            for f in self.findings
            if f.response is Response.LOCKOUT and f.check in mapping
        )

    def summary(self) -> str:
        if self.is_healthy:
            return "healthy"
        parts = [f"{f.check}={f.response.value}" for f in self.findings if f.is_actionable]
        return "; ".join(parts)


@dataclass
class MonitorThresholds:
    max_calibration_error: float = 0.12
    lockout_calibration_error: float = 0.25
    max_slippage_ratio: float = 2.0
    lockout_slippage_ratio: float = 3.0
    min_fill_rate: float = 0.5
    max_drawdown_fraction: float = 0.06
    max_feature_drift: float = 3.0
    """Standardized shift in a feature's mean that counts as drift."""

    max_no_trade_rate: float = 0.995
    """A system that never trades is not failing safe, it is failing to work."""


@dataclass
class SystemMonitor:
    """Runs the spec 41 checks over recent operating history."""

    thresholds: MonitorThresholds = field(default_factory=MonitorThresholds)

    def check_calibration(
        self, probabilities: Sequence[float], outcomes: Sequence[float]
    ) -> HealthFinding:
        if len(probabilities) < 30:
            return HealthFinding(
                "calibration", Response.NONE, "insufficient samples", 0.0, 0.0
            )
        error = expected_calibration_error(probabilities, outcomes)
        if error >= self.thresholds.lockout_calibration_error:
            response = Response.LOCKOUT
        elif error >= self.thresholds.max_calibration_error:
            response = Response.REVERT_TO_BASELINE
        else:
            response = Response.NONE
        return HealthFinding(
            "calibration",
            response,
            f"expected calibration error {error:.3f}",
            error,
            self.thresholds.max_calibration_error,
        )

    def check_slippage(self, expected: Sequence[float], realized: Sequence[float]) -> HealthFinding:
        total_expected = float(np.sum(expected))
        if total_expected <= 0.0 or not expected:
            return HealthFinding("slippage_drift", Response.NONE, "no samples", 1.0, 1.0)
        ratio = float(np.sum(realized)) / total_expected
        if ratio >= self.thresholds.lockout_slippage_ratio:
            response = Response.LOCKOUT
        elif ratio >= self.thresholds.max_slippage_ratio:
            response = Response.REDUCE_SIZE
        else:
            response = Response.NONE
        return HealthFinding(
            "slippage_drift",
            response,
            f"realized/expected slippage {ratio:.2f}",
            ratio,
            self.thresholds.max_slippage_ratio,
        )

    def check_fill_rate(self, submitted: int, filled: int) -> HealthFinding:
        if submitted == 0:
            return HealthFinding("fill_rate", Response.NONE, "no orders", 1.0, 1.0)
        rate = filled / submitted
        response = (
            Response.DISABLE_COMPLEX if rate < self.thresholds.min_fill_rate else Response.NONE
        )
        return HealthFinding(
            "fill_rate", response, f"fill rate {rate:.2f}", rate, self.thresholds.min_fill_rate
        )

    def check_drawdown(self, equity_curve: Sequence[float]) -> HealthFinding:
        from spy_der.backtest.metrics import max_drawdown

        _, fraction = max_drawdown(equity_curve)
        response = (
            Response.LOCKOUT
            if fraction >= self.thresholds.max_drawdown_fraction
            else Response.NONE
        )
        return HealthFinding(
            "drawdown",
            response,
            f"drawdown {fraction:.1%} from peak",
            fraction,
            self.thresholds.max_drawdown_fraction,
        )

    def check_feature_drift(
        self, reference: Sequence[float], recent: Sequence[float]
    ) -> HealthFinding:
        """Standardized shift in a feature's mean between two periods."""
        ref = np.asarray(reference, dtype=np.float64)
        cur = np.asarray(recent, dtype=np.float64)
        if ref.size < 10 or cur.size < 10:
            return HealthFinding("feature_drift", Response.NONE, "insufficient samples", 0.0, 0.0)

        scale = float(np.std(ref))
        if scale <= 0.0:
            return HealthFinding("feature_drift", Response.NONE, "degenerate reference", 0.0, 0.0)
        shift = abs(float(np.mean(cur)) - float(np.mean(ref))) / scale
        response = (
            Response.REVERT_TO_BASELINE if shift >= self.thresholds.max_feature_drift else Response.NONE
        )
        return HealthFinding(
            "feature_drift",
            response,
            f"mean shifted {shift:.2f} standard deviations",
            shift,
            self.thresholds.max_feature_drift,
        )

    def check_activity(self, decisions: int, trades: int) -> HealthFinding:
        """Flag a system that has stopped trading entirely.

        Spec 41 monitors no-trade frequency. Abstention is the intended default,
        but a rate of essentially one over a long window means a threshold is
        misconfigured rather than that the market offered nothing.
        """
        if decisions < 200:
            return HealthFinding("activity", Response.NONE, "window too short", 0.0, 0.0)
        rate = 1.0 - trades / decisions
        response = Response.NONE if rate < self.thresholds.max_no_trade_rate else Response.REDUCE_SIZE
        return HealthFinding(
            "activity",
            response,
            f"no-trade rate {rate:.3f} over {decisions} decisions",
            rate,
            self.thresholds.max_no_trade_rate,
        )

    def report(self, **checks: HealthFinding) -> HealthReport:
        return HealthReport(findings=tuple(checks.values()))


def data_health(data_quality: float, *, threshold: float = 0.70) -> HealthFinding:
    """Point-in-time data freshness check."""
    return HealthFinding(
        "data_freshness",
        Response.LOCKOUT if data_quality < threshold else Response.NONE,
        f"data quality {data_quality:.2f}",
        data_quality,
        threshold,
    )
