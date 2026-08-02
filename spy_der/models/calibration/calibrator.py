"""Probability calibration and scoring (spec sections 18, 37.9, 39).

A forecast that is discriminative but miscalibrated is worse than useless to
this system: the optimizer multiplies probabilities by payoffs, so a 0.7 that
is really a 0.5 turns a negative-expectancy trade into an apparently positive
one. Calibration is therefore measured continuously and is itself a kill-switch
input (spec 42).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-12

#: Accepted input for the scoring functions. Spelled out because numpy 2.5's
#: stubs no longer treat ``ndarray`` as structurally compatible with
#: ``Sequence``, and every caller here passes arrays.
FloatVector: TypeAlias = "Sequence[float] | NDArray[np.float64]"


def brier_score(probabilities: FloatVector, outcomes: FloatVector) -> float:
    """``BS = mean((p_i - y_i)^2)`` (spec 37.9)."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss(probabilities: FloatVector, outcomes: FloatVector) -> float:
    """``LL = -mean(y ln p + (1-y) ln(1-p))`` (spec 37.9)."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    probabilities: FloatVector, outcomes: FloatVector, *, bins: int = 10
) -> float:
    """Bin-weighted mean gap between predicted and realized frequency."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges, edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        total += (count / p.size) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(total)


def reliability_curve(
    probabilities: FloatVector, outcomes: FloatVector, *, bins: int = 10
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
    """Return ``(mean_predicted, observed_frequency, counts)`` per bin."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)

    predicted, observed, counts = [], [], []
    for lo, hi in zip(edges, edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        count = int(np.sum(mask))
        counts.append(count)
        predicted.append(float(np.mean(p[mask])) if count else float("nan"))
        observed.append(float(np.mean(y[mask])) if count else float("nan"))
    return (
        np.array(predicted, dtype=np.float64),
        np.array(observed, dtype=np.float64),
        np.array(counts, dtype=np.int_),
    )


@dataclass
class ProbabilityCalibrator:
    """Isotonic or Platt recalibration of a probability stream.

    Isotonic is the default because it corrects arbitrary monotone
    miscalibration; Platt is available for small samples where isotonic
    overfits. Below ``min_samples`` the calibrator is a pass-through, which is
    the honest behaviour — recalibrating on thirty observations invents
    structure that is not there.
    """

    method: str = "isotonic"
    min_samples: int = 100
    _model: object | None = field(default=None, init=False, repr=False)
    _n: int = field(default=0, init=False)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self, probabilities: FloatVector, outcomes: FloatVector
    ) -> ProbabilityCalibrator:
        p = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(outcomes, dtype=np.float64)
        self._n = p.size

        if p.size < self.min_samples or len(np.unique(y)) < 2:
            self._model = None
            return self

        if self.method == "isotonic":
            model: object = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(p, y)  # type: ignore[attr-defined]
        elif self.method == "platt":
            model = LogisticRegression(max_iter=1_000)
            model.fit(p.reshape(-1, 1), y)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"unknown calibration method {self.method!r}")

        self._model = model
        return self

    def transform(self, probability: float) -> float:
        if self._model is None:
            return float(np.clip(probability, 0.0, 1.0))
        if self.method == "isotonic":
            value = float(self._model.predict([probability])[0])  # type: ignore[attr-defined]
        else:
            value = float(
                self._model.predict_proba(np.array([[probability]]))[0, 1]  # type: ignore[attr-defined]
            )
        return float(np.clip(value, 0.0, 1.0))

    def transform_many(self, probabilities: FloatVector) -> NDArray[np.float64]:
        return np.array([self.transform(p) for p in probabilities], dtype=np.float64)


@dataclass(frozen=True)
class CalibrationReport:
    """Scored calibration quality for one probability stream."""

    label: str
    n: int
    brier: float
    log_loss: float
    ece: float
    base_rate: float
    reference_brier: float
    """Brier score of the constant base-rate forecast — the thing to beat."""

    @property
    def brier_skill_score(self) -> float:
        """``1 - BS / BS_ref``; positive means better than always predicting the base rate."""
        if not np.isfinite(self.reference_brier) or self.reference_brier <= 0.0:
            return 0.0
        return float(1.0 - self.brier / self.reference_brier)

    @property
    def quality(self) -> float:
        """Single ``[0, 1]`` score for the confidence term and kill switch."""
        if self.n == 0 or not np.isfinite(self.ece):
            return 0.0
        calibration_term = float(np.clip(1.0 - self.ece / 0.15, 0.0, 1.0))
        skill_term = float(np.clip(0.5 + self.brier_skill_score, 0.0, 1.0))
        return float(np.clip(0.7 * calibration_term + 0.3 * skill_term, 0.0, 1.0))


def score_calibration(
    label: str, probabilities: FloatVector, outcomes: FloatVector, *, bins: int = 10
) -> CalibrationReport:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    base_rate = float(np.mean(y)) if y.size else float("nan")
    reference = float(np.mean((base_rate - y) ** 2)) if y.size else float("nan")

    return CalibrationReport(
        label=label,
        n=int(p.size),
        brier=brier_score(p, y),
        log_loss=log_loss(p, y),
        ece=expected_calibration_error(p, y, bins=bins),
        base_rate=base_rate,
        reference_brier=reference,
    )
