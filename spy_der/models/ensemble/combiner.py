"""Ensemble weighting and confidence (spec sections 17 and 18).

The weights here are the spec's stated initialization values. They are *not*
validated weights — spec 17 requires those to come from walk-forward
validation, and :class:`EnsembleWeights` is deliberately a plain data object so
a fold can carry its own fitted weights without any code change.

State-conditional weighting implements spec 17's requirement that authority
shift by regime: structure and persistence dominate in a pin, nonlinear and
barrier models dominate in a breakout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from spy_der.domain.enums import MarketState
from spy_der.domain.forecast import (
    dispersion,
    normalized_certainty,
    total_variation_stability,
)


@dataclass(frozen=True)
class EnsembleWeights:
    """``w_b``, ``w_n``, ``w_s``, ``w_h`` from spec section 17."""

    baseline: float = 0.35
    nonlinear: float = 0.25
    structural: float = 0.25
    hidden: float = 0.15

    def __post_init__(self) -> None:
        total = self.baseline + self.nonlinear + self.structural + self.hidden
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"ensemble weights must sum to 1.0, got {total}")

    def without_nonlinear(self) -> EnsembleWeights:
        """Redistribute the specialist weight when no specialist is deployed.

        Spec 41 requires the system to be able to revert to baseline models, so
        the absence of a nonlinear challenger must be a supported configuration
        rather than a hole that silently contributes zeros to the average.
        """
        scale = 1.0 / (1.0 - self.nonlinear) if self.nonlinear < 1.0 else 1.0
        return EnsembleWeights(
            baseline=self.baseline * scale,
            nonlinear=0.0,
            structural=self.structural * scale,
            hidden=self.hidden * scale,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "baseline": self.baseline,
            "nonlinear": self.nonlinear,
            "structural": self.structural,
            "hidden": self.hidden,
        }


#: Regime-specific weight overrides (spec section 17).
STATE_WEIGHTS: dict[MarketState, EnsembleWeights] = {
    MarketState.STRONG_PIN: EnsembleWeights(
        baseline=0.25, nonlinear=0.15, structural=0.35, hidden=0.25
    ),
    MarketState.BROAD_RANGE: EnsembleWeights(
        baseline=0.30, nonlinear=0.20, structural=0.30, hidden=0.20
    ),
    MarketState.BULL_BREAKOUT: EnsembleWeights(
        baseline=0.30, nonlinear=0.40, structural=0.20, hidden=0.10
    ),
    MarketState.BEAR_BREAKDOWN: EnsembleWeights(
        baseline=0.30, nonlinear=0.40, structural=0.20, hidden=0.10
    ),
    MarketState.VOL_EXPANSION: EnsembleWeights(
        baseline=0.30, nonlinear=0.35, structural=0.25, hidden=0.10
    ),
    MarketState.VOL_CONTRACTION: EnsembleWeights(
        baseline=0.35, nonlinear=0.25, structural=0.25, hidden=0.15
    ),
}


def blend_state_probabilities(
    *,
    baseline: dict[MarketState, float] | None = None,
    nonlinear: dict[MarketState, float] | None = None,
    structural: dict[MarketState, float],
    hidden: dict[MarketState, float],
    weights: EnsembleWeights | None = None,
) -> dict[MarketState, float]:
    """Weighted blend of state distributions, renormalized to sum to one."""
    active = weights or EnsembleWeights()
    if nonlinear is None:
        active = active.without_nonlinear()

    sources: list[tuple[float, dict[MarketState, float]]] = [
        (active.structural, structural),
        (active.hidden, hidden),
    ]
    if baseline is not None:
        sources.append((active.baseline, baseline))
    else:
        # Fold the unused baseline weight into the structural model rather than
        # dropping it, which would silently shrink the total below one.
        sources[0] = (sources[0][0] + active.baseline, structural)
    if nonlinear is not None:
        sources.append((active.nonlinear, nonlinear))

    blended: dict[MarketState, float] = {}
    for state in MarketState:
        blended[state] = sum(weight * source.get(state, 0.0) for weight, source in sources)

    total = sum(blended.values())
    if total <= 0.0:  # pragma: no cover - defensive
        uniform = 1.0 / len(MarketState)
        return dict.fromkeys(MarketState, uniform)
    return {state: value / total for state, value in blended.items()}


def blend_probability(
    *,
    baseline: float | None = None,
    nonlinear: float | None = None,
    structural: float | None = None,
    hidden: float | None = None,
    weights: EnsembleWeights | None = None,
) -> float:
    """Scalar version of :func:`blend_state_probabilities` (spec section 17)."""
    active = weights or EnsembleWeights()
    pairs = [
        (active.baseline, baseline),
        (active.nonlinear, nonlinear),
        (active.structural, structural),
        (active.hidden, hidden),
    ]
    available = [(w, v) for w, v in pairs if v is not None]
    if not available:
        return 0.5
    total_weight = sum(w for w, _ in available)
    if total_weight <= 0.0:
        return float(np.mean([v for _, v in available]))
    return float(sum(w * v for w, v in available) / total_weight)


@dataclass(frozen=True)
class ConfidenceInputs:
    """Everything feeding the adjusted confidence of spec section 18."""

    raw_confidence: float
    state_probabilities: dict[MarketState, float]
    previous_state_probabilities: dict[MarketState, float] | None
    model_predictions: Sequence[float]
    data_quality: float
    dealer_agreement: float
    calibration_quality: float = 1.0


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Decomposed confidence so a low score can be attributed to a cause."""

    adjusted: float
    raw: float
    certainty: float
    stability: float
    disagreement: float
    data_quality: float
    dealer_agreement: float
    calibration_quality: float
    entropy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "confidence": self.adjusted,
            "confidence_raw": self.raw,
            "certainty": self.certainty,
            "stability": self.stability,
            "model_disagreement": self.disagreement,
            "data_quality": self.data_quality,
            "dealer_agreement": self.dealer_agreement,
            "calibration_quality": self.calibration_quality,
            "state_entropy": self.entropy,
        }


def assess_confidence(inputs: ConfidenceInputs) -> ConfidenceAssessment:
    """``C* = C_raw * Certainty * Stability * (1 - D) * DQ * DealerAgreement``.

    Every factor is in ``[0, 1]`` and they multiply, so the result degrades
    fast when several are mediocre. That is the intent: the spec wants low
    confidence to make no-trade more likely, not to be averaged away.
    """
    from spy_der.domain.forecast import shannon_entropy

    certainty = normalized_certainty(inputs.state_probabilities)
    stability = total_variation_stability(
        inputs.state_probabilities, inputs.previous_state_probabilities
    )
    disagreement = float(np.clip(dispersion(list(inputs.model_predictions)), 0.0, 1.0))

    adjusted = (
        float(np.clip(inputs.raw_confidence, 0.0, 1.0))
        * certainty
        * stability
        * (1.0 - disagreement)
        * float(np.clip(inputs.data_quality, 0.0, 1.0))
        * float(np.clip(inputs.dealer_agreement, 0.0, 1.0))
        * float(np.clip(inputs.calibration_quality, 0.0, 1.0))
    )

    return ConfidenceAssessment(
        adjusted=float(np.clip(adjusted, 0.0, 1.0)),
        raw=inputs.raw_confidence,
        certainty=certainty,
        stability=stability,
        disagreement=disagreement,
        data_quality=inputs.data_quality,
        dealer_agreement=inputs.dealer_agreement,
        calibration_quality=inputs.calibration_quality,
        entropy=shannon_entropy(inputs.state_probabilities),
    )


def weights_for_state(
    state: MarketState, *, default: EnsembleWeights | None = None
) -> EnsembleWeights:
    return STATE_WEIGHTS.get(state, default or EnsembleWeights())


@dataclass
class EnsembleConfig:
    """Runtime ensemble configuration recorded in the audit trail."""

    base_weights: EnsembleWeights = field(default_factory=EnsembleWeights)
    state_conditional: bool = True
    nonlinear_enabled: bool = False
    """Off by default: spec 39 forbids promoting a specialist that has not
    beaten the baselines on walk-forward Brier, log loss, and net expectancy."""

    def resolve(self, state: MarketState) -> EnsembleWeights:
        weights = (
            weights_for_state(state, default=self.base_weights)
            if self.state_conditional
            else self.base_weights
        )
        return weights if self.nonlinear_enabled else weights.without_nonlinear()

    def with_nonlinear(self, enabled: bool) -> EnsembleConfig:
        return replace(self, nonlinear_enabled=enabled)
