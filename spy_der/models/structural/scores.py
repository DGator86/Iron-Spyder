"""Structural mechanics model (spec section 13).

Produces an interpretable score per market state from options mechanics alone,
then maps the scores to probabilities with a softmax. Nothing here is fitted:
the weights are priors expressing how dealer hedging is believed to work, which
is exactly what makes this component able to *veto* a statistically attractive
but mechanically implausible trade (spec 13, final paragraph).

The weights below are initialization values. Spec 17 requires final weights to
come from walk-forward validation, and :mod:`spy_der.models.ensemble` is where
that authority is actually apportioned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spy_der.domain.enums import MarketState
from spy_der.features.builder import MarketFeatures


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class ScoreScales:
    """Normalization constants converting raw features to ``[0, 1]`` drivers.

    Each is the feature value at which the corresponding driver saturates.
    They are stated as data rather than buried in expressions so they can be
    re-estimated from SPY history without touching the scoring logic.
    """

    quiet_realized_vol: float = 0.11
    active_realized_vol: float = 0.16
    vol_expansion_realized: float = 0.25
    drift_full_scale: float = 0.005
    """Return since the open at which a directional driver saturates (0.5%)."""

    iv_rv_full_scale: float = 0.09
    flip_proximity: float = 0.004
    """Normalized distance to the gamma flip counting as "at" the flip."""

    wall_proximity: float = 0.010
    pin_concentration_floor: float = 0.25
    pin_concentration_ceiling: float = 0.75
    """Bounds on the gamma-dominance ratio; see ``_pin_candidate``."""

    rv_acceleration_scale: float = 0.02
    term_slope_scale: float = 0.04


@dataclass(frozen=True)
class StructuralDrivers:
    """Normalized, interpretable drivers shared by the state scores."""

    positive_gamma: float
    negative_gamma: float
    near_flip: float
    pin_concentration: float
    pin_proximity: float
    quiet: float
    active: float
    expanding: float
    iv_rich: float
    iv_cheap: float
    drift_up: float
    drift_down: float
    flow_up: float
    flow_down: float
    em_used: float
    em_unused: float
    inside_walls: float
    call_wall_near: float
    put_wall_near: float
    wall_asymmetry: float
    rv_accelerating: float
    iv_rising: float
    iv_falling: float
    time_decay: float
    uncertainty: float


def compute_drivers(
    features: MarketFeatures, scales: ScoreScales | None = None
) -> StructuralDrivers:
    """Normalize raw features into bounded drivers."""
    s = scales or ScoreScales()
    rv = features.realized_vol

    pin_span = max(s.pin_concentration_ceiling - s.pin_concentration_floor, 1e-9)
    pin_conc = _clip01((features.pin_concentration - s.pin_concentration_floor) / pin_span)

    # Gate proximity by concentration. The dominant gamma strike is adjacent to
    # spot on essentially every chain, so ungated proximity is a constant and
    # would hand every quiet snapshot a large pin score regardless of whether a
    # pin exists. Being near the pin only counts once there is one.
    pin_prox = (
        _clip01(1.0 - abs(features.pin_distance_norm) / s.wall_proximity) * pin_conc
        if features.pin_strike is not None
        else 0.0
    )

    near_flip = (
        _clip01(1.0 - abs(features.flip_distance_norm) / s.flip_proximity)
        if features.gamma_flip is not None
        else 0.0
    )

    inside = 1.0 if 0.05 <= features.position_in_range <= 0.95 else 0.0
    call_near = _clip01(1.0 - abs(features.call_wall_distance_norm) / s.wall_proximity)
    put_near = _clip01(1.0 - abs(features.put_wall_distance_norm) / s.wall_proximity)
    asymmetry = _clip01(
        abs(abs(features.call_wall_distance_norm) - abs(features.put_wall_distance_norm))
        / max(s.wall_proximity, 1e-9)
    )

    # Structural read is untrustworthy when the dealer conventions disagree or
    # the data is degraded; this drives the Unstable score directly.
    uncertainty = _clip01(
        0.5 * (1.0 - features.gex_agreement)
        + 0.3 * (1.0 - features.gex_sign_consensus)
        + 0.2 * (1.0 - features.data_quality)
    )

    return StructuralDrivers(
        positive_gamma=features.in_positive_gamma,
        negative_gamma=1.0 - features.in_positive_gamma,
        near_flip=near_flip,
        pin_concentration=pin_conc,
        pin_proximity=pin_prox,
        quiet=_clip01(1.0 - rv / s.quiet_realized_vol),
        active=_clip01((rv - s.active_realized_vol) / max(s.active_realized_vol, 1e-9)),
        expanding=_clip01((rv - s.vol_expansion_realized) / max(s.vol_expansion_realized, 1e-9)),
        iv_rich=_clip01(features.iv_rv_spread / s.iv_rv_full_scale),
        iv_cheap=_clip01(-features.iv_rv_spread / s.iv_rv_full_scale),
        drift_up=_clip01(features.return_since_open / s.drift_full_scale),
        drift_down=_clip01(-features.return_since_open / s.drift_full_scale),
        flow_up=_clip01(features.flow_tilt),
        flow_down=_clip01(-features.flow_tilt),
        em_used=_clip01(features.em_utilization),
        em_unused=features.em_remaining_fraction,
        inside_walls=inside,
        call_wall_near=call_near,
        put_wall_near=put_near,
        wall_asymmetry=asymmetry,
        rv_accelerating=_clip01(features.rv_acceleration / s.rv_acceleration_scale),
        iv_rising=_clip01(features.iv_change / 0.01),
        iv_falling=_clip01(-features.iv_change / 0.01),
        time_decay=features.session_fraction,
        uncertainty=uncertainty,
    )


#: Per-state linear weights over :class:`StructuralDrivers` fields.
#: Negative entries are penalties, matching the spec's example score forms.
DEFAULT_WEIGHTS: dict[MarketState, dict[str, float]] = {
    MarketState.STRONG_PIN: {
        "pin_concentration": 2.6,
        "pin_proximity": 1.1,
        "positive_gamma": 0.7,
        "quiet": 1.5,
        "time_decay": 0.4,
        "active": -2.0,
        "drift_up": -0.9,
        "drift_down": -0.9,
    },
    MarketState.BROAD_RANGE: {
        "inside_walls": 1.2,
        "positive_gamma": 0.7,
        "iv_rich": 1.1,
        "quiet": 0.9,
        "em_unused": 0.8,
        "pin_concentration": -1.1,
        "active": -1.6,
        "drift_up": -0.7,
        "drift_down": -0.7,
    },
    MarketState.BULL_GRIND: {
        "drift_up": 2.1,
        "quiet": 1.0,
        "put_wall_near": 0.6,
        "positive_gamma": 0.5,
        "iv_falling": 0.4,
        "flow_up": 0.4,
        "em_unused": 0.5,
        "active": -1.6,
        "expanding": -1.8,
        "drift_down": -2.2,
    },
    MarketState.BEAR_GRIND: {
        "drift_down": 2.1,
        "quiet": 1.0,
        "call_wall_near": 0.6,
        "positive_gamma": 0.5,
        "iv_rising": 0.4,
        "flow_down": 0.4,
        "em_unused": 0.5,
        "active": -1.6,
        "drift_up": -2.2,
    },
    MarketState.BULL_BREAKOUT: {
        "drift_up": 1.5,
        "active": 2.4,
        "negative_gamma": 0.6,
        "em_used": 0.7,
        "rv_accelerating": 0.5,
        "flow_up": 0.4,
        "quiet": -2.6,
        "drift_down": -2.4,
    },
    MarketState.BEAR_BREAKDOWN: {
        "drift_down": 1.5,
        "active": 2.4,
        "negative_gamma": 0.6,
        "em_used": 0.7,
        "rv_accelerating": 0.5,
        "flow_down": 0.4,
        "quiet": -2.6,
        "drift_up": -2.4,
    },
    MarketState.VOL_EXPANSION: {
        "expanding": 3.8,
        "active": 1.2,
        "iv_cheap": 1.0,
        "negative_gamma": 0.6,
        "rv_accelerating": 0.5,
        "near_flip": 0.3,
        "quiet": -2.5,
        "drift_up": -0.8,
        "drift_down": -0.8,
    },
    MarketState.VOL_CONTRACTION: {
        "iv_rich": 3.4,
        "quiet": 3.0,
        "positive_gamma": 0.6,
        "iv_falling": 0.5,
        "inside_walls": 0.4,
        "pin_concentration": -0.9,
        "drift_up": -1.2,
        "drift_down": -1.2,
        "active": -2.0,
        "expanding": -2.5,
    },
    MarketState.DIRECTIONAL_RANGE: {
        "wall_asymmetry": 2.6,
        "inside_walls": 0.8,
        "positive_gamma": 0.4,
        "quiet": 0.5,
        "em_unused": 0.4,
        "pin_concentration": -0.8,
        "active": -1.6,
        "expanding": -1.8,
    },
    MarketState.TRANSITION: {
        "near_flip": 2.4,
        "uncertainty": 0.8,
        "rv_accelerating": 0.4,
    },
    MarketState.UNSTABLE: {
        "uncertainty": 3.4,
    },
}

#: Per-state intercepts. Transition and Unstable sit slightly below the others
#: so they win only on positive evidence, not by default.
DEFAULT_BIASES: dict[MarketState, float] = {
    MarketState.TRANSITION: -0.8,
    MarketState.UNSTABLE: -1.4,
}


@dataclass(frozen=True)
class StructuralAssessment:
    scores: dict[MarketState, float]
    probabilities: dict[MarketState, float]
    drivers: StructuralDrivers

    @property
    def dominant(self) -> MarketState:
        return max(self.probabilities.items(), key=lambda kv: kv[1])[0]


@dataclass(frozen=True)
class StructuralModel:
    """Interpretable state scorer (spec section 13)."""

    weights: dict[MarketState, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}
    )
    biases: dict[MarketState, float] = field(default_factory=lambda: dict(DEFAULT_BIASES))
    scales: ScoreScales = field(default_factory=ScoreScales)
    temperature: float = 1.0

    def assess(self, features: MarketFeatures) -> StructuralAssessment:
        drivers = compute_drivers(features, self.scales)
        values = {
            name: getattr(drivers, name) for name in StructuralDrivers.__dataclass_fields__
        }

        scores: dict[MarketState, float] = {}
        for state in MarketState:
            weights = self.weights.get(state, {})
            total = self.biases.get(state, 0.0)
            for name, weight in weights.items():
                total += weight * values[name]
            scores[state] = total

        return StructuralAssessment(
            scores=scores,
            probabilities=softmax(scores, temperature=self.temperature),
            drivers=drivers,
        )


def softmax(scores: dict[MarketState, float], *, temperature: float = 1.0) -> dict[MarketState, float]:
    """``P(Z_i) = exp(Score_i) / sum_j exp(Score_j)`` (spec section 13).

    The maximum is subtracted before exponentiating for numerical stability;
    the result is unchanged.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    keys = list(scores)
    raw = np.array([scores[k] for k in keys], dtype=np.float64) / temperature
    shifted = np.exp(raw - np.max(raw))
    total = float(np.sum(shifted))
    return {k: float(v / total) for k, v in zip(keys, shifted)}
