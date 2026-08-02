from __future__ import annotations

from dataclasses import dataclass

STATES = [
    "Strong pin",
    "Broad range",
    "Bull grind",
    "Bear grind",
    "Bull breakout",
    "Bear breakdown",
    "Volatility expansion",
    "Volatility contraction",
    "Directional range",
    "Transition",
    "Unstable",
]


@dataclass
class StateClassification:
    state: str
    probabilities: dict[str, float]


def classify_market_state(expected_move: float, spot: float, gamma_flip: float | None, skew: float) -> StateClassification:
    ratio = expected_move / max(spot, 1)
    probs = {s: 0.03 for s in STATES}
    if ratio < 0.008:
        state = "Strong pin"
    elif ratio < 0.012:
        state = "Broad range"
    elif skew < -0.01:
        state = "Bull grind"
    elif skew > 0.01:
        state = "Bear grind"
    elif ratio > 0.03:
        state = "Volatility expansion"
    else:
        state = "Transition"
    probs[state] = 0.7
    if gamma_flip is None:
        probs["Unstable"] = 0.15
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    return StateClassification(state=state, probabilities=probs)
