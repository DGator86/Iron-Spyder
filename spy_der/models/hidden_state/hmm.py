"""Hidden-state model (spec section 14).

A discrete-state hidden Markov filter over the market-state taxonomy. Its job
is not to be a better classifier than the structural model but to impose
*temporal consistency*: without it, a state estimate driven by instantaneous
features flickers between neighbouring regimes on noise, and the optimizer
churns strategy families in response.

The filter runs forward only. It conditions on ``X_{1:t}`` and never on future
observations, so it is safe inside a point-in-time backtest — a smoother would
not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from spy_der.domain.enums import MarketState

STATES: tuple[MarketState, ...] = tuple(MarketState)
STATE_INDEX: dict[MarketState, int] = {s: i for i, s in enumerate(STATES)}

#: Baseline probability of remaining in the current state at each step.
#: Higher for regimes that empirically persist (pinning, ranging), lower for
#: the two transient states that exist precisely to be passed through.
DEFAULT_PERSISTENCE: dict[MarketState, float] = {
    MarketState.STRONG_PIN: 0.94,
    MarketState.BROAD_RANGE: 0.92,
    MarketState.BULL_GRIND: 0.90,
    MarketState.BEAR_GRIND: 0.90,
    MarketState.BULL_BREAKOUT: 0.82,
    MarketState.BEAR_BREAKDOWN: 0.82,
    MarketState.VOL_EXPANSION: 0.85,
    MarketState.VOL_CONTRACTION: 0.91,
    MarketState.DIRECTIONAL_RANGE: 0.88,
    MarketState.TRANSITION: 0.60,
    MarketState.UNSTABLE: 0.70,
}

#: States reachable from each state with elevated probability. Regimes do not
#: teleport: a strong pin decays into a range or a transition, not directly
#: into a breakdown.
NEIGHBOURS: dict[MarketState, tuple[MarketState, ...]] = {
    MarketState.STRONG_PIN: (MarketState.BROAD_RANGE, MarketState.TRANSITION),
    MarketState.BROAD_RANGE: (
        MarketState.STRONG_PIN,
        MarketState.DIRECTIONAL_RANGE,
        MarketState.TRANSITION,
    ),
    MarketState.BULL_GRIND: (MarketState.BULL_BREAKOUT, MarketState.TRANSITION),
    MarketState.BEAR_GRIND: (MarketState.BEAR_BREAKDOWN, MarketState.TRANSITION),
    MarketState.BULL_BREAKOUT: (MarketState.VOL_EXPANSION, MarketState.BULL_GRIND),
    MarketState.BEAR_BREAKDOWN: (MarketState.VOL_EXPANSION, MarketState.BEAR_GRIND),
    MarketState.VOL_EXPANSION: (
        MarketState.TRANSITION,
        MarketState.BULL_BREAKOUT,
        MarketState.BEAR_BREAKDOWN,
    ),
    MarketState.VOL_CONTRACTION: (MarketState.BROAD_RANGE, MarketState.STRONG_PIN),
    MarketState.DIRECTIONAL_RANGE: (
        MarketState.BROAD_RANGE,
        MarketState.BULL_GRIND,
        MarketState.BEAR_GRIND,
    ),
    MarketState.TRANSITION: tuple(
        s for s in STATES if s not in {MarketState.TRANSITION, MarketState.UNSTABLE}
    ),
    MarketState.UNSTABLE: (MarketState.TRANSITION,),
}


def build_transition_matrix(
    persistence: dict[MarketState, float] | None = None,
    neighbours: dict[MarketState, tuple[MarketState, ...]] | None = None,
    *,
    neighbour_share: float = 0.7,
) -> NDArray[np.float64]:
    """Row-stochastic transition matrix ``P(Z_t = j | Z_{t-1} = i)``.

    Each row spends ``persistence`` on staying put, ``neighbour_share`` of the
    remainder on structurally adjacent states, and the rest uniformly elsewhere
    so no transition has probability exactly zero.
    """
    persist = persistence or DEFAULT_PERSISTENCE
    adjacency = neighbours or NEIGHBOURS
    n = len(STATES)
    matrix = np.zeros((n, n), dtype=np.float64)

    for state in STATES:
        i = STATE_INDEX[state]
        stay = float(np.clip(persist.get(state, 0.85), 0.01, 0.99))
        matrix[i, i] = stay
        remaining = 1.0 - stay

        adjacent = tuple(s for s in adjacency.get(state, ()) if s is not state)
        others = tuple(s for s in STATES if s is not state and s not in adjacent)

        if adjacent:
            per_adjacent = remaining * neighbour_share / len(adjacent)
            for s in adjacent:
                matrix[i, STATE_INDEX[s]] = per_adjacent
            leftover = remaining * (1.0 - neighbour_share)
        else:
            leftover = remaining

        if others:
            per_other = leftover / len(others)
            for s in others:
                matrix[i, STATE_INDEX[s]] = per_other
        elif adjacent:
            matrix[i, STATE_INDEX[adjacent[0]]] += leftover

    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


@dataclass
class HiddenStateFilter:
    """Forward HMM filter over market states (spec section 14)."""

    transition: NDArray[np.float64] = field(default_factory=build_transition_matrix)
    belief: NDArray[np.float64] = field(
        default_factory=lambda: np.full(len(STATES), 1.0 / len(STATES))
    )
    observation_weight: float = 1.0
    """Exponent on the emission likelihood.

    Values below ``1.0`` damp the observation and lean harder on persistence,
    which is the knob for trading off responsiveness against whiplash.
    """

    _steps: int = 0

    def reset(self) -> None:
        self.belief = np.full(len(STATES), 1.0 / len(STATES))
        self._steps = 0

    def update(self, emission: dict[MarketState, float]) -> dict[MarketState, float]:
        """Advance one step given per-state emission likelihoods.

        ``emission`` is normally the structural model's probability vector,
        treated as ``P(X_t | Z_t)`` up to a constant.
        """
        likelihood = np.array(
            [max(emission.get(s, 0.0), 1e-12) for s in STATES], dtype=np.float64
        )
        if self.observation_weight != 1.0:
            likelihood = likelihood**self.observation_weight

        predicted = self.belief @ self.transition
        posterior = predicted * likelihood
        total = float(np.sum(posterior))
        if total <= 0.0 or not np.isfinite(total):  # pragma: no cover - defensive
            posterior = predicted
            total = float(np.sum(predicted))

        self.belief = posterior / total
        self._steps += 1
        return self.probabilities

    @property
    def probabilities(self) -> dict[MarketState, float]:
        return {s: float(self.belief[STATE_INDEX[s]]) for s in STATES}

    @property
    def dominant(self) -> MarketState:
        return STATES[int(np.argmax(self.belief))]

    def transition_probabilities(self, state: MarketState) -> dict[MarketState, float]:
        row = self.transition[STATE_INDEX[state]]
        return {s: float(row[STATE_INDEX[s]]) for s in STATES}

    def expected_duration(self, state: MarketState) -> float:
        """Expected remaining dwell time in ``state``, in steps.

        For a geometric dwell time with self-transition ``p``, the mean is
        ``1 / (1 - p)``.
        """
        p = float(self.transition[STATE_INDEX[state], STATE_INDEX[state]])
        return 1.0 / max(1e-9, 1.0 - p)

    def reversal_probability(self, state: MarketState) -> float:
        """Probability of leaving ``state`` on the next step."""
        return 1.0 - float(self.transition[STATE_INDEX[state], STATE_INDEX[state]])

    @property
    def steps_observed(self) -> int:
        return self._steps
