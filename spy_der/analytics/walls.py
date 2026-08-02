"""Call and put wall detection and interaction tracking (spec 8.6-8.7)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from spy_der.analytics.chain import ChainArrays
from spy_der.analytics.exposures import gamma_exposure


class WallMethod(str, Enum):
    """Alternative wall calculations (spec 8.6)."""

    GAMMA_WEIGHTED = "gamma_weighted"
    OPEN_INTEREST = "open_interest"
    DELTA_WEIGHTED = "delta_weighted"
    VOLUME_WEIGHTED_GAMMA = "volume_weighted_gamma"


class WallInteraction(str, Enum):
    """Observed price behaviour at a wall (spec 8.7)."""

    AWAY = "away"
    APPROACH = "approach"
    TOUCH = "touch"
    REJECTION = "rejection"
    BREAK = "break"
    ACCEPTANCE = "acceptance"
    MIGRATION = "migration"


@dataclass(frozen=True)
class Wall:
    strike: float
    magnitude: float
    method: WallMethod

    def distance_from(self, spot: float) -> float:
        return self.strike - spot

    def normalized_distance(self, spot: float) -> float:
        return (self.strike - spot) / max(spot, 1e-9)


@dataclass(frozen=True)
class WallSet:
    call_wall: Wall | None
    put_wall: Wall | None
    spot: float
    concentration: float
    """Share of total wall magnitude held by the single largest strike."""

    @property
    def inside_walls(self) -> bool:
        if self.call_wall is None or self.put_wall is None:
            return False
        return self.put_wall.strike <= self.spot <= self.call_wall.strike

    @property
    def wall_width(self) -> float | None:
        if self.call_wall is None or self.put_wall is None:
            return None
        return self.call_wall.strike - self.put_wall.strike

    @property
    def position_in_range(self) -> float | None:
        """Where spot sits between the walls: ``0`` at the put wall, ``1`` at the call wall."""
        width = self.wall_width
        if width is None or width <= 0.0 or self.put_wall is None:
            return None
        return float(np.clip((self.spot - self.put_wall.strike) / width, 0.0, 1.0))


def _weights(arrays: ChainArrays, method: WallMethod) -> np.ndarray:
    if method is WallMethod.GAMMA_WEIGHTED:
        return np.abs(gamma_exposure(arrays, np.ones(len(arrays))))
    if method is WallMethod.OPEN_INTEREST:
        return arrays.open_interest.copy()
    if method is WallMethod.DELTA_WEIGHTED:
        return np.abs(arrays.delta) * arrays.open_interest
    if method is WallMethod.VOLUME_WEIGHTED_GAMMA:
        return np.abs(arrays.gamma) * (arrays.open_interest + arrays.volume)
    raise ValueError(f"unhandled wall method {method!r}")  # pragma: no cover


def find_walls(
    arrays: ChainArrays, *, method: WallMethod = WallMethod.GAMMA_WEIGHTED
) -> WallSet:
    """Locate the dominant call and put strikes (spec 8.6, 8.7)."""
    weights = _weights(arrays, method)

    def dominant(mask: np.ndarray) -> Wall | None:
        if not np.any(mask):
            return None
        strikes = arrays.strike[mask]
        values = weights[mask]
        totals: dict[float, float] = {}
        for strike, value in zip(strikes.tolist(), values.tolist()):
            totals[strike] = totals.get(strike, 0.0) + value
        if not totals or max(totals.values()) <= 0.0:
            return None
        strike, magnitude = max(totals.items(), key=lambda kv: kv[1])
        return Wall(strike=strike, magnitude=magnitude, method=method)

    call_wall = dominant(arrays.call_mask)
    put_wall = dominant(arrays.put_mask)

    total = float(np.sum(weights))
    largest = max(
        (w.magnitude for w in (call_wall, put_wall) if w is not None), default=0.0
    )
    concentration = largest / total if total > 0.0 else 0.0

    return WallSet(
        call_wall=call_wall,
        put_wall=put_wall,
        spot=arrays.spot,
        concentration=float(np.clip(concentration, 0.0, 1.0)),
    )


@dataclass
class WallTracker:
    """Classifies price behaviour at a wall across successive snapshots.

    Distinguishing a break from a rejection needs history, so this object is
    deliberately stateful and is owned by the pipeline rather than recreated
    per snapshot.
    """

    touch_tolerance: float = 0.0015
    acceptance_bars: int = 3
    history_size: int = 64
    _spots: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    _walls: deque[float] = field(default_factory=lambda: deque(maxlen=64))

    def update(self, spot: float, wall_strike: float | None) -> WallInteraction:
        if wall_strike is None:
            self._spots.append(spot)
            return WallInteraction.AWAY

        previous_spots = list(self._spots)
        previous_walls = list(self._walls)
        self._spots.append(spot)
        self._walls.append(wall_strike)

        if previous_walls and abs(wall_strike - previous_walls[-1]) > 1e-9:
            return WallInteraction.MIGRATION

        distance = abs(spot - wall_strike) / max(wall_strike, 1e-9)
        beyond = spot > wall_strike

        if beyond:
            recent = previous_spots[-self.acceptance_bars :]
            if len(recent) >= self.acceptance_bars and all(s > wall_strike for s in recent):
                return WallInteraction.ACCEPTANCE
            return WallInteraction.BREAK

        if distance <= self.touch_tolerance:
            return WallInteraction.TOUCH

        if previous_spots:
            was_beyond_or_touching = (
                previous_spots[-1] > wall_strike
                or abs(previous_spots[-1] - wall_strike) / max(wall_strike, 1e-9)
                <= self.touch_tolerance
            )
            if was_beyond_or_touching:
                return WallInteraction.REJECTION
            if spot > previous_spots[-1] and distance < 0.01:
                return WallInteraction.APPROACH

        return WallInteraction.AWAY
