"""Gamma profile across a hypothetical SPY grid (spec section 8.5).

The whole chain is repriced at every grid point rather than approximated by a
distance weighting. Gamma is strongly nonlinear in spot, so a weighting kernel
over static per-contract gamma produces a curve whose zero crossing — the one
number the state engine actually consumes — sits in the wrong place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from spy_der.analytics.chain import ChainArrays
from spy_der.analytics.exposures import dealer_signs
from spy_der.domain.enums import DealerSignConvention

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class GammaProfile:
    """Net dealer gamma as a function of hypothetical spot."""

    spots: FloatArray
    gex: FloatArray
    slope: FloatArray
    spot: float
    convention: DealerSignConvention

    flip_points: tuple[float, ...]
    nearest_flip: float | None
    vol_trigger: float | None
    local_maxima: tuple[float, ...]
    local_minima: tuple[float, ...]
    positive_zones: tuple[tuple[float, float], ...]
    negative_zones: tuple[tuple[float, float], ...]

    @property
    def gex_at_spot(self) -> float:
        return float(np.interp(self.spot, self.spots, self.gex))

    @property
    def slope_at_spot(self) -> float:
        return float(np.interp(self.spot, self.spots, self.slope))

    @property
    def in_positive_gamma(self) -> bool:
        return self.gex_at_spot > 0.0

    @property
    def distance_to_flip(self) -> float | None:
        """Signed distance from spot to the nearest flip, in SPY points."""
        if self.nearest_flip is None:
            return None
        return self.nearest_flip - self.spot

    @property
    def normalized_flip_distance(self) -> float | None:
        """Flip distance as a fraction of spot; ``None`` when there is no flip."""
        distance = self.distance_to_flip
        if distance is None or self.spot <= 0.0:
            return None
        return distance / self.spot


def build_gamma_profile(
    arrays: ChainArrays,
    *,
    convention: DealerSignConvention = DealerSignConvention.CALLS_POSITIVE_PUTS_NEGATIVE,
    flow_signs: FloatArray | None = None,
    width: float = 0.05,
    points: int = 81,
) -> GammaProfile:
    """Reprice the chain over ``S_j`` in ``[S(1-a), S(1+a)]`` (spec 8.5).

    The default convention is calls-positive/puts-negative because a flip point
    only exists when calls and puts carry opposite signs. Under a uniform-sign
    convention such as ``CUSTOMER_LONG`` the curve never crosses zero and
    ``nearest_flip`` would always be ``None`` — correct arithmetic, useless
    signal. Aggregate exposure levels still use every convention; see
    :func:`spy_der.analytics.exposures.compute_exposures`.
    """
    if points < 5:
        raise ValueError("points must be >= 5 to resolve slope and extrema")
    if not 0.0 < width < 1.0:
        raise ValueError("width must be in (0, 1)")

    spot = arrays.spot
    spots = np.linspace(spot * (1.0 - width), spot * (1.0 + width), points)
    signs = dealer_signs(arrays, convention, flow_signs=flow_signs)

    # Reprice gamma for the whole chain at each grid point. Shape: (points, n).
    grid = spots[:, None]
    gamma_grid = arrays.gamma_at(grid)
    weights = arrays.open_interest * 100.0 * 0.01 * signs
    gex = np.sum(gamma_grid * weights * grid * grid, axis=1)

    slope = np.gradient(gex, spots)

    flips = _zero_crossings(spots, gex)
    nearest = min(flips, key=lambda f: abs(f - spot)) if flips else None

    trigger_index = int(np.argmax(np.abs(slope)))
    vol_trigger = float(spots[trigger_index]) if flips or np.any(slope) else None

    return GammaProfile(
        spots=spots,
        gex=gex,
        slope=slope,
        spot=spot,
        convention=convention,
        flip_points=flips,
        nearest_flip=nearest,
        vol_trigger=vol_trigger,
        local_maxima=_local_extrema(spots, gex, maxima=True),
        local_minima=_local_extrema(spots, gex, maxima=False),
        positive_zones=_sign_zones(spots, gex, positive=True),
        negative_zones=_sign_zones(spots, gex, positive=False),
    )


def _zero_crossings(spots: FloatArray, values: FloatArray) -> tuple[float, ...]:
    """Exact linear-interpolated zero crossings of a sampled curve."""
    out: list[float] = []
    for i in range(len(spots) - 1):
        v0, v1 = float(values[i]), float(values[i + 1])
        if v0 == 0.0:
            out.append(float(spots[i]))
        elif v0 * v1 < 0.0:
            s0, s1 = float(spots[i]), float(spots[i + 1])
            out.append(s0 + (s1 - s0) * (-v0) / (v1 - v0))
    if values[-1] == 0.0:
        out.append(float(spots[-1]))
    return tuple(out)


def _local_extrema(spots: FloatArray, values: FloatArray, *, maxima: bool) -> tuple[float, ...]:
    out: list[float] = []
    for i in range(1, len(values) - 1):
        prev, cur, nxt = float(values[i - 1]), float(values[i]), float(values[i + 1])
        if maxima and cur > prev and cur > nxt or not maxima and cur < prev and cur < nxt:
            out.append(float(spots[i]))
    return tuple(out)


def _sign_zones(
    spots: FloatArray, values: FloatArray, *, positive: bool
) -> tuple[tuple[float, float], ...]:
    """Contiguous ``[low, high]`` intervals where the curve holds one sign."""
    zones: list[tuple[float, float]] = []
    start: float | None = None
    for spot, value in zip(spots.tolist(), values.tolist()):
        in_zone = value > 0.0 if positive else value < 0.0
        if in_zone and start is None:
            start = spot
        elif not in_zone and start is not None:
            zones.append((start, spot))
            start = None
    if start is not None:
        zones.append((start, float(spots[-1])))
    return tuple(zones)


def profile_agreement(profiles: Sequence[GammaProfile]) -> float:
    """Agreement between flip locations computed under different conventions.

    A flip that moves by a few tenths of a percent between conventions is
    usable; one that jumps several percent is not, and this is the number that
    tells the state engine which case it is in.
    """
    flips = [p.nearest_flip for p in profiles if p.nearest_flip is not None]
    if len(flips) < 2:
        return 1.0 if flips else 0.0
    spot = profiles[0].spot
    spread = (max(flips) - min(flips)) / max(spot, 1e-9)
    return float(np.clip(1.0 - spread / 0.02, 0.0, 1.0))
