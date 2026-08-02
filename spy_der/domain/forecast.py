"""Forecast domain types (spec sections 19 and 34.2).

The forecast engine emits distributions and probabilities only. It never emits
an instruction to trade a particular structure — that separation is spec 2.4
and is enforced here by the absence of any strategy-shaped field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from spy_der.domain.enums import MarketState

#: Forecast horizons required by spec section 19.
HORIZONS: tuple[str, ...] = (
    "5m",
    "15m",
    "30m",
    "60m",
    "eod",
    "next_session",
    "expiration",
)

HORIZON_MINUTES: dict[str, float] = {
    "5m": 5.0,
    "15m": 15.0,
    "30m": 30.0,
    "60m": 60.0,
    "eod": 390.0,
    "next_session": 780.0,
    "expiration": 390.0,
}

#: Quantile grid reported for price and implied volatility.
QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass(frozen=True, slots=True)
class HorizonForecast:
    """Distributional forecast for a single horizon."""

    horizon: str
    minutes: float
    price_quantiles: dict[float, float]
    iv_quantiles: dict[float, float]
    expected_price: float
    median_price: float
    price_std: float
    skewness: float
    kurtosis: float
    touch_probabilities: dict[float, float] = field(default_factory=dict)
    finish_probabilities: dict[float, float] = field(default_factory=dict)
    expected_mfe: float = 0.0
    expected_mae: float = 0.0
    mfe_quantiles: dict[float, float] = field(default_factory=dict)
    mae_quantiles: dict[float, float] = field(default_factory=dict)

    def quantile(self, tau: float) -> float:
        """Interpolate the price quantile function at ``tau``."""
        return _interpolate_quantile(self.price_quantiles, tau)

    def iv_quantile(self, tau: float) -> float:
        return _interpolate_quantile(self.iv_quantiles, tau)

    def probability_below(self, level: float) -> float:
        """Invert the quantile function to a CDF value at ``level``."""
        return _invert_quantile(self.price_quantiles, level)

    def probability_between(self, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.0
        return max(0.0, self.probability_below(upper) - self.probability_below(lower))


@dataclass(frozen=True, slots=True)
class ForecastBundle:
    """Complete forecast output (spec 34.2)."""

    timestamp: datetime
    spot: float
    horizons: dict[str, HorizonForecast]
    state_probabilities: dict[MarketState, float]
    pin_probabilities: dict[float, float] = field(default_factory=dict)
    range_probabilities: dict[tuple[float, float], float] = field(default_factory=dict)
    confidence: float = 0.0
    model_disagreement: float = 0.0
    calibration_score: float = 0.0
    state_entropy: float = 0.0
    state_stability: float = 1.0
    dealer_agreement: float = 1.0
    data_quality: float = 1.0
    expected_state_duration_minutes: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = sum(self.state_probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"state probabilities must sum to 1.0, got {total!r}")

    @property
    def dominant_state(self) -> MarketState:
        return max(self.state_probabilities.items(), key=lambda kv: kv[1])[0]

    @property
    def dominant_state_probability(self) -> float:
        return self.state_probabilities[self.dominant_state]

    def horizon(self, name: str) -> HorizonForecast:
        try:
            return self.horizons[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown horizon {name!r}; have {sorted(self.horizons)}") from exc

    def state_probability(self, state: MarketState) -> float:
        return self.state_probabilities.get(state, 0.0)


def _interpolate_quantile(grid: dict[float, float], tau: float) -> float:
    """Piecewise-linear interpolation of a quantile function."""
    if not grid:
        raise ValueError("empty quantile grid")
    points = sorted(grid.items())
    if tau <= points[0][0]:
        return points[0][1]
    if tau >= points[-1][0]:
        return points[-1][1]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= tau <= t1:
            if t1 == t0:
                return v0
            weight = (tau - t0) / (t1 - t0)
            return v0 + weight * (v1 - v0)
    return points[-1][1]  # pragma: no cover - unreachable


def _invert_quantile(grid: dict[float, float], level: float) -> float:
    """Map a value back to a cumulative probability via the quantile grid.

    Outside the tabulated support the result saturates at the extreme
    quantiles rather than extrapolating, which keeps probabilities inside
    ``[0, 1]`` and prevents a distant strike from producing a nonsense CDF.
    """
    if not grid:
        raise ValueError("empty quantile grid")
    points = sorted(grid.items(), key=lambda kv: kv[1])
    if level <= points[0][1]:
        return points[0][0]
    if level >= points[-1][1]:
        return points[-1][0]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if v0 <= level <= v1:
            if v1 == v0:
                return t0
            weight = (level - v0) / (v1 - v0)
            return t0 + weight * (t1 - t0)
    return points[-1][0]  # pragma: no cover - unreachable


def shannon_entropy(probabilities: dict[MarketState, float]) -> float:
    """``H(P) = -sum p ln p`` (spec section 18)."""
    return -sum(p * math.log(p) for p in probabilities.values() if p > 0.0)


def normalized_certainty(probabilities: dict[MarketState, float]) -> float:
    """``1 - H(P) / ln(n)`` (spec section 18)."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    return max(0.0, 1.0 - shannon_entropy(probabilities) / math.log(n))


def total_variation_stability(
    current: dict[MarketState, float], previous: dict[MarketState, float] | None
) -> float:
    """``1 - 0.5 * sum |P_t - P_{t-1}|`` (spec section 18)."""
    if not previous:
        return 1.0
    keys = set(current) | set(previous)
    l1 = sum(abs(current.get(k, 0.0) - previous.get(k, 0.0)) for k in keys)
    return max(0.0, 1.0 - 0.5 * l1)


def dispersion(values: list[float]) -> float:
    """Population standard deviation across model outputs (spec section 18)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
