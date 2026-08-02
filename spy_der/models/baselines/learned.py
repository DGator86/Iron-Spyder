"""Fitted interpretable baselines (spec section 15).

Regularized linear models only. Spec 15 keeps these active even once nonlinear
challengers are deployed, so they are the system's floor: if a specialist
cannot beat a ridge-penalized logistic regression out of sample, the specialist
does not get promoted.

Each model degrades to an explicit prior when unfitted rather than refusing to
predict. A fresh deployment therefore produces calibrated-by-construction
forecasts from the analytic baselines instead of crashing or, worse, emitting
0.5 everywhere and letting the optimizer treat that as information.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import ElasticNet, LogisticRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler

from spy_der.features.builder import MarketFeatures

MODEL_VERSION = "1.0.0"


@dataclass
class FittedState:
    """Everything learned during ``fit``, kept together for versioning."""

    feature_names: tuple[str, ...]
    scaler: StandardScaler
    estimator: object
    n_samples: int
    version: str = MODEL_VERSION


class BaselineModel:
    """Common fit/predict plumbing with an explicit unfitted prior."""

    def __init__(self, feature_names: Sequence[str], *, name: str = "baseline") -> None:
        self.feature_names = tuple(feature_names)
        self.name = name
        self.state: FittedState | None = None

    @property
    def is_fitted(self) -> bool:
        return self.state is not None

    def _matrix(self, rows: Sequence[MarketFeatures]) -> NDArray[np.float64]:
        return np.vstack([r.vector(self.feature_names) for r in rows])

    def _build_estimator(self) -> object:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, rows: Sequence[MarketFeatures], targets: Sequence[float]) -> BaselineModel:
        if len(rows) != len(targets):
            raise ValueError("rows and targets must be the same length")
        if len(rows) < self.min_samples:
            raise ValueError(
                f"{self.name} needs at least {self.min_samples} samples, got {len(rows)}"
            )

        matrix = self._matrix(rows)
        scaler = StandardScaler().fit(matrix)
        estimator = self._build_estimator()
        estimator.fit(scaler.transform(matrix), np.asarray(targets))  # type: ignore[attr-defined]
        self.state = FittedState(
            feature_names=self.feature_names,
            scaler=scaler,
            estimator=estimator,
            n_samples=len(rows),
        )
        return self

    min_samples: int = 50


class LogisticBaseline(BaselineModel):
    """Regularized logistic regression for a binary event (spec 15.1-15.5)."""

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        name: str = "logistic",
        prior: Callable[[MarketFeatures], float] | None = None,
        C: float = 0.5,
    ) -> None:
        super().__init__(feature_names, name=name)
        self.prior = prior
        self.C = C

    def _build_estimator(self) -> object:
        return LogisticRegression(C=self.C, max_iter=2_000, penalty="l2")

    def predict(self, features: MarketFeatures) -> float:
        if self.state is None:
            return self._prior_value(features)
        x = self.state.scaler.transform(features.vector(self.feature_names).reshape(1, -1))
        proba = self.state.estimator.predict_proba(x)[0]  # type: ignore[attr-defined]
        # Guard against a single-class training fold collapsing predict_proba.
        if proba.shape[0] < 2:  # pragma: no cover - degenerate fold
            return self._prior_value(features)
        return float(np.clip(proba[1], 1e-6, 1.0 - 1e-6))

    def _prior_value(self, features: MarketFeatures) -> float:
        return float(np.clip(self.prior(features), 1e-6, 1.0 - 1e-6)) if self.prior else 0.5


class QuantileBaseline(BaselineModel):
    """Linear quantile regression for return quantiles (spec 15.6)."""

    def __init__(
        self,
        feature_names: Sequence[str],
        quantile: float,
        *,
        name: str | None = None,
        alpha: float = 0.01,
        prior: Callable[[MarketFeatures], float] | None = None,
    ) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        super().__init__(feature_names, name=name or f"quantile_{quantile:g}")
        self.quantile = quantile
        self.alpha = alpha
        self.prior = prior

    def _build_estimator(self) -> object:
        return QuantileRegressor(quantile=self.quantile, alpha=self.alpha, solver="highs")

    def predict(self, features: MarketFeatures) -> float:
        if self.state is None:
            return self.prior(features) if self.prior else 0.0
        x = self.state.scaler.transform(features.vector(self.feature_names).reshape(1, -1))
        return float(self.state.estimator.predict(x)[0])  # type: ignore[attr-defined]


class ElasticNetBaseline(BaselineModel):
    """Elastic net for expected IV change (spec 15.7)."""

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        name: str = "iv_change",
        alpha: float = 0.001,
        l1_ratio: float = 0.5,
        prior: Callable[[MarketFeatures], float] | None = None,
    ) -> None:
        super().__init__(feature_names, name=name)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.prior = prior

    def _build_estimator(self) -> object:
        return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5_000)

    def predict(self, features: MarketFeatures) -> float:
        if self.state is None:
            return self.prior(features) if self.prior else 0.0
        x = self.state.scaler.transform(features.vector(self.feature_names).reshape(1, -1))
        return float(self.state.estimator.predict(x)[0])  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Default feature subsets and priors
# --------------------------------------------------------------------------

DIRECTIONAL_FEATURES: tuple[str, ...] = (
    "return_since_open",
    "vwap_distance_norm",
    "flow_tilt",
    "flow_delta",
    "flow_persistence",
    "risk_reversal",
    "dex_total",
    "in_positive_gamma",
    "flip_distance_norm",
    "call_wall_distance_norm",
    "put_wall_distance_norm",
    "em_utilization",
    "session_fraction",
)

VOLATILITY_FEATURES: tuple[str, ...] = (
    "atm_iv",
    "realized_vol",
    "iv_rv_spread",
    "rv_change",
    "rv_acceleration",
    "term_slope",
    "term_curvature",
    "butterfly",
    "in_positive_gamma",
    "gamma_slope_at_spot",
    "em_utilization",
    "session_fraction",
)

CONTAINMENT_FEATURES: tuple[str, ...] = (
    "pin_concentration",
    "pin_distance_norm",
    "wall_concentration",
    "position_in_range",
    "in_positive_gamma",
    "realized_vol",
    "atm_iv",
    "expected_move_pct",
    "em_remaining_fraction",
    "session_fraction",
    "seconds_to_nearest_expiry",
)


def trend_opportunity_prior(features: MarketFeatures) -> float:
    """``P(TrendOpportunity | X_t)`` prior (spec 15.1).

    Trend opportunity rises with realized volatility and negative gamma, and
    falls when price is pinned inside a tight positive-gamma range.
    """
    vol_term = float(np.clip(features.realized_vol / 0.25, 0.0, 1.0))
    gamma_term = 0.0 if features.in_positive_gamma else 0.35
    pin_penalty = 0.30 * features.pin_concentration
    return float(np.clip(0.20 + 0.45 * vol_term + gamma_term - pin_penalty, 0.02, 0.95))


def direction_prior(features: MarketFeatures) -> float:
    """``P(Up | TrendOpportunity, X_t)`` prior (spec 15.2).

    Deliberately weak. Directional information in options structure is thin,
    and a prior that swings far from 0.5 would hand the optimizer confidence
    that has not been earned.
    """
    tilt = 0.5
    tilt += 0.10 * float(np.clip(features.return_since_open / 0.005, -1.0, 1.0))
    tilt += 0.05 * float(np.clip(features.flow_tilt, -1.0, 1.0))
    tilt += 0.03 * float(np.clip(features.risk_reversal / 0.02, -1.0, 1.0))
    return float(np.clip(tilt, 0.25, 0.75))


def iv_change_prior(features: MarketFeatures) -> float:
    """``E[dIV]`` prior: implied volatility mean-reverts toward realized."""
    gap = features.iv_rv_spread
    return float(np.clip(-0.10 * gap, -0.05, 0.05))
