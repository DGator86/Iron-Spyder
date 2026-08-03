"""Forecast engine: assembles the distributional forecast (spec sections 12, 19).

Wires the hybrid architecture of spec 12 together — structural mechanics,
hidden state, interpretable baselines, calibration, ensemble, path simulation —
and emits a :class:`~spy_der.domain.forecast.ForecastBundle`.

It emits distributions and probabilities only. There is no code path from here
to a strategy recommendation; spec 2.4 makes that separation architectural, and
the optimizer is the only component allowed to translate a forecast into a
trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spy_der.domain.enums import MarketState
from spy_der.domain.forecast import (
    HORIZON_MINUTES,
    QUANTILES,
    ForecastBundle,
    HorizonForecast,
)
from spy_der.features.builder import AnalyticsBundle, MarketFeatures
from spy_der.models.calibration.calibrator import CalibrationReport
from spy_der.models.ensemble.combiner import (
    ConfidenceInputs,
    EnsembleConfig,
    assess_confidence,
    blend_state_probabilities,
)
from spy_der.models.hidden_state.hmm import HiddenStateFilter
from spy_der.models.structural.scores import StructuralModel
from spy_der.simulation.price_paths import (
    PathBundle,
    PathSimulator,
    blend_dynamics,
)

MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0


@dataclass
class ForecastEngineConfig:
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    n_paths: int = 2_000
    strike_span_pct: float = 0.04
    """Half-width of the strike grid for touch and finish probabilities."""

    pin_epsilon_pct: float = 0.0015
    """Half-width of the pin window, as a fraction of spot."""

    range_widths_pct: tuple[float, ...] = (0.003, 0.006, 0.010, 0.015)
    max_horizon_minutes: float = 390.0


@dataclass
class ForecastEngine:
    """Produces a calibrated distributional forecast from one snapshot."""

    structural: StructuralModel = field(default_factory=StructuralModel)
    hidden: HiddenStateFilter = field(default_factory=HiddenStateFilter)
    simulator: PathSimulator = field(default_factory=PathSimulator)
    config: ForecastEngineConfig = field(default_factory=ForecastEngineConfig)

    calibration: dict[str, CalibrationReport] = field(default_factory=dict)
    _previous_states: dict[MarketState, float] | None = field(default=None, init=False)

    def reset(self) -> None:
        """Clear temporal state; used between walk-forward folds."""
        self.hidden.reset()
        self._previous_states = None

    def forecast(
        self,
        analytics: AnalyticsBundle,
        *,
        baseline_states: dict[MarketState, float] | None = None,
        nonlinear_states: dict[MarketState, float] | None = None,
    ) -> tuple[ForecastBundle, PathBundle]:
        """Return the forecast bundle and the paths it was derived from.

        The paths are returned rather than discarded because the optimizer
        values every candidate on the *same* paths. Re-simulating per candidate
        would let Monte Carlo noise, rather than payoff geometry, decide the
        ranking.
        """
        features = analytics.features
        spot = features.spot

        structural = self.structural.assess(features)
        hidden_states = self.hidden.update(structural.probabilities)

        provisional = max(structural.probabilities.items(), key=lambda kv: kv[1])[0]
        weights = self.config.ensemble.resolve(provisional)
        states = blend_state_probabilities(
            baseline=baseline_states,
            nonlinear=nonlinear_states if self.config.ensemble.nonlinear_enabled else None,
            structural=structural.probabilities,
            hidden=hidden_states,
            weights=weights,
        )

        confidence = assess_confidence(
            ConfidenceInputs(
                raw_confidence=max(states.values()),
                state_probabilities=states,
                previous_state_probabilities=self._previous_states,
                model_predictions=self._disagreement_sample(
                    structural.probabilities, hidden_states, baseline_states, nonlinear_states
                ),
                data_quality=features.data_quality,
                dealer_agreement=features.gex_agreement,
                calibration_quality=self.calibration_quality,
                structural_weight=weights.structural,
            )
        )
        self._previous_states = states

        bundle = self._simulate(features, states)
        horizons = self._build_horizons(features, bundle)

        dominant = max(states.items(), key=lambda kv: kv[1])[0]
        forecast = ForecastBundle(
            timestamp=features.timestamp,
            spot=spot,
            horizons=horizons,
            state_probabilities=states,
            pin_probabilities=self._pin_probabilities(features, bundle),
            range_probabilities=self._range_probabilities(features, bundle),
            confidence=confidence.adjusted,
            model_disagreement=confidence.disagreement,
            calibration_score=self.calibration_quality,
            state_entropy=confidence.entropy,
            state_stability=confidence.stability,
            dealer_agreement=features.gex_agreement,
            data_quality=features.data_quality,
            expected_state_duration_minutes=self._expected_duration_minutes(dominant),
            diagnostics={
                **confidence.as_dict(),
                **{f"weight_{k}": v for k, v in weights.as_dict().items()},
                "structural_dominant_probability": max(structural.probabilities.values()),
                "hidden_dominant_probability": max(hidden_states.values()),
            },
        )
        return forecast, bundle

    # -- internals ---------------------------------------------------------

    @property
    def calibration_quality(self) -> float:
        if not self.calibration:
            return 1.0
        return float(np.mean([r.quality for r in self.calibration.values()]))

    @staticmethod
    def _disagreement_sample(
        structural: dict[MarketState, float],
        hidden: dict[MarketState, float],
        baseline: dict[MarketState, float] | None,
        nonlinear: dict[MarketState, float] | None,
    ) -> list[float]:
        """Each model's probability for the structurally favoured state.

        Measuring dispersion on a common target is what makes the numbers
        comparable; dispersion across each model's own maximum would conflate
        disagreement about *which* state with disagreement about *how likely*.
        """
        target = max(structural.items(), key=lambda kv: kv[1])[0]
        sample = [structural[target], hidden.get(target, 0.0)]
        for source in (baseline, nonlinear):
            if source is not None:
                sample.append(source.get(target, 0.0))
        return sample

    def _expected_duration_minutes(self, state: MarketState) -> float:
        # HMM steps correspond to snapshot intervals; five minutes is the
        # system's decision cadence.
        return self.hidden.expected_duration(state) * 5.0

    def _simulate(
        self, features: MarketFeatures, states: dict[MarketState, float]
    ) -> PathBundle:
        horizon_minutes = min(
            self.config.max_horizon_minutes,
            max(
                HORIZON_MINUTES["60m"],
                features.minutes_to_close,
                features.seconds_to_nearest_expiry / 60.0,
            ),
        )
        horizon_years = horizon_minutes / MINUTES_PER_YEAR
        base_vol = features.atm_iv if features.atm_iv > 0.0 else max(features.realized_vol, 0.05)

        return self.simulator.simulate(
            spot=features.spot,
            base_vol=base_vol,
            horizon_years=horizon_years,
            dynamics=blend_dynamics(states),
            n_paths=self.config.n_paths,
        )

    def _build_horizons(
        self, features: MarketFeatures, bundle: PathBundle
    ) -> dict[str, HorizonForecast]:
        spot = features.spot
        strikes = self._strike_grid(features)
        horizons: dict[str, HorizonForecast] = {}

        for name, minutes in HORIZON_MINUTES.items():
            effective = self._effective_minutes(name, minutes, features)
            sub = bundle.truncate(effective / MINUTES_PER_YEAR)
            terminal = sub.terminal

            horizons[name] = HorizonForecast(
                horizon=name,
                minutes=effective,
                price_quantiles=sub.price_quantiles(QUANTILES),
                iv_quantiles=sub.iv_quantiles(QUANTILES, features.atm_iv),
                expected_price=float(np.mean(terminal)),
                median_price=float(np.median(terminal)),
                price_std=float(np.std(terminal)),
                skewness=_skewness(terminal),
                kurtosis=_kurtosis(terminal),
                touch_probabilities={
                    k: (
                        sub.touch_probability_up(k)
                        if k >= spot
                        else sub.touch_probability_down(k)
                    )
                    for k in strikes
                },
                finish_probabilities={k: sub.finish_probability_above(k) for k in strikes},
                expected_mfe=float(np.mean(sub.mfe)),
                expected_mae=float(np.mean(sub.mae)),
                mfe_quantiles={
                    q: float(v) for q, v in zip(QUANTILES, np.quantile(sub.mfe, QUANTILES))
                },
                mae_quantiles={
                    q: float(v) for q, v in zip(QUANTILES, np.quantile(sub.mae, QUANTILES))
                },
            )
        return horizons

    @staticmethod
    def _effective_minutes(name: str, minutes: float, features: MarketFeatures) -> float:
        """Clamp each horizon to what the session and the chain actually allow."""
        if name == "eod":
            return max(5.0, min(minutes, features.minutes_to_close))
        if name == "expiration":
            return max(5.0, features.seconds_to_nearest_expiry / 60.0)
        return minutes

    def _strike_grid(self, features: MarketFeatures) -> tuple[float, ...]:
        spot = features.spot
        span = spot * self.config.strike_span_pct
        step = max(1.0, round(span / 8.0))
        low = round((spot - span) / step) * step
        count = int(2 * span / step) + 1
        return tuple(float(low + i * step) for i in range(count))

    def _pin_probabilities(
        self, features: MarketFeatures, bundle: PathBundle
    ) -> dict[float, float]:
        """Terminal probability of finishing within the pin window of each candidate strike."""
        spot = features.spot
        epsilon = spot * self.config.pin_epsilon_pct
        candidates: set[float] = {float(round(spot))}
        if features.pin_strike is not None:
            candidates.add(features.pin_strike)
        if features.call_wall is not None:
            candidates.add(features.call_wall)
        if features.put_wall is not None:
            candidates.add(features.put_wall)

        horizon = bundle.truncate(
            max(5.0, features.seconds_to_nearest_expiry / 60.0) / MINUTES_PER_YEAR
        )
        return {
            float(k): horizon.finish_between(k - epsilon, k + epsilon) for k in sorted(candidates)
        }

    def _range_probabilities(
        self, features: MarketFeatures, bundle: PathBundle
    ) -> dict[tuple[float, float], float]:
        """No-touch probability for symmetric bands around spot.

        No-touch rather than finish: a range strategy that is stopped out on a
        breach does not care that price came back before the close.
        """
        spot = features.spot
        eod = bundle.truncate(
            max(5.0, min(390.0, features.minutes_to_close)) / MINUTES_PER_YEAR
        )
        out: dict[tuple[float, float], float] = {}
        for width in self.config.range_widths_pct:
            lower, upper = spot * (1.0 - width), spot * (1.0 + width)
            out[(round(lower, 2), round(upper, 2))] = eod.no_touch_probability(lower, upper)
        return out


def _skewness(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0.0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    """Excess kurtosis; zero for a normal distribution."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0.0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 4) - 3.0)
