"""Calibration and forecast-quality tests (spec sections 18, 37.9, 39)."""

from __future__ import annotations

import numpy as np
import pytest

from spy_der.domain.enums import MarketState
from spy_der.domain.forecast import (
    dispersion,
    normalized_certainty,
    total_variation_stability,
)
from spy_der.models.baselines import analytic
from spy_der.models.calibration.calibrator import (
    ProbabilityCalibrator,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_curve,
    score_calibration,
)
from spy_der.models.ensemble.combiner import (
    ConfidenceInputs,
    EnsembleConfig,
    EnsembleWeights,
    assess_confidence,
    blend_state_probabilities,
)
from spy_der.models.hidden_state.hmm import HiddenStateFilter, build_transition_matrix

# --------------------------------------------------------------------------
# Scoring rules (spec 37.9)
# --------------------------------------------------------------------------


def test_perfect_forecast_scores_zero():
    outcomes = [1, 0, 1, 0, 1]
    perfect = [1.0, 0.0, 1.0, 0.0, 1.0]
    assert brier_score(perfect, outcomes) == pytest.approx(0.0)
    assert log_loss(perfect, outcomes) == pytest.approx(0.0, abs=1e-9)


def test_uninformative_forecast_scores_the_expected_constants():
    outcomes = [1, 0, 1, 0]
    half = [0.5] * 4
    assert brier_score(half, outcomes) == pytest.approx(0.25)
    assert log_loss(half, outcomes) == pytest.approx(np.log(2.0))


def test_log_loss_is_finite_for_a_confident_miss():
    """Clipping keeps a wrong 0.0 from producing infinity."""
    assert np.isfinite(log_loss([0.0], [1]))


def test_calibrated_forecast_has_low_ece():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.05, 0.95, size=4_000)
    outcomes = (rng.uniform(size=4_000) < probabilities).astype(float)
    assert expected_calibration_error(probabilities, outcomes) < 0.05


def test_miscalibrated_forecast_has_high_ece():
    rng = np.random.default_rng(1)
    probabilities = rng.uniform(0.05, 0.95, size=4_000)
    # Outcomes realize at a much lower rate than predicted.
    outcomes = (rng.uniform(size=4_000) < probabilities * 0.4).astype(float)
    assert expected_calibration_error(probabilities, outcomes) > 0.15


def test_reliability_curve_shape():
    rng = np.random.default_rng(2)
    probabilities = rng.uniform(size=500)
    outcomes = (rng.uniform(size=500) < probabilities).astype(float)
    predicted, observed, counts = reliability_curve(probabilities, outcomes, bins=10)
    assert len(predicted) == len(observed) == len(counts) == 10
    assert counts.sum() == 500


def test_calibration_report_rewards_skill_over_the_base_rate():
    rng = np.random.default_rng(3)
    outcomes = (rng.uniform(size=2_000) < 0.3).astype(float)
    skilled = np.where(outcomes > 0, 0.75, 0.15)
    naive = np.full(2_000, 0.3)

    good = score_calibration("skilled", skilled, outcomes)
    flat = score_calibration("base-rate", naive, outcomes)
    assert good.brier < flat.brier
    assert good.brier_skill_score > 0.0
    # The constant forecast has no skill. The tolerance is sampling noise: the
    # reference is the *realized* base rate, which differs slightly from the
    # 0.3 the naive forecast was told to use.
    assert flat.brier_skill_score == pytest.approx(0.0, abs=1e-2)
    assert 0.0 <= good.quality <= 1.0


# --------------------------------------------------------------------------
# Recalibration
# --------------------------------------------------------------------------


def test_isotonic_calibration_corrects_a_systematic_bias():
    rng = np.random.default_rng(4)
    truth = rng.uniform(0.05, 0.95, size=3_000)
    outcomes = (rng.uniform(size=3_000) < truth).astype(float)
    biased = np.clip(truth * 0.6 + 0.2, 0.0, 1.0)  # compressed toward the middle

    before = expected_calibration_error(biased, outcomes)
    calibrator = ProbabilityCalibrator(method="isotonic").fit(biased, outcomes)
    after = expected_calibration_error(calibrator.transform_many(biased), outcomes)
    assert calibrator.is_fitted
    assert after < before


def test_calibrator_is_a_passthrough_below_the_sample_floor():
    """Recalibrating on a handful of points invents structure; refuse to."""
    calibrator = ProbabilityCalibrator(min_samples=100).fit([0.2, 0.8], [0, 1])
    assert not calibrator.is_fitted
    assert calibrator.transform(0.37) == pytest.approx(0.37)


def test_calibrator_handles_single_class_outcomes():
    calibrator = ProbabilityCalibrator(min_samples=5).fit([0.3] * 20, [1] * 20)
    assert not calibrator.is_fitted


# --------------------------------------------------------------------------
# Confidence (spec 18)
# --------------------------------------------------------------------------


def _states(peak: float) -> dict[MarketState, float]:
    rest = (1.0 - peak) / (len(MarketState) - 1)
    return {
        state: (peak if state is MarketState.BROAD_RANGE else rest) for state in MarketState
    }


def test_confidence_falls_with_every_degrading_factor():
    base = ConfidenceInputs(
        raw_confidence=0.8,
        state_probabilities=_states(0.8),
        previous_state_probabilities=_states(0.8),
        model_predictions=[0.8, 0.79, 0.81],
        data_quality=1.0,
        dealer_agreement=1.0,
    )
    reference = assess_confidence(base).adjusted

    from dataclasses import replace

    for field, value in (
        ("data_quality", 0.5),
        ("dealer_agreement", 0.3),
        ("model_predictions", [0.1, 0.9, 0.5]),
        ("previous_state_probabilities", _states(0.2)),
        ("calibration_quality", 0.4),
    ):
        degraded = assess_confidence(replace(base, **{field: value})).adjusted
        assert degraded < reference, field


def test_confidence_is_bounded():
    assessment = assess_confidence(
        ConfidenceInputs(
            raw_confidence=1.0,
            state_probabilities=_states(0.999),
            previous_state_probabilities=None,
            model_predictions=[0.5],
            data_quality=1.0,
            dealer_agreement=1.0,
        )
    )
    assert 0.0 <= assessment.adjusted <= 1.0
    assert 0.0 <= assessment.certainty <= 1.0
    assert 0.0 <= assessment.stability <= 1.0


def test_uniform_distribution_has_zero_certainty():
    uniform = {state: 1.0 / len(MarketState) for state in MarketState}
    assert normalized_certainty(uniform) == pytest.approx(0.0, abs=1e-9)


def test_stability_measures_total_variation():
    a = _states(0.8)
    assert total_variation_stability(a, a) == pytest.approx(1.0)
    assert total_variation_stability(a, None) == 1.0
    b = _states(0.1)
    assert total_variation_stability(a, b) < 1.0


def test_dispersion_is_zero_for_agreement():
    assert dispersion([0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert dispersion([0.1, 0.9]) > 0.0


# --------------------------------------------------------------------------
# Ensemble (spec 17)
# --------------------------------------------------------------------------


def test_ensemble_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        EnsembleWeights(baseline=0.9, nonlinear=0.9, structural=0.1, hidden=0.1)


def test_spec_initialization_weights():
    weights = EnsembleWeights()
    assert (weights.baseline, weights.nonlinear, weights.structural, weights.hidden) == (
        0.35,
        0.25,
        0.25,
        0.15,
    )


def test_dropping_the_nonlinear_model_redistributes_its_weight():
    reduced = EnsembleWeights().without_nonlinear()
    assert reduced.nonlinear == 0.0
    assert sum(reduced.as_dict().values()) == pytest.approx(1.0)


def test_blended_state_probabilities_sum_to_one():
    structural = _states(0.6)
    hidden = _states(0.4)
    blended = blend_state_probabilities(structural=structural, hidden=hidden)
    assert sum(blended.values()) == pytest.approx(1.0)


def test_state_conditional_weights_shift_authority():
    config = EnsembleConfig()
    pin = config.resolve(MarketState.STRONG_PIN)
    breakout = config.resolve(MarketState.BULL_BREAKOUT)
    # Structure and persistence dominate a pin; the reverse in a breakout.
    assert pin.structural + pin.hidden > breakout.structural + breakout.hidden


def test_nonlinear_is_disabled_by_default():
    """Spec 39 forbids promoting an unvalidated specialist."""
    assert EnsembleConfig().nonlinear_enabled is False
    assert EnsembleConfig().resolve(MarketState.BROAD_RANGE).nonlinear == 0.0


# --------------------------------------------------------------------------
# Hidden-state model (spec 14)
# --------------------------------------------------------------------------


def test_transition_matrix_is_row_stochastic():
    matrix = build_transition_matrix()
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert np.all(matrix >= 0.0)
    assert np.all(np.diag(matrix) > 0.5)  # states persist


def test_hmm_belief_remains_a_distribution():
    hmm = HiddenStateFilter()
    emission = _states(0.7)
    for _ in range(20):
        probabilities = hmm.update(emission)
        assert sum(probabilities.values()) == pytest.approx(1.0)
    assert hmm.dominant is MarketState.BROAD_RANGE


def test_hmm_damps_a_single_conflicting_observation():
    """Temporal consistency is the point of the filter (spec 14)."""
    hmm = HiddenStateFilter()
    for _ in range(15):
        hmm.update(_states(0.8))

    spike = {
        state: (0.8 if state is MarketState.BEAR_BREAKDOWN else 0.02) for state in MarketState
    }
    total = sum(spike.values())
    after = hmm.update({k: v / total for k, v in spike.items()})
    # The filter moves toward the spike but does not jump to it.
    assert after[MarketState.BEAR_BREAKDOWN] < 0.8
    assert after[MarketState.BROAD_RANGE] > after[MarketState.BEAR_BREAKDOWN]


def test_expected_duration_matches_the_geometric_mean():
    hmm = HiddenStateFilter()
    for state in MarketState:
        p = hmm.transition_probabilities(state)[state]
        assert hmm.expected_duration(state) == pytest.approx(1.0 / (1.0 - p))
        assert hmm.reversal_probability(state) == pytest.approx(1.0 - p)


def test_hmm_reset_restores_the_uniform_prior():
    hmm = HiddenStateFilter()
    hmm.update(_states(0.9))
    hmm.reset()
    assert hmm.steps_observed == 0
    assert all(
        v == pytest.approx(1.0 / len(MarketState)) for v in hmm.probabilities.values()
    )


# --------------------------------------------------------------------------
# Analytic baselines as oracles (spec 15)
# --------------------------------------------------------------------------


def test_barrier_probabilities_are_monotone_in_distance():
    spot, tau, vol = 550.0, 2 / 365, 0.18
    near = analytic.touch_probability_up(spot, 553.0, tau, vol)
    far = analytic.touch_probability_up(spot, 565.0, tau, vol)
    assert 0.0 < far < near < 1.0

    near_down = analytic.touch_probability_down(spot, 547.0, tau, vol)
    far_down = analytic.touch_probability_down(spot, 535.0, tau, vol)
    assert 0.0 < far_down < near_down < 1.0


def test_touch_probability_exceeds_finish_probability():
    """Price can touch a level and come back, so touch dominates finish."""
    spot, tau, vol = 550.0, 2 / 365, 0.18
    for barrier in (552.0, 556.0, 560.0):
        touch = analytic.touch_probability_up(spot, barrier, tau, vol)
        finish = analytic.finish_probability_above(spot, barrier, tau, vol)
        assert touch > finish


def test_no_touch_is_bounded_by_terminal_containment():
    """Surviving both barriers implies finishing between them, never the reverse."""
    spot, tau, vol = 550.0, 2 / 365, 0.18
    for lower, upper in ((545.0, 555.0), (540.0, 560.0), (535.0, 565.0)):
        no_touch = analytic.no_touch_probability(spot, lower, upper, tau, vol)
        finish = analytic.range_containment_probability(spot, lower, upper, tau, vol)
        assert 0.0 <= no_touch <= finish <= 1.0


def test_quantiles_are_ordered_and_straddle_spot():
    spot, tau, vol = 550.0, 5 / 365, 0.2
    quantiles = [analytic.price_quantile(spot, tau, vol, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert quantiles == sorted(quantiles)
    assert quantiles[0] < spot < quantiles[-1]


def test_pin_probability_peaks_at_spot():
    spot, tau, vol = 550.0, 1 / 365, 0.12
    at_spot = analytic.pin_probability(spot, 550.0, 1.0, tau, vol)
    away = analytic.pin_probability(spot, 556.0, 1.0, tau, vol)
    assert at_spot > away
