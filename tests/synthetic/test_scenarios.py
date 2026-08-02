"""Synthetic scenario tests (spec section 37.3).

Each scenario carries a predefined expected state and expected risk decision,
so these assert against a ground truth rather than against whatever the system
happens to output.

Scope caveat worth stating plainly: the structural weights were calibrated
*against these scenarios*, so passing shows the state engine separates the
regimes the generator encodes — not that it separates real SPY regimes. Spec 17
requires walk-forward validation on real data before the weights can be trusted.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from spy_der.data.synthetic import SCENARIOS, build_scenario
from spy_der.data.validators.quality import DataQualityConfig, assess
from spy_der.domain.enums import MarketState
from spy_der.features.builder import FeatureContext, build_features
from spy_der.models.forecast_engine import ForecastEngine, ForecastEngineConfig
from spy_der.optimizer.family_filter import eligible_families

TRADEABLE = sorted(n for n, s in SCENARIOS.items() if s.expected_tradeable)
ADVERSE = sorted(
    n
    for n, s in SCENARIOS.items()
    if not s.expected_tradeable and n != "event_lockout"
)


def _classify(name: str, *, warmup: int = 3):
    snapshot = build_scenario(name)
    snapshot = replace(snapshot, data_quality_score=assess(snapshot).score)
    context = FeatureContext()
    bundle = build_features(snapshot, context)
    for _ in range(warmup - 1):
        bundle = build_features(snapshot, context)

    engine = ForecastEngine(config=ForecastEngineConfig(n_paths=300))
    forecast, _ = engine.forecast(bundle)
    for _ in range(warmup - 1):
        forecast, _ = engine.forecast(bundle)
    return snapshot, bundle, forecast


@pytest.mark.parametrize("name", TRADEABLE + ["event_lockout"])
def test_scenario_classifies_to_expected_state(name):
    _, _, forecast = _classify(name)
    expected = SCENARIOS[name].expected_state
    assert forecast.dominant_state is expected, (
        f"{name}: expected {expected.value}, got {forecast.dominant_state.value} "
        f"(p={forecast.dominant_state_probability:.2f})"
    )


@pytest.mark.parametrize("name", TRADEABLE)
def test_state_probabilities_are_a_distribution(name):
    _, _, forecast = _classify(name)
    total = sum(forecast.state_probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= p <= 1.0 for p in forecast.state_probabilities.values())


@pytest.mark.parametrize("name", TRADEABLE)
def test_eligible_families_match_the_state(name):
    """Spec 37.3: the eligible strategy families are predefined per scenario."""
    _, _, forecast = _classify(name)
    families = eligible_families(forecast.state_probabilities)
    assert families, f"{name} produced no eligible families"

    from spy_der.optimizer.family_filter import families_for_state

    modal = families_for_state(forecast.dominant_state)
    assert modal <= families


@pytest.mark.parametrize("name", ADVERSE)
def test_adverse_scenarios_are_locked_out(name):
    """Degraded data must disable trading (spec 6, 44)."""
    snapshot = build_scenario(name)
    config = DataQualityConfig(require_same_day_expiry=(name == "missing_same_day"))
    report = assess(snapshot, config)
    assert not report.is_tradeable, f"{name} scored DQ={report.score:.2f} and was tradeable"


def test_corrupt_quotes_are_detected_individually():
    snapshot = build_scenario("corrupt_quotes")
    report = assess(snapshot)
    assert report.score == 0.0
    assert report.has_fatal
    assert {"crossed_market", "negative_bid", "duplicate_contracts"} <= report.codes


def test_stale_data_is_fatal_above_the_stale_fraction():
    report = assess(build_scenario("stale_data"))
    assert "stale_quotes" in report.codes
    assert report.score == 0.0
    assert not report.is_tradeable


def test_clean_scenarios_score_well():
    for name in TRADEABLE:
        report = assess(build_scenario(name))
        assert report.is_tradeable, f"{name} scored DQ={report.score:.2f}"
        assert report.score > 0.7


def test_pin_scenario_has_a_dominant_gamma_strike():
    _, bundle, _ = _classify("strong_pin")
    features = bundle.features
    assert features.pin_strike is not None
    assert features.pin_concentration > 0.5
    assert abs(features.pin_distance_norm) < 0.01


def test_range_scenario_places_spot_between_distinct_walls():
    _, bundle, _ = _classify("broad_range")
    features = bundle.features
    assert features.call_wall is not None and features.put_wall is not None
    assert features.put_wall < features.spot < features.call_wall
    assert 0.2 < features.position_in_range < 0.8


def test_breakout_scenarios_are_in_negative_gamma():
    for name in ("bull_breakout", "bear_breakdown", "vol_expansion"):
        _, bundle, _ = _classify(name)
        assert bundle.features.in_positive_gamma == 0.0, name


def test_transition_and_unstable_have_no_eligible_families():
    """Spec 11.10, 11.11: these states must produce no trade."""
    for state in (MarketState.TRANSITION, MarketState.UNSTABLE):
        assert eligible_families({state: 1.0}) == frozenset()
