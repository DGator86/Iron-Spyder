"""Shared fixtures for the SPY-DER test suite."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from spy_der.data.synthetic import build_scenario
from spy_der.data.validators.quality import assess
from spy_der.domain.enums import OptionRight, StrategyFamily
from spy_der.domain.strategy import OptionLeg, Strategy
from spy_der.features.builder import FeatureContext, build_features
from spy_der.models.forecast_engine import ForecastEngine, ForecastEngineConfig
from spy_der.optimizer.engine import OptimizerConfig, StrategyOptimizer

NEAR = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
FAR = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)


def leg(
    quantity: int,
    strike: float,
    right: OptionRight,
    price: float = 1.0,
    expiry: datetime = NEAR,
    iv: float = 0.15,
) -> OptionLeg:
    return OptionLeg(
        expiry=expiry,
        strike=strike,
        right=right,
        quantity=quantity,
        entry_price=price,
        entry_iv=iv,
    )


@pytest.fixture
def bull_call_spread() -> Strategy:
    """Long 550 call at 4.00, short 555 call at 2.00: debit 2.00, width 5."""
    return Strategy(
        StrategyFamily.BULL_CALL_DEBIT_SPREAD,
        (leg(1, 550, OptionRight.CALL, 4.0), leg(-1, 555, OptionRight.CALL, 2.0)),
    )


@pytest.fixture
def iron_condor() -> Strategy:
    return Strategy(
        StrategyFamily.IRON_CONDOR,
        (
            leg(1, 535, OptionRight.PUT, 0.8),
            leg(-1, 540, OptionRight.PUT, 1.6),
            leg(-1, 560, OptionRight.CALL, 1.5),
            leg(1, 565, OptionRight.CALL, 0.7),
        ),
    )


@pytest.fixture
def long_calendar() -> Strategy:
    """Short near 550 call at 3.00, long far 550 call at 5.00: debit 2.00."""
    return Strategy(
        StrategyFamily.BULLISH_CALENDAR,
        (
            leg(-1, 550, OptionRight.CALL, 3.0, expiry=NEAR),
            leg(1, 550, OptionRight.CALL, 5.0, expiry=FAR),
        ),
    )


@pytest.fixture
def scored_snapshot():
    """A synthetic snapshot with its data-quality score attached."""

    def build(name: str = "broad_range", **overrides):
        snap = build_scenario(name, **overrides)
        return replace(snap, data_quality_score=assess(snap).score)

    return build


@pytest.fixture
def analysed(scored_snapshot):
    """Analytics for a scenario, with temporal features warmed up."""

    def build(name: str = "broad_range", warmup: int = 3):
        snapshot = scored_snapshot(name)
        context = FeatureContext()
        bundle = build_features(snapshot, context)
        for _ in range(warmup - 1):
            bundle = build_features(snapshot, context)
        return snapshot, bundle

    return build


@pytest.fixture
def forecasted(analysed):
    """A forecast and its paths for a scenario."""

    def build(name: str = "broad_range", paths: int = 400, warmup: int = 3):
        snapshot, bundle = analysed(name, warmup=warmup)
        engine = ForecastEngine(config=ForecastEngineConfig(n_paths=paths))
        forecast, path_bundle = engine.forecast(bundle)
        for _ in range(warmup - 1):
            forecast, path_bundle = engine.forecast(bundle)
        return snapshot, bundle, forecast, path_bundle

    return build


@pytest.fixture
def optimizer() -> StrategyOptimizer:
    """A fast optimizer configuration for tests."""
    return StrategyOptimizer(
        OptimizerConfig(valuation_paths=200, valuation_steps=24, max_candidates_per_family=8)
    )


@pytest.fixture
def session_timestamps() -> list[datetime]:
    start = datetime(2026, 8, 3, 13, 30, tzinfo=UTC)
    return [start + timedelta(minutes=30 * i) for i in range(12)]
