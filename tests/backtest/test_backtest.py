"""Backtest, walk-forward, purging, and cost stress (spec sections 37.4-37.8)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from spy_der.backtest.engine import (
    STRESS_PROFILES,
    StressProfile,
    compare_to_baselines,
    run_backtest,
    run_cost_stress,
    stress_summary,
    survives_stress,
)
from spy_der.backtest.metrics import max_drawdown, parameter_stability, summarize
from spy_der.backtest.walk_forward import purge_indices, rolling_windows, split_indices
from spy_der.data.providers.base import ReplayProvider, SyntheticProvider
from spy_der.data.synthetic import scenario
from spy_der.execution.fills import CostModel
from spy_der.execution.paper_broker import PaperBroker
from spy_der.models.forecast_engine import ForecastEngine, ForecastEngineConfig
from spy_der.optimizer.engine import OptimizerConfig, StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig

UTC = UTC


def build_pipeline(costs: CostModel | None = None) -> DecisionPipeline:
    model = costs or CostModel()
    return DecisionPipeline(
        forecast_engine=ForecastEngine(config=ForecastEngineConfig(n_paths=200)),
        optimizer=StrategyOptimizer(
            OptimizerConfig(
                valuation_paths=120,
                valuation_steps=20,
                max_candidates_per_family=6,
                costs=model,
            )
        ),
        broker=PaperBroker(costs=model),
        config=PipelineConfig(costs=model, persist=False),
    )


def snapshots(name: str = "vol_expansion", count: int = 8):
    provider = SyntheticProvider(spec=scenario(name), step=timedelta(minutes=30))
    stamps = provider.session_timestamps(datetime(2026, 8, 3, tzinfo=UTC))[:count]
    return [provider.snapshot(ts) for ts in stamps]


# --------------------------------------------------------------------------
# Walk-forward windows (spec 37.5)
# --------------------------------------------------------------------------


def test_rolling_windows_follow_the_spec_layout():
    windows = rolling_windows(
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2026, 6, 1, tzinfo=UTC),
    )
    assert windows
    first = windows[0]
    assert (first.train_end - first.train_start).days == 182
    assert (first.validate_end - first.validate_start).days == 30
    assert (first.test_end - first.test_start).days == 30
    # Folds roll forward and never overlap in their test ranges.
    for earlier, later in zip(windows, windows[1:]):
        assert later.test_start > earlier.test_start
        assert later.train_start > earlier.train_start


def test_windows_are_ordered_train_validate_test():
    for window in rolling_windows(
        datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC)
    ):
        assert window.train_end == window.validate_start
        assert window.validate_end == window.test_start
        assert window.train_start < window.test_start


def test_rolling_windows_reject_degenerate_durations():
    with pytest.raises(ValueError, match="positive"):
        rolling_windows(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            train=timedelta(0),
        )


# --------------------------------------------------------------------------
# Purging and embargo (spec 37.6)
# --------------------------------------------------------------------------


def test_purging_removes_labels_that_resolve_inside_the_test_window():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [base + timedelta(days=i) for i in range(40)]
    train_range = (base, base + timedelta(days=20))
    test_range = (base + timedelta(days=20), base + timedelta(days=30))

    unpurged = purge_indices(
        timestamps,
        train_range=train_range,
        test_range=test_range,
        label_horizon=timedelta(0),
    )
    purged = purge_indices(
        timestamps,
        train_range=train_range,
        test_range=test_range,
        label_horizon=timedelta(days=5),
    )
    assert len(purged) < len(unpurged)
    # Day 15's label resolves exactly at the test boundary, which is not an
    # overlap, so it survives; days 16-19 resolve strictly inside and are cut.
    assert max(purged) == 15


def test_embargo_excludes_observations_just_after_the_test_window():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [base + timedelta(days=i) for i in range(60)]
    # A training range positioned after the test range exercises the embargo.
    kept = purge_indices(
        timestamps,
        train_range=(base + timedelta(days=30), base + timedelta(days=60)),
        test_range=(base + timedelta(days=20), base + timedelta(days=30)),
        label_horizon=timedelta(0),
        embargo=timedelta(days=5),
    )
    assert all(timestamps[i] >= base + timedelta(days=35) for i in kept)


def test_split_indices_purge_validation_against_test():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [base + timedelta(days=i) for i in range(300)]
    window = rolling_windows(base, base + timedelta(days=300))[0]
    train, validate, test = split_indices(
        timestamps, window, label_horizon=timedelta(days=3), embargo=timedelta(days=2)
    )
    assert train and validate and test
    assert set(train).isdisjoint(test)
    assert set(validate).isdisjoint(test)
    # No purged training observation's label may reach into the test range.
    for i in train:
        assert timestamps[i] + timedelta(days=3) <= window.test_start


def test_negative_horizon_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        purge_indices(
            [datetime(2026, 1, 1, tzinfo=UTC)],
            train_range=(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)),
            test_range=(datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
            label_horizon=timedelta(days=-1),
        )


# --------------------------------------------------------------------------
# Point-in-time replay (spec 37.4)
# --------------------------------------------------------------------------


def test_replay_provider_never_looks_ahead():
    """``Decision_t = f(Data_{<=t})`` enforced structurally."""
    data = snapshots(count=5)
    provider = ReplayProvider(snapshots=tuple(data))
    for snapshot in data:
        assert provider.snapshot(snapshot.timestamp).timestamp <= snapshot.timestamp

    midpoint = data[2].timestamp - timedelta(seconds=1)
    assert provider.snapshot(midpoint).timestamp <= midpoint


def test_replay_provider_raises_before_the_first_snapshot():
    data = snapshots(count=3)
    provider = ReplayProvider(snapshots=tuple(data))
    with pytest.raises(LookupError):
        provider.snapshot(data[0].timestamp - timedelta(days=1))


def test_backtest_rejects_out_of_order_snapshots():
    data = snapshots(count=4)
    with pytest.raises(ValueError, match="chronological"):
        run_backtest(build_pipeline(), list(reversed(data)))


def test_backtest_runs_and_produces_an_equity_curve():
    data = snapshots(count=8)
    result = run_backtest(build_pipeline(), data)
    assert result.decision_count == len(data)
    assert len(result.equity_curve) >= len(data)
    assert 0.0 <= result.no_trade_rate <= 1.0


def test_backtest_prices_option_pnl_from_the_chain():
    """Spec 37.4 forbids estimating option P&L from SPY movement alone."""
    result = run_backtest(build_pipeline(), snapshots(count=10))
    for position in result.closed_positions:
        assert position.exit_reason is not None
        # Entry price came from a quoted structure, not a spot-derived proxy.
        assert position.entry_price != 0.0


# --------------------------------------------------------------------------
# Cost stress (spec 37.7)
# --------------------------------------------------------------------------


def test_stress_grid_covers_the_required_profiles():
    names = {p.name for p in STRESS_PROFILES}
    assert {
        "recorded",
        "spread_1.25x",
        "spread_1.50x",
        "spread_2.00x",
        "one_tick_adverse",
        "two_tick_adverse",
    } <= names


def test_stress_profiles_widen_the_cost_model():
    base = CostModel()
    for profile in STRESS_PROFILES:
        stressed = profile.apply(base)
        assert stressed.spread_multiplier >= base.spread_multiplier
        assert stressed.adverse_ticks >= base.adverse_ticks


def test_cost_stress_runs_every_profile_on_a_fresh_pipeline():
    data = snapshots(count=6)
    profiles = (
        StressProfile("recorded", 1.0, 0.0),
        StressProfile("spread_2.00x", 2.0, 0.0),
    )
    results = run_cost_stress(build_pipeline, data, profiles=profiles)
    assert set(results) == {"recorded", "spread_2.00x"}

    summary = stress_summary(results)
    assert set(summary) == {"recorded", "spread_2.00x"}
    for stats in summary.values():
        assert "expectancy" in stats and "no_trade_rate" in stats


def test_cost_stress_requires_a_factory_not_a_pipeline():
    """Reusing one pipeline would carry equity and state across profiles."""
    with pytest.raises(TypeError, match="callable"):
        run_cost_stress(build_pipeline(), snapshots(count=2))


def test_survives_stress_distinguishes_losing_from_abstaining():
    """A losing profile fails; a profile that took no trades does not.

    Built from explicit results rather than a live run, so the assertion tests
    the rule instead of whichever way the synthetic run happened to go.
    """
    from spy_der.backtest.engine import BacktestResult
    from spy_der.domain.enums import ExitReason, OptionRight, StrategyFamily
    from spy_der.domain.execution import Position
    from spy_der.domain.strategy import ExitPlan, Strategy
    from tests.conftest import leg

    def result_with(pnl: list[float]) -> BacktestResult:
        strategy = Strategy(
            StrategyFamily.BULL_CALL_DEBIT_SPREAD,
            (leg(1, 550, OptionRight.CALL, 4.0), leg(-1, 555, OptionRight.CALL, 2.0)),
        )
        positions = [
            Position(
                position_id=f"P{i}",
                opened_at=datetime(2026, 8, 3, 14, tzinfo=UTC),
                family=strategy.family,
                strategy=strategy,
                contracts=1,
                entry_price=2.0,
                max_loss_per_contract=200.0,
                max_profit_per_contract=300.0,
                exit_plan=ExitPlan(0.6, 0.7, 120.0),
                realized_pnl=value,
                closed_at=datetime(2026, 8, 3, 15, tzinfo=UTC),
                exit_reason=ExitReason.PROFIT_TARGET,
            )
            for i, value in enumerate(pnl)
        ]
        return BacktestResult(
            profile="recorded", closed_positions=positions, equity_curve=[(None, 100_000.0)]
        )

    through = "recorded"
    assert survives_stress({"recorded": result_with([50.0, 30.0])}, require_positive_through=through) is True
    assert survives_stress({"recorded": result_with([-50.0, -30.0])}, require_positive_through=through) is False
    # No trades is abstention, not a failure to survive.
    assert survives_stress({"recorded": result_with([])}, require_positive_through=through) is True
    # A missing profile cannot be claimed to have survived.
    assert survives_stress({}, require_positive_through=through) is False
    # The default demands the profile grid up through 1.50x spreads.
    assert survives_stress({"recorded": result_with([50.0])}) is False

    with pytest.raises(ValueError, match="unknown stress profile"):
        survives_stress({}, require_positive_through="nonexistent")


# --------------------------------------------------------------------------
# Metrics and parameter stability (spec 37.8, 38)
# --------------------------------------------------------------------------


def test_metrics_summary_arithmetic():
    metrics = summarize(
        pnl=[100.0, -50.0, 200.0, -25.0],
        equity_curve=[100_000, 100_100, 100_050, 100_250, 100_225],
        no_trade_rate=0.5,
    )
    assert metrics.trade_count == 4
    assert metrics.total_pnl == pytest.approx(225.0)
    assert metrics.expectancy == pytest.approx(56.25)
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(300.0 / 75.0)
    assert metrics.is_profitable


def test_max_drawdown_finds_the_worst_peak_to_trough():
    dollars, fraction = max_drawdown([100.0, 120.0, 90.0, 110.0, 80.0])
    assert dollars == pytest.approx(40.0)  # 120 down to 80
    assert fraction == pytest.approx(40.0 / 120.0)


def test_empty_metrics_are_well_defined():
    metrics = summarize(pnl=[], equity_curve=[], no_trade_rate=1.0)
    assert metrics.trade_count == 0
    assert metrics.expectancy == 0.0
    assert not metrics.is_profitable


def test_parameter_stability_prefers_a_plateau():
    """Spec 37.8: a single-point optimum is a sign of overfitting."""
    plateau = parameter_stability({0.1: 90.0, 0.2: 100.0, 0.3: 95.0})
    spike = parameter_stability({0.1: 5.0, 0.2: 100.0, 0.3: 4.0})
    assert plateau["is_plateau"]
    assert not spike["is_plateau"]
    assert plateau["best_parameter"] == pytest.approx(0.2)


# --------------------------------------------------------------------------
# Baseline comparison (spec 37.10)
# --------------------------------------------------------------------------


def test_baseline_comparison_reports_the_advantage():
    result = run_backtest(build_pipeline(), snapshots(count=4))
    comparison = compare_to_baselines(
        result,
        {
            "always_long_spy": [10.0, -30.0, 5.0],
            "random_selection": [-5.0, -5.0, -5.0],
            "moving_average_breakout": [1.0, 1.0, -20.0],
        },
    )
    assert "system" in comparison
    for name in ("always_long_spy", "random_selection", "moving_average_breakout"):
        assert "system_advantage" in comparison[name]
        assert comparison[name]["system_advantage"] == pytest.approx(
            comparison["system"]["expectancy"] - comparison[name]["expectancy"]
        )
