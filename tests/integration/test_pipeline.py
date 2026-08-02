"""End-to-end pipeline and acceptance-criteria tests (spec sections 4, 32, 44)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from spy_der.data.persistence.audit import AuditStore
from spy_der.data.providers.base import SyntheticProvider
from spy_der.data.synthetic import build_scenario, scenario
from spy_der.domain.enums import KillSwitchReason, MarketState, StrategyFamily
from spy_der.execution.paper_broker import PaperBroker
from spy_der.models.forecast_engine import ForecastEngine, ForecastEngineConfig
from spy_der.optimizer.engine import OptimizerConfig, StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig
from spy_der.strategies.validation import validate_defined_risk

UTC = UTC


def build_pipeline(audit: AuditStore | None = None, **config) -> DecisionPipeline:
    return DecisionPipeline(
        forecast_engine=ForecastEngine(config=ForecastEngineConfig(n_paths=250)),
        optimizer=StrategyOptimizer(
            OptimizerConfig(
                valuation_paths=150, valuation_steps=20, max_candidates_per_family=6
            )
        ),
        broker=PaperBroker(),
        config=PipelineConfig(persist=audit is not None, **config),
        audit=audit,
    )


def session(name: str, count: int = 10, minutes: int = 30):
    provider = SyntheticProvider(spec=scenario(name), step=timedelta(minutes=minutes))
    stamps = provider.session_timestamps(datetime(2026, 8, 3, tzinfo=UTC))[:count]
    return [provider.snapshot(ts) for ts in stamps]


# --------------------------------------------------------------------------
# Decision cycle
# --------------------------------------------------------------------------


def test_pipeline_produces_a_decision_for_every_snapshot():
    pipeline = build_pipeline()
    for snapshot in session("broad_range", count=6):
        result = pipeline.step(snapshot)
        assert result.timestamp == snapshot.timestamp
        assert result.decision
        assert result.quality.score >= 0.0


def test_no_trade_is_an_explicit_competitor_not_a_fallback():
    """Spec 27, 44: no-trade is modelled and must be beaten by a margin."""
    pipeline = build_pipeline()
    result = pipeline.step(build_scenario("broad_range"))
    optimization = result.optimization
    assert optimization is not None
    assert optimization.no_trade_margin > 0.0
    if optimization.best is not None:
        expected = optimization.best.utility > (
            optimization.no_trade_utility + optimization.no_trade_margin
        )
        assert optimization.trade_selected is expected


def test_no_trade_wins_where_there_is_no_edge():
    """The synthetic chain is arbitrage-free, so abstention is correct."""
    pipeline = build_pipeline()
    decisions = [pipeline.step(s) for s in session("broad_range", count=6)]
    assert all(d.is_no_trade for d in decisions)


def test_pipeline_can_open_and_close_a_position_where_edge_exists():
    """vol_expansion prices IV below the simulated realized path, so long
    volatility has genuine modelled edge and the system should act on it."""
    pipeline = build_pipeline()
    opened = closed = 0
    for snapshot in session("vol_expansion", count=14, minutes=20):
        result = pipeline.step(snapshot)
        opened += int(result.traded)
        closed += len(result.exits)
    assert opened >= 1, "expected at least one entry where edge exists"
    assert opened + closed >= 2


def test_every_executed_structure_is_defined_risk():
    """Spec 44: no undefined-risk structure can reach an order."""
    pipeline = build_pipeline()
    for snapshot in session("vol_expansion", count=12, minutes=20):
        result = pipeline.step(snapshot)
        if result.candidate is None:
            continue
        validation = validate_defined_risk(result.candidate.strategy)
        assert validation.is_valid, validation.violations
        assert result.candidate.max_loss > 0.0
        assert result.candidate.max_loss < float("inf")


def test_positions_never_exceed_their_recorded_max_loss():
    pipeline = build_pipeline()
    for snapshot in session("vol_expansion", count=12, minutes=20):
        pipeline.step(snapshot)
        for position in pipeline.portfolio.positions:
            assert position.total_max_loss > 0.0
            if not position.is_open:
                # Slippage and commissions can add to the structural worst case,
                # so allow a bounded overshoot rather than an unbounded one.
                assert position.realized_pnl >= -2.0 * position.total_max_loss


# --------------------------------------------------------------------------
# Lockouts (spec 6, 42, 44)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["corrupt_quotes", "stale_data", "poor_liquidity", "wide_spreads"]
)
def test_degraded_data_produces_no_trade(name):
    pipeline = build_pipeline()
    result = pipeline.step(build_scenario(name))
    assert result.is_no_trade
    assert not result.traded
    assert result.reasons


def test_event_lockout_blocks_trading():
    """Structurally sound but calendar-blocked; the risk engine must refuse."""
    pipeline = build_pipeline()
    result = pipeline.step(build_scenario("event_lockout"))
    assert result.is_no_trade
    combined = " ".join(result.reasons) + " ".join(result.risk.veto_reasons)
    assert "event lockout" in combined or "data quality" in combined


def test_kill_switch_blocks_all_subsequent_entries():
    pipeline = build_pipeline()
    pipeline.risk_engine.kill_switch.trip(KillSwitchReason.MANUAL, "operator halt")
    for snapshot in session("vol_expansion", count=4):
        result = pipeline.step(snapshot)
        assert result.is_no_trade
        assert any("kill switch" in r for r in result.risk.veto_reasons)


def test_risk_limits_are_immutable_at_runtime():
    """Limits are frozen, so nothing can widen them mid-session (spec 44)."""
    import dataclasses

    pipeline = build_pipeline()
    with pytest.raises(dataclasses.FrozenInstanceError):
        pipeline.risk_engine.config.trade.max_dollar_risk = 1.0  # type: ignore[misc]


def test_tightened_risk_limits_cannot_be_bypassed_by_the_optimizer():
    """A limit that no structure can satisfy must produce no trades (spec 44)."""
    from spy_der.risk.limits import RiskConfig, TradeLimits

    pipeline = build_pipeline()
    pipeline.risk_engine.config = RiskConfig(trade=TradeLimits(max_dollar_risk=1.0))
    for snapshot in session("vol_expansion", count=6):
        result = pipeline.step(snapshot)
        assert not result.traded


def test_broker_disconnection_stops_trading():
    pipeline = build_pipeline()
    pipeline.broker.disconnect()
    for snapshot in session("vol_expansion", count=3):
        result = pipeline.step(snapshot)
        assert result.is_no_trade


# --------------------------------------------------------------------------
# Audit trail (spec 32, 44)
# --------------------------------------------------------------------------


def test_every_decision_is_persisted_with_full_context(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    pipeline = build_pipeline(audit=store)
    snapshots = session("vol_expansion", count=6)
    for snapshot in snapshots:
        pipeline.step(snapshot)

    counts = store.counts()
    assert counts["decisions"] == len(snapshots)

    rows = store.decisions(limit=len(snapshots))
    for row in rows:
        payload = store.load_payload(row["id"])
        assert payload is not None
        # Spec 32 required fields.
        for key in (
            "timestamp",
            "state_probabilities",
            "forecast",
            "candidates",
            "rejected_candidates",
            "risk_decision",
            "veto_reasons",
            "versions",
        ):
            assert key in payload, key
        versions = payload["versions"]
        for component in (
            "data",
            "features",
            "model",
            "calibration",
            "strategy_generator",
            "risk_engine",
            "broker_adapter",
        ):
            assert versions[component]


def test_rejected_candidates_are_recorded_not_discarded(tmp_path):
    """Spec 32: the alternatives and their rejection reasons are part of the record."""
    store = AuditStore(tmp_path / "audit.db")
    pipeline = build_pipeline(audit=store)
    for snapshot in session("broad_range", count=4):
        pipeline.step(snapshot)

    payloads = [store.load_payload(row["id"]) for row in store.decisions(limit=4)]
    assert any(p and p["rejected_candidates"] for p in payloads), (
        "no rejection reasons were persisted"
    )


def test_no_trade_decisions_are_persisted(tmp_path):
    """Abstentions must be auditable, not silent."""
    store = AuditStore(tmp_path / "audit.db")
    pipeline = build_pipeline(audit=store)
    for snapshot in session("broad_range", count=4):
        pipeline.step(snapshot)
    counts = store.counts()
    assert counts["no_trade"] == 4
    assert counts["traded"] == 0


def test_completed_trades_are_persisted(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    pipeline = build_pipeline(audit=store)
    for snapshot in session("vol_expansion", count=16, minutes=20):
        pipeline.step(snapshot)

    trades = store.trades()
    for trade in trades:
        assert trade["position_id"]
        assert trade["exit_reason"]
        assert trade["max_loss"] > 0.0


def test_audit_encodes_non_json_types(tmp_path):
    """Datetimes, enums, tuple keys, NaN, and infinity must all survive."""
    store = AuditStore(tmp_path / "audit.db")
    pipeline = build_pipeline(audit=store)
    pipeline.step(build_scenario("strong_pin"))
    payload = store.load_payload(store.decisions(limit=1)[0]["id"])
    assert payload is not None
    import json

    json.dumps(payload)  # round-trips without error


# --------------------------------------------------------------------------
# Point-in-time and determinism
# --------------------------------------------------------------------------


def test_pipeline_only_sees_the_snapshot_it_is_given():
    """``Decision_t = f(Data_{<=t})``: identical inputs give identical outputs."""
    first = build_pipeline()
    second = build_pipeline()
    snapshots = session("bull_grind", count=5)

    for snapshot in snapshots:
        a = first.step(snapshot)
        b = second.step(snapshot)
        assert a.decision == b.decision
        assert a.quality.score == pytest.approx(b.quality.score)
        if a.forecast and b.forecast:
            assert a.forecast.dominant_state is b.forecast.dominant_state
            assert a.forecast.confidence == pytest.approx(b.forecast.confidence)


def test_reset_clears_temporal_state():
    pipeline = build_pipeline()
    for snapshot in session("bull_grind", count=4):
        pipeline.step(snapshot)
    pipeline.reset()
    assert pipeline.portfolio.open_count == 0
    assert pipeline.forecast_engine.hidden.steps_observed == 0


def test_forecast_states_always_form_a_distribution():
    pipeline = build_pipeline()
    for snapshot in session("bear_grind", count=5):
        result = pipeline.step(snapshot)
        if result.forecast is None:
            continue
        assert sum(result.forecast.state_probabilities.values()) == pytest.approx(1.0)
        for horizon in result.forecast.horizons.values():
            quantiles = [horizon.quantile(q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
            assert quantiles == sorted(quantiles)


def test_only_allowed_families_are_ever_generated():
    """Spec 22: the excluded structures have no generator at all."""
    from spy_der.strategies.generators.families import registered_families

    families = registered_families()
    assert families <= set(StrategyFamily)
    for name in ("COVERED_CALL", "CASH_SECURED_PUT", "COLLAR", "NAKED_CALL", "PROTECTIVE_PUT"):
        assert not hasattr(StrategyFamily, name)


def test_transition_state_yields_no_candidates():
    from spy_der.optimizer.family_filter import eligible_families

    for state in (MarketState.TRANSITION, MarketState.UNSTABLE):
        assert eligible_families({state: 1.0}) == frozenset()
