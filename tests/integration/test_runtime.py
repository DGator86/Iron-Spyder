"""Autonomous runtime: state durability and supervisor behaviour.

The central property here is that a restart must not become a way to escape a
safety control.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from spy_der.data.providers.base import MarketDataProvider, SyntheticProvider
from spy_der.data.synthetic import scenario
from spy_der.domain.enums import ExitReason, KillSwitchReason, OptionRight, StrategyFamily
from spy_der.domain.strategy import ExitPlan, Strategy
from spy_der.execution.paper_broker import PaperBroker
from spy_der.models.forecast_engine import ForecastEngine, ForecastEngineConfig
from spy_der.optimizer.engine import OptimizerConfig, StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig
from spy_der.runtime.state_store import RuntimeStateStore, StateSchemaMismatch
from spy_der.runtime.supervisor import Supervisor, SupervisorConfig
from tests.conftest import leg

SESSION_DAY = datetime(2026, 8, 3, tzinfo=UTC)


def build_pipeline() -> DecisionPipeline:
    return DecisionPipeline(
        forecast_engine=ForecastEngine(config=ForecastEngineConfig(n_paths=150)),
        optimizer=StrategyOptimizer(
            OptimizerConfig(valuation_paths=100, valuation_steps=16, max_candidates_per_family=4)
        ),
        broker=PaperBroker(),
        config=PipelineConfig(persist=False),
    )


def make_position(
    pipeline: DecisionPipeline,
    *,
    position_id: str = "P-1",
    expiry: datetime | None = None,
):
    from spy_der.domain.execution import Position

    expiry = expiry or datetime(2026, 8, 10, 20, 0, tzinfo=UTC)
    strategy = Strategy(
        StrategyFamily.BULL_CALL_DEBIT_SPREAD,
        (
            leg(1, 550, OptionRight.CALL, 4.0, expiry=expiry),
            leg(-1, 555, OptionRight.CALL, 2.0, expiry=expiry),
        ),
    )
    return Position(
        position_id=position_id,
        opened_at=datetime(2026, 8, 3, 14, 0, tzinfo=UTC),
        family=strategy.family,
        strategy=strategy,
        contracts=2,
        entry_price=2.0,
        max_loss_per_contract=200.0,
        max_profit_per_contract=300.0,
        exit_plan=ExitPlan(0.6, 0.7, 180.0, invalidation_states=("Transition",)),
        entry_commission=2.8,
        current_price=2.2,
        marked_at=datetime(2026, 8, 3, 15, 0, tzinfo=UTC),
    )


# --------------------------------------------------------------------------
# The safety property
# --------------------------------------------------------------------------


def test_a_latched_kill_switch_survives_a_restart(tmp_path):
    """Restarting must not clear a lockout (spec 42, fail closed).

    If it did, "restart the daemon" would be a way to resume trading after a
    daily-loss or drawdown halt without anyone reviewing why it tripped.
    """
    store = RuntimeStateStore(tmp_path / "state.db")

    first = build_pipeline()
    first.risk_engine.kill_switch.trip(KillSwitchReason.DAILY_LOSS, "realized -2000.00")
    store.save(first)

    restarted = build_pipeline()
    assert not restarted.risk_engine.kill_switch.is_tripped  # clean process

    report = store.restore(restarted)
    assert report.has_latched_kill_switch
    assert restarted.risk_engine.kill_switch.is_tripped
    assert KillSwitchReason.DAILY_LOSS in restarted.risk_engine.kill_switch
    assert "realized -2000.00" in " ".join(restarted.risk_engine.kill_switch.reasons)


def test_open_positions_survive_a_restart(tmp_path):
    """A restart must not lose the book while the broker still holds it."""
    store = RuntimeStateStore(tmp_path / "state.db")

    first = build_pipeline()
    position = make_position(first)
    first.portfolio.add(position)
    first.broker.register_position(position)
    store.save(first)

    restarted = build_pipeline()
    assert restarted.portfolio.open_count == 0
    report = store.restore(restarted)

    assert report.open_positions == 1
    assert restarted.portfolio.open_count == 1
    recovered = restarted.portfolio.open_positions[0]
    assert recovered.position_id == position.position_id
    assert recovered.contracts == position.contracts
    assert recovered.entry_price == pytest.approx(position.entry_price)
    # The risk envelope fixed at entry must come back byte-for-byte.
    assert recovered.max_loss_per_contract == pytest.approx(200.0)
    assert recovered.total_max_loss == pytest.approx(position.total_max_loss)
    # Legs round-trip exactly, including signs and expiries.
    assert [leg_.quantity for leg_ in recovered.legs] == [1, -1]
    assert [leg_.strike for leg_ in recovered.legs] == [550.0, 555.0]
    assert recovered.legs[0].expiry == position.legs[0].expiry
    assert recovered.exit_plan.profit_target_fraction == pytest.approx(0.6)


def test_restored_positions_are_visible_to_the_risk_engine(tmp_path):
    """Restored risk must actually count against the portfolio limits."""
    store = RuntimeStateStore(tmp_path / "state.db")
    first = build_pipeline()
    position = make_position(first)
    first.portfolio.add(position)
    store.save(first)

    restarted = build_pipeline()
    store.restore(restarted)
    assert restarted.portfolio.total_open_risk > 0.0
    assert restarted.portfolio.open_count == 1


def test_session_accounting_survives_a_restart(tmp_path):
    """Daily loss and consecutive losses feed the session kill switches."""
    store = RuntimeStateStore(tmp_path / "state.db")

    first = build_pipeline()
    session = first.risk_engine.session
    session.roll_to(SESSION_DAY.date(), 100_000.0)
    session.record_trade(-400.0, 99_600.0)
    session.record_trade(-300.0, 99_300.0)
    session.record_slippage(2.0, 5.0)
    store.save(first)

    restarted = build_pipeline()
    store.restore(restarted)
    recovered = restarted.risk_engine.session
    assert recovered.realized_pnl == pytest.approx(-700.0)
    assert recovered.consecutive_losses == 2
    assert recovered.trades_today == 2
    assert recovered.peak_equity == pytest.approx(100_000.0)
    assert recovered.slippage_ratio == pytest.approx(2.5)


def test_model_state_survives_a_restart(tmp_path):
    """The HMM belief and feature history carry the temporal consistency."""
    store = RuntimeStateStore(tmp_path / "state.db")

    first = build_pipeline()
    from spy_der.domain.enums import MarketState

    emission = {s: (0.7 if s is MarketState.STRONG_PIN else 0.03) for s in MarketState}
    total = sum(emission.values())
    for _ in range(10):
        first.forecast_engine.hidden.update({k: v / total for k, v in emission.items()})
    first.features.push(atm_iv=0.14, realized_vol=0.09, spot=550.0)
    first.features.push(atm_iv=0.15, realized_vol=0.10, spot=551.0)
    before = first.forecast_engine.hidden.probabilities

    restarted = build_pipeline()
    store.save(first)
    store.restore(restarted)

    after = restarted.forecast_engine.hidden.probabilities
    for state, probability in before.items():
        assert after[state] == pytest.approx(probability, abs=1e-9)
    assert restarted.features.iv_history == [0.14, 0.15]
    assert restarted.features.spot_history == [550.0, 551.0]


def test_restore_on_an_empty_store_starts_fresh(tmp_path):
    store = RuntimeStateStore(tmp_path / "state.db")
    pipeline = build_pipeline()
    report = store.restore(pipeline)
    assert not report.restored
    assert "starting fresh" in report.summary()
    assert pipeline.portfolio.open_count == 0


def test_a_schema_change_is_refused_rather_than_misread(tmp_path):
    store = RuntimeStateStore(tmp_path / "state.db")
    store.save(build_pipeline())

    import sqlite3

    with sqlite3.connect(str(tmp_path / "state.db")) as conn:
        conn.execute("UPDATE runtime_state SET schema_version = '0.0.1' WHERE id = 1")
        conn.commit()

    with pytest.raises(StateSchemaMismatch):
        store.load()


def test_an_unrebuildable_position_trips_a_kill_switch(tmp_path):
    """A partial book is worse than no book; refuse to trade on it."""
    store = RuntimeStateStore(tmp_path / "state.db")
    first = build_pipeline()
    first.portfolio.add(make_position(first))
    store.save(first)

    import json
    import sqlite3

    with sqlite3.connect(str(tmp_path / "state.db")) as conn:
        row = conn.execute("SELECT payload FROM runtime_state WHERE id = 1").fetchone()
        payload = json.loads(row[0])
        payload["positions"][0]["family"] = "NotARealFamily"
        conn.execute(
            "UPDATE runtime_state SET payload = ? WHERE id = 1", (json.dumps(payload),)
        )
        conn.commit()

    restarted = build_pipeline()
    report = store.restore(restarted)
    assert report.open_positions == 0
    assert report.warnings
    assert KillSwitchReason.BROKER_STATE in restarted.risk_engine.kill_switch


def test_save_is_idempotent_and_keeps_one_row(tmp_path):
    store = RuntimeStateStore(tmp_path / "state.db")
    pipeline = build_pipeline()
    for _ in range(3):
        store.save(pipeline)

    import sqlite3

    with sqlite3.connect(str(tmp_path / "state.db")) as conn:
        count = conn.execute("SELECT COUNT(*) FROM runtime_state").fetchone()[0]
    assert count == 1


# --------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------


class _ExplodingProvider(MarketDataProvider):
    """Fails a set number of times, then delegates."""

    def __init__(self, inner: MarketDataProvider, failures: int) -> None:
        self.inner = inner
        self.remaining = failures
        self.calls = 0

    def snapshot(self, at):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("simulated provider outage")
        return self.inner.snapshot(at)

    def session_timestamps(self, day):
        return self.inner.session_timestamps(day)


def build_supervisor(tmp_path, provider=None, *, real_sleep=False, **config):
    """A supervisor wired to a synthetic provider.

    ``real_sleep`` keeps the production wait, which is the only way to test that
    a stop actually interrupts one; every other test stubs it out so the suite
    does not spend wall-clock time sleeping.
    """
    pipeline = build_pipeline()
    return Supervisor(
        pipeline=pipeline,
        provider=provider
        or SyntheticProvider(spec=scenario("broad_range"), step=timedelta(minutes=5)),
        state_store=RuntimeStateStore(tmp_path / "state.db"),
        config=SupervisorConfig(**config),
        sleeper=None if real_sleep else (lambda _seconds: None),
    )


def test_supervisor_runs_cycles_and_records_stats(tmp_path):
    supervisor = build_supervisor(tmp_path, max_cycles=3, decision_interval=timedelta(minutes=5))
    supervisor.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)
    stats = supervisor.run()
    assert stats.cycles == 3
    assert stats.decisions == 3
    assert stats.failures == 0


def test_a_failing_cycle_does_not_stop_the_run(tmp_path):
    """One bad snapshot must not end an unattended session."""
    inner = SyntheticProvider(spec=scenario("broad_range"), step=timedelta(minutes=5))
    provider = _ExplodingProvider(inner, failures=2)
    supervisor = build_supervisor(tmp_path, provider=provider, max_cycles=4)
    supervisor.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)

    stats = supervisor.run()
    assert stats.failures == 2
    assert stats.decisions == 2  # the two that succeeded
    assert stats.consecutive_failures == 0  # recovered
    assert stats.last_error is not None


def test_persistent_failures_halt_the_run_with_a_kill_switch(tmp_path):
    """A system that cannot make a decision must stop, not spin."""
    inner = SyntheticProvider(spec=scenario("broad_range"), step=timedelta(minutes=5))
    provider = _ExplodingProvider(inner, failures=99)
    supervisor = build_supervisor(
        tmp_path, provider=provider, max_cycles=50, max_consecutive_failures=3
    )
    supervisor.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)

    stats = supervisor.run()
    assert stats.consecutive_failures == 3
    assert supervisor.pipeline.risk_engine.kill_switch.is_tripped
    assert stats.cycles < 50  # halted early


def test_supervisor_waits_while_the_market_is_closed(tmp_path):
    """Outside a session the loop sleeps instead of deciding."""
    slept: list[float] = []
    supervisor = build_supervisor(tmp_path, max_cycles=2)
    supervisor.clock = lambda: datetime(2026, 8, 1, 15, 0, tzinfo=UTC)  # Saturday
    supervisor.sleeper = lambda seconds: (slept.append(seconds), supervisor.request_stop())[0]

    stats = supervisor.run()
    assert stats.decisions == 0
    assert slept  # it waited rather than trading


def test_startup_restores_state_before_the_first_decision(tmp_path):
    store = RuntimeStateStore(tmp_path / "state.db")
    seeded = build_pipeline()
    seeded.risk_engine.kill_switch.trip(KillSwitchReason.DRAWDOWN, "8% from peak")
    store.save(seeded)

    supervisor = build_supervisor(tmp_path, max_cycles=1)
    supervisor.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)
    report = supervisor.startup()

    assert report.has_latched_kill_switch
    assert supervisor.pipeline.risk_engine.kill_switch.is_tripped


def test_startup_reconciliation_mismatch_trips_a_kill_switch(tmp_path):
    """A restored book the broker does not confirm must block trading."""
    supervisor = build_supervisor(tmp_path, max_cycles=1)
    # Add to the internal book only; the broker never saw it.
    supervisor.pipeline.portfolio.add(make_position(supervisor.pipeline))
    supervisor.startup()
    assert KillSwitchReason.BROKER_STATE in supervisor.pipeline.risk_engine.kill_switch


def test_state_is_saved_after_every_cycle(tmp_path):
    supervisor = build_supervisor(tmp_path, max_cycles=2, save_every_cycle=True)
    supervisor.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)
    supervisor.run()
    assert supervisor.state_store.load() is not None


def test_alerts_are_emitted_and_a_broken_sink_cannot_stop_trading(tmp_path):
    received: list[tuple[str, str]] = []
    supervisor = build_supervisor(tmp_path, max_cycles=1)
    supervisor.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)
    supervisor.alert = lambda severity, message: received.append((severity, message))
    supervisor.pipeline.risk_engine.kill_switch.trip(KillSwitchReason.MANUAL, "test halt")
    supervisor.run()
    assert any(severity == "critical" for severity, _ in received)

    exploding = build_supervisor(tmp_path, max_cycles=1)
    exploding.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)
    exploding.alert = lambda *_: (_ for _ in ()).throw(RuntimeError("pager down"))
    stats = exploding.run()  # must not raise
    assert stats.cycles == 1


def test_request_stop_ends_the_loop(tmp_path):
    supervisor = build_supervisor(tmp_path, max_cycles=100)
    supervisor.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)
    supervisor.request_stop()
    stats = supervisor.run()
    assert stats.cycles == 0


def test_a_stop_interrupts_a_closed_market_wait(tmp_path):
    """SIGTERM during a weekend wait must be acted on now, not in five minutes.

    A process supervisor escalates to SIGKILL on its own schedule — ``docker
    stop`` after ten seconds, systemd after ninety — so a loop that finishes its
    sleep before noticing the flag gets killed mid-flight instead of saving
    state, which is the ungraceful exit the state store exists to survive.
    """
    supervisor = build_supervisor(
        tmp_path, max_cycles=100, real_sleep=True, idle_poll=timedelta(seconds=30)
    )
    supervisor.clock = lambda: datetime(2026, 8, 1, 15, 0, tzinfo=UTC)  # Saturday

    finished = threading.Event()

    def drive():
        supervisor.run()
        finished.set()

    worker = threading.Thread(target=drive, daemon=True)
    started = time.monotonic()
    worker.start()
    time.sleep(0.2)  # let it reach the wait
    supervisor.request_stop()

    assert finished.wait(timeout=10.0), "run() did not return after request_stop()"
    # An uninterruptible sleep would have taken the full 30s idle_poll.
    assert time.monotonic() - started < 5.0


def test_a_stop_interrupts_a_failure_backoff(tmp_path):
    """The same guarantee on the error path, where backoff reaches ten minutes."""
    inner = SyntheticProvider(spec=scenario("broad_range"), step=timedelta(minutes=5))
    supervisor = build_supervisor(
        tmp_path,
        provider=_ExplodingProvider(inner, failures=100),
        max_cycles=100,
        real_sleep=True,
        failure_backoff=timedelta(seconds=30),
    )
    supervisor.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)

    finished = threading.Event()
    worker = threading.Thread(target=lambda: (supervisor.run(), finished.set()), daemon=True)
    started = time.monotonic()
    worker.start()
    time.sleep(0.2)
    supervisor.request_stop()

    assert finished.wait(timeout=10.0)
    assert time.monotonic() - started < 5.0


def test_the_closed_market_wait_is_announced_once_per_target(tmp_path, caplog):
    """Silence over a weekend is indistinguishable from a hung process.

    But repeating the line every idle poll buries anything that matters, so the
    target is announced once and re-announced only when it changes.
    """
    polls = {"n": 0}

    def poll(_seconds):
        polls["n"] += 1
        if polls["n"] >= 4:
            supervisor.request_stop()

    supervisor = build_supervisor(tmp_path, max_cycles=100, idle_poll=timedelta(minutes=5))
    supervisor.clock = lambda: datetime(2026, 8, 1, 15, 0, tzinfo=UTC)  # Saturday
    supervisor.sleeper = poll

    with caplog.at_level(logging.INFO, logger="spy_der.supervisor"):
        supervisor.run()

    announcements = [
        r for r in caplog.records if "waiting for the open" in r.getMessage()
    ]
    assert len(announcements) == 1
    # Monday 2026-08-03 09:30 ET is 13:30 UTC.
    assert "2026-08-03T13:30" in announcements[0].getMessage()
    assert polls["n"] >= 4


def test_shutdown_leaves_open_positions_in_place(tmp_path):
    """Dumping a defined-risk book at shutdown is usually worse than holding it."""
    supervisor = build_supervisor(tmp_path, max_cycles=0)
    position = make_position(supervisor.pipeline)
    supervisor.pipeline.portfolio.add(position)
    supervisor.pipeline.broker.register_position(position)
    warnings: list[tuple[str, str]] = []
    supervisor.alert = lambda severity, message: warnings.append((severity, message))
    supervisor.clock = lambda: datetime(2026, 8, 3, 15, 0, tzinfo=UTC)

    supervisor.run()
    assert supervisor.pipeline.portfolio.open_count == 1
    assert any("open position" in message for _, message in warnings)
    # And the position is recoverable on the next start.
    restarted = build_pipeline()
    assert supervisor.state_store.restore(restarted).open_positions == 1


def test_session_end_flatten_uses_the_real_close(tmp_path):
    """The flatten must fire against the actual bell, including a half day."""
    supervisor = build_supervisor(
        tmp_path, max_cycles=1, decision_interval=timedelta(minutes=5)
    )
    # 12:57 ET on Black Friday: three minutes to a 13:00 close.
    now = datetime(2026, 11, 27, 17, 57, tzinfo=UTC)
    supervisor.clock = lambda: now
    supervisor.provider = SyntheticProvider(
        spec=scenario("broad_range"), step=timedelta(minutes=5)
    )

    # Hold contracts the snapshot actually quotes, or the close cannot be
    # priced and this would test the stranded-position path instead.
    chain_expiry = supervisor.provider.snapshot(now).expirations[-1]
    position = make_position(supervisor.pipeline, expiry=chain_expiry)
    supervisor.pipeline.portfolio.add(position)
    supervisor.pipeline.broker.register_position(position)

    supervisor.run()
    assert supervisor.pipeline.portfolio.open_count == 0
    assert not supervisor.pipeline.risk_engine.kill_switch.is_tripped


def test_a_position_that_cannot_be_priced_escalates_rather_than_lingering(tmp_path):
    """An unattended run must not carry an unclosable position past the bell."""
    supervisor = build_supervisor(
        tmp_path, max_cycles=1, decision_interval=timedelta(minutes=5)
    )
    # An expiry no generated chain contains, so the close cannot be priced.
    position = make_position(
        supervisor.pipeline, expiry=datetime(2030, 1, 18, 21, 0, tzinfo=UTC)
    )
    supervisor.pipeline.portfolio.add(position)
    supervisor.pipeline.broker.register_position(position)
    alerts: list[tuple[str, str]] = []
    supervisor.alert = lambda severity, message: alerts.append((severity, message))
    supervisor.clock = lambda: datetime(2026, 11, 27, 17, 57, tzinfo=UTC)

    supervisor.run()
    assert supervisor.pipeline.portfolio.open_count == 1  # honestly still open
    assert KillSwitchReason.BROKER_STATE in supervisor.pipeline.risk_engine.kill_switch
    assert any("could not be priced" in message for _, message in alerts)


def test_no_flatten_early_in_the_session(tmp_path):
    supervisor = build_supervisor(tmp_path, max_cycles=1)
    position = make_position(supervisor.pipeline)
    supervisor.pipeline.portfolio.add(position)
    supervisor.pipeline.broker.register_position(position)
    supervisor.clock = lambda: datetime(2026, 8, 3, 14, 0, tzinfo=UTC)  # 10:00 ET
    supervisor.run()
    assert supervisor.pipeline.portfolio.open_count == 1


def test_force_close_all_records_an_exit_reason(tmp_path):
    from spy_der.pipeline import force_close_all

    supervisor = build_supervisor(tmp_path, max_cycles=0)
    position = make_position(supervisor.pipeline)
    supervisor.pipeline.portfolio.add(position)
    supervisor.pipeline.broker.register_position(position)
    snapshot = supervisor.provider.snapshot(datetime(2026, 8, 3, 19, 55, tzinfo=UTC))

    closed = force_close_all(supervisor.pipeline, snapshot, ExitReason.END_OF_SESSION)
    for entry in closed:
        assert entry.exit_reason is ExitReason.END_OF_SESSION
        assert entry.closed_at is not None
