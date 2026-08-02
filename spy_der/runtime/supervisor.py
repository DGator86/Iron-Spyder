"""Autonomous supervisor: the loop that runs the system unattended.

Responsibilities, in the order they matter:

**Fail closed on startup.** Restore prior state before the first decision, and
reconcile the restored book against the broker. A mismatch trips a kill switch
rather than trading against a book that does not exist.

**Contain per-cycle faults.** A single bad snapshot, a provider timeout, or an
unexpected exception must not end the run. Failures are counted; consecutive
failures back off and eventually trip a kill switch, so a persistently broken
system stops trading instead of retrying forever.

**Persist after every cycle.** State is written after each decision, so the
worst case a crash can cost is one cycle rather than the whole session.

**Sleep on the calendar, not on a fixed timer.** Outside a session the loop
waits for the next open instead of spinning.

**Shut down gracefully.** SIGTERM and SIGINT stop the loop at a cycle boundary
and save state. Waits are interruptible, so a signal that lands mid-sleep is
acted on at once rather than whenever the sleep happens to end — a process
supervisor escalates to SIGKILL on its own schedule (``docker stop`` after ten
seconds, systemd after ninety), and being killed mid-sleep is exactly the
ungraceful exit the durable state store exists to survive.

Open positions are *not* auto-liquidated on shutdown: dumping a defined-risk
book into whatever bid exists at that instant is usually worse than leaving it
with its stops intact, and the operator may simply be restarting the process.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from spy_der.data.providers.base import MarketDataProvider
from spy_der.domain.enums import ExitReason, KillSwitchReason
from spy_der.execution.reconciliation import reconcile
from spy_der.pipeline import DecisionPipeline, PipelineResult, force_close_all
from spy_der.runtime import calendar as market_calendar
from spy_der.runtime.state_store import RestoreReport, RuntimeStateStore

logger = logging.getLogger("spy_der.supervisor")



@dataclass
class SupervisorConfig:
    decision_interval: timedelta = timedelta(minutes=5)
    """Cadence of decisions inside a session."""

    close_positions_at_session_end: bool = True
    """Flatten before the bell.

    Distinct from shutdown behaviour: this is a scheduled, in-session exit with
    the market still liquid, which is exactly what the exit plans assume.
    """

    max_consecutive_failures: int = 5
    """Consecutive failed cycles before the run is halted by a kill switch."""

    failure_backoff: timedelta = timedelta(seconds=30)
    max_failure_backoff: timedelta = timedelta(minutes=10)
    heartbeat_interval: timedelta = timedelta(minutes=15)
    idle_poll: timedelta = timedelta(minutes=5)
    """Cap on how long the loop sleeps while the market is closed, so a
    shutdown signal is noticed promptly rather than after hours."""

    save_every_cycle: bool = True
    max_cycles: int | None = None
    """Stop after this many cycles. Used by tests and dry runs."""


@dataclass
class SupervisorStats:
    cycles: int = 0
    decisions: int = 0
    trades: int = 0
    exits: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    started_at: datetime | None = None
    last_cycle_at: datetime | None = None
    last_error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "decisions": self.decisions,
            "trades": self.trades,
            "exits": self.exits,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "last_error": self.last_error,
        }


@dataclass
class Supervisor:
    """Runs a :class:`DecisionPipeline` unattended."""

    pipeline: DecisionPipeline
    provider: MarketDataProvider
    state_store: RuntimeStateStore
    config: SupervisorConfig = field(default_factory=SupervisorConfig)
    alert: Callable[[str, str], None] | None = None
    """Optional ``(severity, message)`` sink for out-of-band notification."""

    clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC)

    sleeper: Callable[[float], None] | None = None
    """Override the wait, for tests. ``None`` uses the interruptible default."""

    stats: SupervisorStats = field(default_factory=SupervisorStats)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _last_heartbeat: datetime | None = field(default=None, init=False)
    _closed_session: object = field(default=None, init=False)
    _awaiting_open: datetime | None = field(default=None, init=False)

    # -- lifecycle ---------------------------------------------------------

    @property
    def _stopping(self) -> bool:
        return self._stop_event.is_set()

    def install_signal_handlers(self) -> None:
        """Stop at the next cycle boundary on SIGTERM/SIGINT.

        Only valid on the main thread; callers embedding the supervisor
        elsewhere should drive :meth:`request_stop` themselves.
        """
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle_signal)
            except ValueError:  # pragma: no cover - not on the main thread
                logger.warning("could not install handler for %s", sig)

    def _handle_signal(self, signum: int, _frame: object) -> None:  # pragma: no cover
        logger.info("received signal %s; stopping", signum)
        self.request_stop()

    def request_stop(self) -> None:
        """Ask the loop to stop, waking it if it is sleeping."""
        self._stop_event.set()

    def _sleep(self, seconds: float) -> None:
        """Wait, returning early if a stop is requested.

        ``time.sleep`` would hold the process for the full duration — up to
        ``idle_poll`` over a weekend — so a SIGTERM would sit unacted-on until
        it expired and the process supervisor would SIGKILL first.
        """
        if seconds <= 0:
            return
        if self.sleeper is not None:
            self.sleeper(seconds)
            return
        self._stop_event.wait(seconds)

    def startup(self) -> RestoreReport:
        """Restore state and verify the book before the first decision."""
        self.stats.started_at = self.clock()

        report = self.state_store.restore(self.pipeline)
        logger.info("startup: %s", report.summary())
        for warning in report.warnings:
            logger.warning("startup: %s", warning)
            self._notify("warning", f"state restore: {warning}")

        if report.has_latched_kill_switch:
            # Surfaced loudly: the system is up but deliberately not trading.
            message = (
                "started with latched kill switch(es): "
                + ", ".join(report.kill_switches)
                + " — no trades will be placed until reset"
            )
            logger.warning(message)
            self._notify("warning", message)

        self._verify_book()
        return report

    def _verify_book(self) -> None:
        """Compare the restored book against the broker (spec 44)."""
        internal = tuple(self.pipeline.portfolio.open_positions)
        broker_positions = self.pipeline.broker.positions()
        result = reconcile(internal, broker_positions)
        if result.is_reconciled:
            logger.info("startup reconciliation: %s", result.summary())
            return

        message = "startup reconciliation failed: " + "; ".join(result.discrepancies)
        logger.error(message)
        self.pipeline.risk_engine.kill_switch.trip(KillSwitchReason.BROKER_STATE, message)
        self._notify("critical", message)

    # -- main loop ---------------------------------------------------------

    def run(self) -> SupervisorStats:
        """Run until stopped, or until ``max_cycles`` is reached."""
        self.startup()

        while not self._stopping:
            if self.config.max_cycles is not None and self.stats.cycles >= self.config.max_cycles:
                logger.info("reached max_cycles=%s; stopping", self.config.max_cycles)
                break

            now = self.clock()
            session = market_calendar.current_session(now)

            if session is None:
                self._on_market_closed(now)
                continue

            self._maybe_close_for_session_end(session, now)
            self.run_cycle(now)
            self.stats.cycles += 1
            self._heartbeat(now)

            if not self._stopping:
                self._sleep_until_next_decision(now)

        self._shutdown()
        return self.stats

    def run_cycle(self, now: datetime | None = None) -> PipelineResult | None:
        """One decision cycle, with faults contained.

        Returns ``None`` when the cycle failed; the exception is logged and
        counted rather than propagated, because an unattended run must survive
        a single bad snapshot.
        """
        moment = now or self.clock()
        try:
            snapshot = self.provider.snapshot(moment)
            result = self.pipeline.step(snapshot)
        except Exception as exc:  # noqa: BLE001 - containment is the point
            self._on_failure(exc)
            return None

        self.stats.consecutive_failures = 0
        self.stats.decisions += 1
        self.stats.trades += int(result.traded)
        self.stats.exits += len(result.exits)
        self.stats.last_cycle_at = moment

        self._log_result(result)
        self._check_kill_switch()

        if self.config.save_every_cycle:
            self._save()
        return result

    def _log_result(self, result: PipelineResult) -> None:
        if result.traded:
            logger.info(
                "%s TRADE %s", result.timestamp.isoformat(timespec="seconds"), result.decision
            )
            self._notify("info", f"opened {result.decision}")
        for signal_ in result.exits:
            logger.info(
                "%s EXIT %s: %s",
                result.timestamp.isoformat(timespec="seconds"),
                signal_.reason.value,
                signal_.detail,
            )
            self._notify("info", f"closed {signal_.position_id}: {signal_.reason.value}")
        if not result.traded and not result.exits:
            reason = result.reasons[0] if result.reasons else "no candidate cleared the bar"
            logger.debug(
                "%s no_trade: %s", result.timestamp.isoformat(timespec="seconds"), reason
            )

    def _on_failure(self, exc: Exception) -> None:
        self.stats.failures += 1
        self.stats.consecutive_failures += 1
        self.stats.last_error = f"{type(exc).__name__}: {exc}"
        logger.exception("decision cycle failed (%s consecutive)", self.stats.consecutive_failures)

        if self.stats.consecutive_failures >= self.config.max_consecutive_failures:
            message = (
                f"{self.stats.consecutive_failures} consecutive cycle failures; "
                f"halting. Last error: {self.stats.last_error}"
            )
            self.pipeline.risk_engine.kill_switch.trip(KillSwitchReason.MANUAL, message)
            self._notify("critical", message)
            self.request_stop()
            return

        # Back off geometrically so a transient outage is retried gently and a
        # persistent one does not hammer the provider.
        delay = min(
            self.config.failure_backoff * (2 ** (self.stats.consecutive_failures - 1)),
            self.config.max_failure_backoff,
        )
        self._notify("warning", f"cycle failed: {self.stats.last_error}")
        self._sleep(delay.total_seconds())

    def _check_kill_switch(self) -> None:
        switch = self.pipeline.risk_engine.kill_switch
        if switch.is_tripped:
            message = "kill switch active: " + "; ".join(switch.reasons)
            logger.warning(message)
            self._notify("critical", message)

    # -- scheduling --------------------------------------------------------

    def _on_market_closed(self, now: datetime) -> None:
        """Wait for the next open, in bounded increments."""
        upcoming = market_calendar.next_session(now)
        if upcoming is None:  # pragma: no cover - implies a broken calendar
            logger.error("no upcoming session found; stopping")
            self.request_stop()
            return

        wait = min(
            (upcoming.open_at - now).total_seconds(), self.config.idle_poll.total_seconds()
        )
        # Announce each new target once at INFO, then poll quietly. An
        # unattended process that logs nothing for a weekend is
        # indistinguishable from a hung one, but repeating the same line every
        # five minutes buries anything that matters.
        if upcoming.open_at != self._awaiting_open:
            self._awaiting_open = upcoming.open_at
            logger.info(
                "market closed; waiting for the open at %s",
                upcoming.open_at.isoformat(timespec="minutes"),
            )
        else:
            logger.debug("market closed; sleeping %.0fs", wait)
        self._sleep(wait)

    def _sleep_until_next_decision(self, now: datetime) -> None:
        elapsed = (self.clock() - now).total_seconds()
        self._sleep(self.config.decision_interval.total_seconds() - elapsed)

    def _maybe_close_for_session_end(self, session: object, now: datetime) -> None:
        """Flatten before the bell, once per session."""
        if not self.config.close_positions_at_session_end:
            return
        if self._closed_session == getattr(session, "day", None):
            return
        if not self.pipeline.portfolio.open_positions:
            return

        minutes_left = market_calendar.minutes_to_close(now)
        # One interval of headroom, so the close order has a cycle to fill in.
        threshold = max(
            5.0, self.config.decision_interval.total_seconds() / 60.0 + 1.0
        )
        if minutes_left > threshold:
            return

        intended = self.pipeline.portfolio.open_count
        logger.info("session end in %.0f minutes; flattening", minutes_left)
        try:
            snapshot = self.provider.snapshot(now)
            closed = force_close_all(self.pipeline, snapshot, ExitReason.END_OF_SESSION)
        except Exception as exc:  # noqa: BLE001
            logger.exception("session-end flatten failed")
            self._notify("critical", f"session-end flatten failed: {exc}")
            return

        self._closed_session = getattr(session, "day", None)
        for position in closed:
            logger.info("session-end close %s pnl=%.2f", position.position_id, position.realized_pnl)
        if closed:
            self._notify("info", f"flattened {len(closed)} position(s) at session end")

        # A position the close could not price stays open by design — marking
        # it closed without an offsetting fill would desynchronize the book
        # from the broker. But an unattended run must not carry that quietly
        # into the close, so it escalates instead of being retried forever.
        stranded = self.pipeline.portfolio.open_count
        if stranded:
            message = (
                f"session-end flatten closed {len(closed)} of {intended}; "
                f"{stranded} position(s) could not be priced and remain open"
            )
            logger.error(message)
            self.pipeline.risk_engine.kill_switch.trip(
                KillSwitchReason.BROKER_STATE, message
            )
            self._notify("critical", message)
        self._save()

    # -- housekeeping ------------------------------------------------------

    def _heartbeat(self, now: datetime) -> None:
        if (
            self._last_heartbeat is not None
            and now - self._last_heartbeat < self.config.heartbeat_interval
        ):
            return
        self._last_heartbeat = now
        logger.info(
            "heartbeat cycles=%s decisions=%s trades=%s exits=%s open=%s equity=%.2f",
            self.stats.cycles,
            self.stats.decisions,
            self.stats.trades,
            self.stats.exits,
            self.pipeline.portfolio.open_count,
            self.pipeline.broker.equity,
        )

    def _save(self) -> None:
        try:
            self.state_store.save(self.pipeline, now=self.clock())
        except Exception as exc:  # noqa: BLE001
            # A failed save is serious — the next restart would restore a stale
            # book — but it must not take down a running system that is
            # otherwise healthy.
            logger.exception("failed to persist runtime state")
            self._notify("critical", f"state save failed: {exc}")

    def _shutdown(self) -> None:
        logger.info("shutting down: %s", self.stats.as_dict())
        self._save()
        open_count = self.pipeline.portfolio.open_count
        if open_count:
            # Deliberate: see the module docstring. Say so loudly instead.
            message = (
                f"stopped with {open_count} open position(s); they were left in place "
                "with their stops and will be restored on restart"
            )
            logger.warning(message)
            self._notify("warning", message)

    def _notify(self, severity: str, message: str) -> None:
        if self.alert is None:
            return
        try:
            self.alert(severity, message)
        except Exception:  # noqa: BLE001 - a broken alert sink must not stop trading
            logger.exception("alert sink raised")


def log_alert(severity: str, message: str) -> None:
    """Default alert sink: route to the logger at a matching level."""
    level = {"info": logging.INFO, "warning": logging.WARNING}.get(severity, logging.CRITICAL)
    logging.getLogger("spy_der.alerts").log(level, message)
