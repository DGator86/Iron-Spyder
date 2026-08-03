"""VPS supervisor service — the long-running unit on the box.

Wraps :class:`~spy_der.runtime.supervisor.Supervisor` and publishes:

* a heartbeat under ``<state_root>/health/supervisor.json``
* ``<state_root>/live_state.json`` for the dashboard API

Decision math stays in the pipeline. This process owns only process lifecycle
and the file surface other VPS services read.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from spy_der.config.settings import Mode, load_settings
from spy_der.data.persistence.audit import AuditStore
from spy_der.data.providers.base import MarketDataProvider, SyntheticProvider
from spy_der.data.providers.tradier import TradierError, TradierProvider
from spy_der.data.synthetic import SCENARIOS, scenario
from spy_der.execution.paper_broker import PaperBroker, PaperBrokerConfig
from spy_der.models.forecast_engine import ForecastEngine
from spy_der.optimizer.engine import StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig, PipelineResult
from spy_der.risk.engine import RiskEngine
from spy_der.runtime.state_store import RuntimeStateStore
from spy_der.runtime.supervisor import Supervisor, SupervisorConfig, SupervisorStats, log_alert
from spy_der.vps.heartbeat import write_heartbeat
from spy_der.vps.live_state import (
    STATUS_RUNNING,
    STATUS_STARTING,
    STATUS_STOPPED,
    build_live_state,
    write_live_state,
)
from spy_der.vps.paths import ensure_state_tree, state_paths

__all__ = ["VpsService", "VpsServiceConfig", "build_arg_parser", "build_service", "main"]

log = logging.getLogger("spy_der.vps.service")

SERVICE_NAME = "supervisor"


@dataclass(frozen=True, slots=True)
class VpsServiceConfig:
    state_root: str = "/var/lib/iron-spyder"
    mode: str = Mode.PAPER.value
    decision_interval: timedelta = timedelta(minutes=5)
    scenario: str = "broad_range"
    use_tradier: bool = True
    """Consult ``TRADIER_ACCESS_TOKEN``. Off forces synthetic without a token check."""
    tradier_expirations: int = 3
    max_cycles: int | None = None
    """Stop after N cycles. Used by tests and dry runs; the unit leaves this unset."""


@dataclass
class VpsService:
    """Run the decision supervisor and publish VPS state files."""

    supervisor: Supervisor
    config: VpsServiceConfig
    mode: str
    _last_result: PipelineResult | None = field(default=None, init=False)
    _original_cycle: Callable[..., PipelineResult | None] | None = field(
        default=None, init=False, repr=False
    )
    _original_heartbeat: Callable[..., None] | None = field(default=None, init=False, repr=False)
    _original_closed: Callable[..., None] | None = field(default=None, init=False, repr=False)

    def run(self) -> SupervisorStats:
        ensure_state_tree(self.config.state_root)
        self._install_hooks()
        try:
            # Publish once at start so the dashboard never sees a silent fresh unit.
            self._publish(
                None,
                self.supervisor.clock(),
                note="supervisor starting",
                status=STATUS_STARTING,
            )
            return self.supervisor.run()
        finally:
            self._remove_hooks()
            # The exit write must say stopped. This is the last thing the
            # dashboard will read until the unit comes back, so reporting
            # "running" here is how a dead daemon goes unnoticed.
            self._publish(
                self._last_result,
                self.supervisor.clock(),
                note="supervisor stopped",
                status=STATUS_STOPPED,
            )

    def _install_hooks(self) -> None:
        self._original_cycle = self.supervisor.run_cycle
        self._original_heartbeat = self.supervisor._heartbeat
        self._original_closed = self.supervisor._on_market_closed

        def instrumented_cycle(now: datetime | None = None) -> PipelineResult | None:
            assert self._original_cycle is not None
            result = self._original_cycle(now)
            if result is not None:
                self._last_result = result
            return result

        def instrumented_heartbeat(now: datetime) -> None:
            assert self._original_heartbeat is not None
            self._original_heartbeat(now)
            self._publish(self._last_result, now)

        def instrumented_closed(now: datetime) -> None:
            assert self._original_closed is not None
            # Refresh liveness before sleeping so a weekend wait is not "stale".
            self._publish(self._last_result, now, note="market closed; waiting for next open")
            self._original_closed(now)

        self.supervisor.run_cycle = instrumented_cycle  # type: ignore[method-assign]
        self.supervisor._heartbeat = instrumented_heartbeat  # type: ignore[method-assign]
        self.supervisor._on_market_closed = instrumented_closed  # type: ignore[method-assign]

    def _remove_hooks(self) -> None:
        if self._original_cycle is not None:
            self.supervisor.run_cycle = self._original_cycle  # type: ignore[method-assign]
        if self._original_heartbeat is not None:
            self.supervisor._heartbeat = self._original_heartbeat  # type: ignore[method-assign]
        if self._original_closed is not None:
            self.supervisor._on_market_closed = self._original_closed  # type: ignore[method-assign]

    def _refresh_interval(self) -> float:
        """How often this service republishes, in seconds.

        Floored at 60 so a fast decision interval does not make every reader
        flag the file as late during the gap between two cycles. The
        market-closed path republishes on each idle poll, which is more often
        than this, so the floor bounds the slowest case.
        """
        return max(self.config.decision_interval.total_seconds(), 60.0)

    def _publish(
        self,
        result: PipelineResult | None,
        now: datetime,
        *,
        note: str = "",
        status: str = STATUS_RUNNING,
    ) -> None:
        if result is not None:
            self._last_result = result
        stats = self.supervisor.stats.as_dict()
        kill = list(self.supervisor.pipeline.risk_engine.kill_switch.reasons)
        payload = build_live_state(
            mode=self.mode,
            stats=stats,
            result=self._last_result,
            kill_switches=kill,
            open_positions=self.supervisor.pipeline.portfolio.open_count,
            equity=self.supervisor.pipeline.broker.equity,
            note=note,
            status=status,
            refresh_interval_seconds=self._refresh_interval(),
            now=now,
        )
        try:
            write_live_state(payload, state_root=self.config.state_root)
        except OSError:
            log.exception("failed to write live_state")

        detail = (
            f"cycles={stats['cycles']} trades={stats['trades']} open={payload['open_positions']}"
        )
        write_heartbeat(
            self.config.state_root,
            SERVICE_NAME,
            interval_seconds=self._refresh_interval(),
            detail=detail,
            extra={"mode": self.mode, "kill_switches": kill},
            now=now,
        )


def build_pipeline(
    *,
    mode: str,
    state_root: str | Path,
    audit_path: str | Path | None = None,
) -> DecisionPipeline:
    settings = load_settings(mode)
    settings.assert_live_allowed()
    paths = state_paths(state_root)
    return DecisionPipeline(
        forecast_engine=ForecastEngine(config=settings.forecast),
        optimizer=StrategyOptimizer(settings.optimizer),
        risk_engine=RiskEngine(config=settings.risk),
        broker=PaperBroker(
            config=PaperBrokerConfig(starting_equity=settings.starting_equity),
            costs=settings.costs,
        ),
        config=PipelineConfig(data_quality=settings.data_quality, costs=settings.costs),
        audit=AuditStore(str(audit_path or paths.audit_db)),
    )


def select_provider(config: VpsServiceConfig) -> MarketDataProvider:
    """Choose the market-data source, and never lie about which one won.

    The previous behaviour was a bare ``SyntheticProvider(...)`` default, while
    ``.env.example`` advertised ``TRADIER_ACCESS_TOKEN`` under "supervisor falls
    back to synthetic". Nothing read the token, so an operator who set a real
    one got a healthy-looking daemon trading a seeded random walk, with no
    signal anywhere that its inputs were invented.

    So the rule here is asymmetric on purpose:

    * **No token** — synthetic, logged as such at WARNING. Legitimate for demos
      and dry runs, but never quiet.
    * **Token set and working** — live.
    * **Token set and broken** — raise. Falling back would reproduce exactly the
      failure this function exists to remove, and it is the worst case of the
      three: someone deliberately configured a live feed, so a silent downgrade
      is a downgrade they have no reason to look for.
    """
    if not config.use_tradier:
        return _synthetic(config, reason="disabled by configuration")

    tradier = TradierProvider.from_env(expirations=config.tradier_expirations)
    if tradier is None:
        return _synthetic(config, reason="TRADIER_ACCESS_TOKEN is not set")

    try:
        log.info("market data: %s", tradier.check_connectivity())
    except TradierError as exc:
        raise RuntimeError(
            f"TRADIER_ACCESS_TOKEN is set but the feed is unusable: {exc}\n"
            "Refusing to start on synthetic data while a live feed is configured — "
            "every decision would be based on invented prices. Fix the token, or "
            "unset it to run synthetic deliberately."
        ) from exc
    return tradier


def _synthetic(config: VpsServiceConfig, *, reason: str) -> MarketDataProvider:
    log.warning(
        "MARKET DATA IS SYNTHETIC (%s). Scenario %r is a seeded generator: every "
        "price, greek, and forecast below is invented and none of it describes "
        "the real market.",
        reason,
        config.scenario,
    )
    return SyntheticProvider(spec=scenario(config.scenario), step=config.decision_interval)


def build_service(
    config: VpsServiceConfig,
    *,
    provider: MarketDataProvider | None = None,
    alert: Callable[[str, str], None] | None = log_alert,
    clock: Callable[[], datetime] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> VpsService:
    paths = ensure_state_tree(config.state_root)
    pipeline = build_pipeline(mode=config.mode, state_root=config.state_root)
    market = provider or select_provider(config)
    supervisor = Supervisor(
        pipeline=pipeline,
        provider=market,
        state_store=RuntimeStateStore(paths.runtime_db),
        config=SupervisorConfig(
            decision_interval=config.decision_interval,
            max_cycles=config.max_cycles,
            # Publish VPS heartbeats every cycle rather than every 15 minutes —
            # the file surface is what the dashboard reads, not the log line.
            heartbeat_interval=timedelta(seconds=0),
        ),
        alert=alert,
    )
    if clock is not None:
        supervisor.clock = clock
    if sleeper is not None:
        supervisor.sleeper = sleeper
    return VpsService(supervisor=supervisor, config=config, mode=config.mode)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iron-Spyder VPS supervisor (decision loop + live_state publisher)"
    )
    parser.add_argument(
        "--state-root",
        default=None,
        help="VPS state root (default: $IRON_SPYDER_STATE_ROOT or /var/lib/iron-spyder)",
    )
    parser.add_argument(
        "--mode",
        default=Mode.PAPER.value,
        choices=[m.value for m in Mode],
        help="deployment mode (live is refused until phase-2 gates are met)",
    )
    parser.add_argument("--interval", type=int, default=5, help="minutes between decisions")
    parser.add_argument("--scenario", default="broad_range", choices=sorted(SCENARIOS))
    parser.add_argument("--max-cycles", type=int, help="stop after N cycles (dry run)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="force the synthetic generator even if TRADIER_ACCESS_TOKEN is set",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="report which market-data source would be used, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    root = args.state_root or state_paths().root
    config = VpsServiceConfig(
        state_root=str(root),
        mode=args.mode,
        decision_interval=timedelta(minutes=args.interval),
        scenario=args.scenario,
        use_tradier=not args.synthetic,
        max_cycles=args.max_cycles,
    )
    if args.check_data:
        return _report_data_source(config)
    service = build_service(config)
    service.supervisor.install_signal_handlers()
    stats = service.run()
    log.info("final: %s", stats.as_dict())
    return 0


def _report_data_source(config: VpsServiceConfig) -> int:
    """Answer "what is this thing actually reading?" without starting a daemon.

    Worth its own flag because the answer is otherwise only visible in the
    startup log, and it is the single most consequential fact about a running
    instance.
    """
    try:
        chosen = select_provider(config)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    if isinstance(chosen, TradierProvider):
        log.info("LIVE — %s", chosen.check_connectivity())
        return 0
    log.warning("SYNTHETIC — scenario %r; no live market data.", config.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
