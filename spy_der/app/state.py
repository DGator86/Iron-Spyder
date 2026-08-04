"""Application state shared by the API and dashboard.

Holds one :class:`~spy_der.pipeline.DecisionPipeline` and the most recent
result. The pipeline carries temporal state — the HMM belief, the flow and wall
trackers, the open book — so it must be a single long-lived instance rather
than rebuilt per request.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from spy_der.config.settings import Mode, Settings, load_settings
from spy_der.data.persistence.audit import AuditStore
from spy_der.data.providers.base import MarketDataProvider, SyntheticProvider
from spy_der.data.providers.tradier import TradierError, TradierProvider
from spy_der.data.synthetic import scenario
from spy_der.execution.paper_broker import PaperBroker, PaperBrokerConfig
from spy_der.models.forecast_engine import ForecastEngine
from spy_der.optimizer.engine import StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig, PipelineResult
from spy_der.risk.engine import RiskEngine
from spy_der.risk.limits import KillSwitch

log = logging.getLogger(__name__)


def resolve_market_provider(
    scenario_name: str = "broad_range",
) -> tuple[MarketDataProvider, str]:
    """Pick Tradier when configured, otherwise the named synthetic scenario.

    Mirrors :func:`spy_der.vps.service.select_provider` so the public FastAPI
    surface and the supervisor cannot disagree about whether prices are real.
    A broken token fails closed to synthetic with a loud warning — the dashboard
    must stay up after hours, but it must never look "live" while inventing.
    """
    force_synthetic = os.environ.get("IRON_SPYDER_FORCE_SYNTHETIC", "").strip() in {
        "1",
        "true",
        "yes",
    }
    if not force_synthetic:
        tradier = TradierProvider.from_env()
        if tradier is not None:
            try:
                log.info("API market data: %s", tradier.check_connectivity())
                return tradier, "tradier"
            except TradierError as exc:
                log.error(
                    "TRADIER_ACCESS_TOKEN is set but unusable (%s); "
                    "API falling back to synthetic scenario %r",
                    exc,
                    scenario_name,
                )
    log.warning(
        "API MARKET DATA IS SYNTHETIC (scenario %r). Set TRADIER_ACCESS_TOKEN "
        "for live SPY chain reads.",
        scenario_name,
    )
    return (
        SyntheticProvider(spec=scenario(scenario_name), step=timedelta(minutes=15)),
        "synthetic",
    )


@dataclass
class AppState:
    """Long-lived application state."""

    settings: Settings = field(default_factory=lambda: load_settings(Mode.PAPER))
    provider: MarketDataProvider | None = None
    pipeline: DecisionPipeline | None = None
    audit: AuditStore | None = None
    last_result: PipelineResult | None = None
    scenario_name: str = "broad_range"
    #: ``tradier`` or ``synthetic`` — surfaced on /health so operators can tell.
    feed: str = "synthetic"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider, self.feed = resolve_market_provider(self.scenario_name)
        if self.audit is None:
            self.audit = AuditStore(self.settings.audit_path)
        if self.pipeline is None:
            self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> DecisionPipeline:
        settings = self.settings
        return DecisionPipeline(
            forecast_engine=ForecastEngine(config=settings.forecast),
            optimizer=StrategyOptimizer(settings.optimizer),
            risk_engine=RiskEngine(config=settings.risk),
            broker=PaperBroker(
                config=PaperBrokerConfig(starting_equity=settings.starting_equity),
                costs=settings.costs,
            ),
            config=PipelineConfig(data_quality=settings.data_quality, costs=settings.costs),
            audit=self.audit,
        )

    def run_once(self, at: datetime | None = None) -> PipelineResult:
        """Run one decision cycle. Serialized: the pipeline is not reentrant."""
        assert self.provider is not None and self.pipeline is not None
        with self._lock:
            timestamp = at or datetime.now(UTC)
            snapshot = self.provider.snapshot(timestamp)
            self.last_result = self.pipeline.step(snapshot)
            return self.last_result

    def current(self) -> PipelineResult:
        """Return a fresh-enough result for desk endpoints.

        The first call seeds ``last_result``. Later calls reuse it only while it
        is younger than :meth:`_max_result_age` — otherwise the public API and
        Vercel chart freeze on a single morning snapshot while the supervisor
        keeps trading against live Tradier.
        """
        result = self.last_result
        if result is not None and not self._is_stale(result):
            return result
        return self.run_once()

    def _max_result_age(self) -> timedelta:
        raw = os.environ.get("IRON_SPYDER_API_MAX_AGE_SECONDS", "").strip()
        if raw:
            try:
                return timedelta(seconds=max(15, int(raw)))
            except ValueError:
                pass
        # Cap at 60s so a 5s UI poll cannot serve a multi-hour-old chain.
        minutes = float(os.environ.get("IRON_SPYDER_INTERVAL_MINUTES") or 5)
        return timedelta(seconds=min(60.0, max(30.0, minutes * 60.0)))

    def _is_stale(self, result: PipelineResult) -> bool:
        ts = result.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return datetime.now(UTC) - ts.astimezone(UTC) >= self._max_result_age()

    def set_scenario(self, name: str) -> None:
        """Point the provider at a synthetic scenario (research aid).

        Explicitly switches off Tradier for this process — scenario control is
        only meaningful for the seeded generator.
        """
        with self._lock:
            self.scenario_name = name
            self.provider = SyntheticProvider(
                spec=scenario(name), step=timedelta(minutes=15)
            )
            self.feed = "synthetic"
            self.last_result = None

    def reset(self) -> None:
        with self._lock:
            self.pipeline = self._build_pipeline()
            self.last_result = None
            # Re-resolve feed so a reset after env changes picks up Tradier.
            self.provider, self.feed = resolve_market_provider(self.scenario_name)

    @property
    def kill_switch(self) -> KillSwitch:
        assert self.pipeline is not None
        return self.pipeline.risk_engine.kill_switch


STATE = AppState()
