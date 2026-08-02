"""Application state shared by the API and dashboard.

Holds one :class:`~spy_der.pipeline.DecisionPipeline` and the most recent
result. The pipeline carries temporal state — the HMM belief, the flow and wall
trackers, the open book — so it must be a single long-lived instance rather
than rebuilt per request.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from spy_der.config.settings import Mode, Settings, load_settings
from spy_der.data.persistence.audit import AuditStore
from spy_der.data.providers.base import MarketDataProvider, SyntheticProvider
from spy_der.data.synthetic import scenario
from spy_der.execution.paper_broker import PaperBroker, PaperBrokerConfig
from spy_der.models.forecast_engine import ForecastEngine
from spy_der.optimizer.engine import StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig, PipelineResult
from spy_der.risk.engine import RiskEngine
from spy_der.risk.limits import KillSwitch


@dataclass
class AppState:
    """Long-lived application state."""

    settings: Settings = field(default_factory=lambda: load_settings(Mode.PAPER))
    provider: MarketDataProvider | None = None
    pipeline: DecisionPipeline | None = None
    audit: AuditStore | None = None
    last_result: PipelineResult | None = None
    scenario_name: str = "broad_range"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = SyntheticProvider(
                spec=scenario(self.scenario_name), step=timedelta(minutes=15)
            )
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
        """Return the last result, running a cycle if none exists yet."""
        return self.last_result or self.run_once()

    def set_scenario(self, name: str) -> None:
        """Point the synthetic provider at a different scenario."""
        with self._lock:
            self.scenario_name = name
            self.provider = SyntheticProvider(
                spec=scenario(name), step=timedelta(minutes=15)
            )
            self.last_result = None

    def reset(self) -> None:
        with self._lock:
            self.pipeline = self._build_pipeline()
            self.last_result = None

    @property
    def kill_switch(self) -> KillSwitch:
        assert self.pipeline is not None
        return self.pipeline.risk_engine.kill_switch


STATE = AppState()
