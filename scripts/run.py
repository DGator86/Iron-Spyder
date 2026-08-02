"""Run SPY-DER unattended.

    python -m scripts.run --mode paper --interval 5

Restores prior state, reconciles against the broker, then decides on a schedule
until stopped. SIGTERM and SIGINT stop it cleanly at a cycle boundary.
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta

from spy_der.config.settings import Mode, load_settings
from spy_der.data.persistence.audit import AuditStore
from spy_der.data.providers.base import SyntheticProvider
from spy_der.data.synthetic import SCENARIOS, scenario
from spy_der.execution.paper_broker import PaperBroker, PaperBrokerConfig
from spy_der.models.forecast_engine import ForecastEngine
from spy_der.optimizer.engine import StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig
from spy_der.risk.engine import RiskEngine
from spy_der.runtime.state_store import RuntimeStateStore
from spy_der.runtime.supervisor import Supervisor, SupervisorConfig, log_alert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", default="paper", choices=[m.value for m in Mode], help="deployment mode"
    )
    parser.add_argument("--interval", type=int, default=5, help="minutes between decisions")
    parser.add_argument("--state", default="spyder_state.db", help="runtime state database")
    parser.add_argument("--scenario", default="broad_range", choices=sorted(SCENARIOS))
    parser.add_argument("--max-cycles", type=int, help="stop after N cycles (dry run)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    settings = load_settings(args.mode)
    settings.assert_live_allowed()  # refuses live until the spec 40 gates are met

    pipeline = DecisionPipeline(
        forecast_engine=ForecastEngine(config=settings.forecast),
        optimizer=StrategyOptimizer(settings.optimizer),
        risk_engine=RiskEngine(config=settings.risk),
        broker=PaperBroker(
            config=PaperBrokerConfig(starting_equity=settings.starting_equity),
            costs=settings.costs,
        ),
        config=PipelineConfig(data_quality=settings.data_quality, costs=settings.costs),
        audit=AuditStore(settings.audit_path),
    )

    # Swap this for a live market-data provider once one is wired up; the
    # supervisor only needs the MarketDataProvider interface.
    provider = SyntheticProvider(
        spec=scenario(args.scenario), step=timedelta(minutes=args.interval)
    )

    supervisor = Supervisor(
        pipeline=pipeline,
        provider=provider,
        state_store=RuntimeStateStore(args.state),
        config=SupervisorConfig(
            decision_interval=timedelta(minutes=args.interval),
            max_cycles=args.max_cycles,
        ),
        alert=log_alert,
    )
    supervisor.install_signal_handlers()

    stats = supervisor.run()
    logging.getLogger("spy_der").info("final: %s", stats.as_dict())


if __name__ == "__main__":
    main()
