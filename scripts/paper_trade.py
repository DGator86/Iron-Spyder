"""Replay a synthetic session through the full paper-trading pipeline.

    python -m scripts.paper_trade --scenario vol_expansion
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from spy_der.config.settings import Mode, load_settings
from spy_der.data.persistence.audit import AuditStore
from spy_der.data.providers.base import SyntheticProvider
from spy_der.data.synthetic import SCENARIOS, scenario
from spy_der.execution.paper_broker import PaperBroker, PaperBrokerConfig
from spy_der.models.forecast_engine import ForecastEngine
from spy_der.optimizer.engine import StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig
from spy_der.risk.engine import RiskEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="vol_expansion", choices=sorted(SCENARIOS))
    parser.add_argument("--steps", type=int, default=26)
    parser.add_argument("--interval", type=int, default=15, help="minutes between decisions")
    parser.add_argument("--audit", default="spyder_paper.db")
    args = parser.parse_args()

    settings = load_settings(Mode.PAPER)
    store = AuditStore(args.audit)
    pipeline = DecisionPipeline(
        forecast_engine=ForecastEngine(config=settings.forecast),
        optimizer=StrategyOptimizer(settings.optimizer),
        risk_engine=RiskEngine(config=settings.risk),
        broker=PaperBroker(
            config=PaperBrokerConfig(starting_equity=settings.starting_equity),
            costs=settings.costs,
        ),
        config=PipelineConfig(data_quality=settings.data_quality, costs=settings.costs),
        audit=store,
    )

    provider = SyntheticProvider(
        spec=scenario(args.scenario), step=timedelta(minutes=args.interval)
    )
    stamps = provider.session_timestamps(datetime(2026, 8, 3, tzinfo=UTC))[: args.steps]

    for timestamp in stamps:
        result = pipeline.step(provider.snapshot(timestamp))
        marker = "TRADE" if result.traded else "     "
        detail = result.reasons[0] if result.reasons else ""
        state = result.forecast.dominant_state.value if result.forecast else "-"
        print(f"{timestamp:%H:%M} {marker} {state:<16} {result.decision:<34} {detail[:60]}")
        for signal in result.exits:
            print(f"        EXIT  {signal.reason.value}: {signal.detail}")

    print(f"\nequity {pipeline.broker.equity:,.2f}   audit {store.counts()}")


if __name__ == "__main__":
    main()
