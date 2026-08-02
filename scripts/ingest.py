"""Inspect, ingest, and replay historical SPY option data.

Run ``inspect`` first against unfamiliar data — it reports how the columns map
onto the canonical schema without loading the whole dataset:

    python -m scripts.ingest inspect  --path /data/spy_chains
    python -m scripts.ingest load     --path /data/spy_chains --bars /data/spy_1m.csv
    python -m scripts.ingest backtest --path /data/spy_chains --bars /data/spy_1m.csv

``backtest`` streams the ingested snapshots straight through the same
``DecisionPipeline`` the live path uses, so nothing about the decision logic
differs between replay and production.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from spy_der.backtest.engine import run_backtest
from spy_der.config.settings import load_settings
from spy_der.data.providers.historical import (
    HistoricalProvider,
    IngestConfig,
    IngestStats,
    discover_files,
    inspect,
    iter_snapshots,
    load_bars,
)
from spy_der.domain.market import MarketSnapshot
from spy_der.execution.paper_broker import PaperBroker, PaperBrokerConfig
from spy_der.models.forecast_engine import ForecastEngine
from spy_der.optimizer.engine import StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig
from spy_der.risk.engine import RiskEngine


def build_pipeline(mode: str) -> DecisionPipeline:
    """The production pipeline, with persistence off.

    Deliberately the same construction the daemon uses. A replay that assembled
    a different pipeline would be measuring something other than the system.
    """
    settings = load_settings(mode)
    return DecisionPipeline(
        forecast_engine=ForecastEngine(config=settings.forecast),
        optimizer=StrategyOptimizer(settings.optimizer),
        risk_engine=RiskEngine(config=settings.risk),
        broker=PaperBroker(
            config=PaperBrokerConfig(starting_equity=settings.starting_equity),
            costs=settings.costs,
        ),
        config=PipelineConfig(
            data_quality=settings.data_quality, costs=settings.costs, persist=False
        ),
    )


def counted(
    snapshots: Iterator[MarketSnapshot], every: int, stats: IngestStats
) -> Iterator[MarketSnapshot]:
    """Pass snapshots through, reporting progress.

    A year of chains takes a while; a silent process gives no way to tell
    progress from a hang.
    """
    for index, snapshot in enumerate(snapshots, start=1):
        if index % every == 0:
            print(f"  ... {index} snapshots ({snapshot.timestamp:%Y-%m-%d %H:%M})", flush=True)
        yield snapshot
    print(f"  ingest: {stats.summary()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("inspect", "load", "backtest"):
        sp = sub.add_parser(name)
        sp.add_argument("--path", required=True, help="file or directory of chain data")
        sp.add_argument("--pattern", default="*", help="glob applied under a directory")
        if name != "inspect":
            sp.add_argument("--bars", help="optional SPY price-bar file")
            sp.add_argument("--limit", type=int, help="stop after N snapshots")
            sp.add_argument(
                "--local-time",
                action="store_true",
                help="treat naive timestamps as local rather than UTC",
            )
        if name == "backtest":
            sp.add_argument("--mode", default="backtest", help="settings profile")
            sp.add_argument(
                "--progress", type=int, default=200, help="report every N snapshots"
            )

    args = parser.parse_args()
    root = Path(args.path)
    files = discover_files(root, pattern=args.pattern)
    if not files:
        raise SystemExit(f"no chain files found under {root}")

    if args.command == "inspect":
        report = inspect(files)
        print(json.dumps(report, indent=2, default=str))
        if report.get("missing_required"):
            raise SystemExit(
                "\nMissing required fields. Add the source column names to "
                "COLUMN_ALIASES in spy_der/data/providers/historical.py."
            )
        print("\nSchema OK — every required field mapped.")
        return

    bars = load_bars(Path(args.bars)) if args.bars else []
    config = IngestConfig(assume_utc=not args.local_time)

    if args.command == "backtest":
        run_replay(files, bars=bars, config=config, args=args)
        return

    provider, stats = HistoricalProvider.from_files(
        files, bars=bars, config=config, limit=args.limit
    )

    print(f"files:  {len(files)}")
    print(f"bars:   {len(bars)}")
    print(f"ingest: {stats.summary()}")
    for note in stats.errors[:10]:
        print(f"  note: {note}")

    span = provider.span()
    if span is None:
        raise SystemExit("no usable snapshots were produced")
    print(f"span:   {span[0]} .. {span[1]}")
    print(f"days:   {len(provider.trading_days)}")

    quality = [s.data_quality_score for s in provider.snapshots]
    print(f"DQ:     min={min(quality):.3f} mean={sum(quality) / len(quality):.3f}")
    tradeable = sum(1 for q in quality if q >= config.quality.threshold)
    print(f"        {tradeable}/{len(quality)} snapshots clear the DQ threshold")


def run_replay(files, *, bars, config: IngestConfig, args) -> None:
    """Stream the history through the pipeline and report the outcome.

    Streaming rather than materializing matters at real sizes: a year of
    one-minute chains does not fit in memory, and ``run_backtest`` consumes an
    iterable one snapshot at a time.
    """
    stats = IngestStats()
    pipeline = build_pipeline(args.mode)

    stream = iter_snapshots(files, bars=bars, config=config, stats=stats)
    if args.limit is not None:
        stream = islice(stream, args.limit)

    print(f"files:  {len(files)}   bars: {len(bars)}")
    print("replaying...", flush=True)
    result = run_backtest(pipeline, counted(stream, args.progress, stats))

    if not result.decisions:
        raise SystemExit("no usable snapshots were produced; run `inspect` on this data")

    metrics = result.metrics()
    print()
    for key, value in metrics.as_dict().items():
        print(f"  {key:<22} {value}")
    print(f"  {'decisions':<22} {result.decision_count}")

    # No-trade dominating is the expected outcome, not a failure: the system is
    # required to beat an explicit no-trade competitor by a margin before acting.
    if metrics.trade_count == 0:
        print(
            "\nNo trades were taken. That is a valid outcome — every candidate "
            "lost to the no-trade competitor. Check the DQ line from `load` "
            "first if you expected otherwise."
        )


if __name__ == "__main__":
    main()
