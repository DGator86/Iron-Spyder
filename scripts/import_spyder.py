"""Import and replay SPY-DER VPS market recordings.

The dedicated Iron-Spyder CPU VPS stages legacy tapes under::

    /var/lib/iron-spyder/imports/spy-der/market/*.jsonl

Inspect first, then stream a backtest through the same DecisionPipeline the
live path uses:

    python -m scripts.import_spyder inspect \\
        --path /var/lib/iron-spyder/imports/spy-der/market

    python -m scripts.import_spyder backtest \\
        --path /var/lib/iron-spyder/imports/spy-der/market \\
        --session 2026-07-31 --limit 50
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from spy_der.backtest.engine import run_backtest
from spy_der.config.settings import load_settings
from spy_der.data.providers.spyder_recordings import (
    ImportStats,
    discover_session_files,
    inspect_recordings,
    iter_snapshots,
)
from spy_der.domain.market import MarketSnapshot
from spy_der.execution.paper_broker import PaperBroker, PaperBrokerConfig
from spy_der.models.forecast_engine import ForecastEngine
from spy_der.optimizer.engine import StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig
from spy_der.risk.engine import RiskEngine


def build_pipeline(mode: str) -> DecisionPipeline:
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
    snapshots: Iterator[MarketSnapshot], every: int, stats: ImportStats
) -> Iterator[MarketSnapshot]:
    for index, snapshot in enumerate(snapshots, start=1):
        if every and index % every == 0:
            print(
                f"  ... {index} snapshots ({snapshot.timestamp:%Y-%m-%d %H:%M}) "
                f"dq={snapshot.data_quality_score:.2f} chain={len(snapshot.option_chain)}",
                flush=True,
            )
        yield snapshot
    print(f"  import: {stats.summary()}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("inspect", "load", "backtest"):
        sp = sub.add_parser(name)
        sp.add_argument(
            "--path",
            default="/var/lib/iron-spyder/imports/spy-der/market",
            help="session tape file or directory",
        )
        sp.add_argument("--pattern", default="*.jsonl")
        sp.add_argument(
            "--include-closed",
            action="store_true",
            help="include PRE_OPEN/CLOSED records (default: OPEN only)",
        )
        if name != "inspect":
            sp.add_argument("--session", help="restrict to YYYY-MM-DD filename stem")
            sp.add_argument("--limit", type=int, help="stop after N snapshots")
            sp.add_argument("--progress", type=int, default=50)
        if name == "backtest":
            sp.add_argument("--mode", default="backtest")

    args = parser.parse_args(argv)
    root = Path(args.path)
    files = discover_session_files(root, pattern=args.pattern)
    if not files:
        raise SystemExit(f"no session tapes under {root}")

    if getattr(args, "session", None):
        files = [p for p in files if p.stem == args.session]
        if not files:
            raise SystemExit(f"no tape for session {args.session!r} under {root}")

    if args.command == "inspect":
        report = inspect_recordings(files, sample=0)
        print(json.dumps(report, indent=2, default=str))
        return 0

    stats = ImportStats()
    open_only = not args.include_closed
    stream = iter_snapshots(files, stats=stats, open_only=open_only)
    if args.limit:
        stream = islice(stream, args.limit)

    if args.command == "load":
        count = 0
        for snapshot in counted(stream, args.progress, stats):
            count += 1
            _ = snapshot
        print(f"loaded {count} snapshots")
        return 0 if count else 1

    # backtest
    pipeline = build_pipeline(args.mode)
    result = run_backtest(pipeline, counted(stream, args.progress, stats))
    print(
        json.dumps(
            {
                "decisions": result.decision_count,
                "trades": result.trade_count,
                "import": stats.summary(),
                "metrics": result.metrics().as_dict(),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
