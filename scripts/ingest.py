"""Inspect and ingest historical SPY option data.

Run ``inspect`` first against unfamiliar data — it reports how the columns map
onto the canonical schema without loading the whole dataset:

    python -m scripts.ingest inspect --path /data/spy_chains
    python -m scripts.ingest load    --path /data/spy_chains --bars /data/spy_1m.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spy_der.data.providers.historical import (
    HistoricalProvider,
    IngestConfig,
    discover_files,
    inspect,
    load_bars,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("inspect", "load"):
        sp = sub.add_parser(name)
        sp.add_argument("--path", required=True, help="file or directory of chain data")
        sp.add_argument("--pattern", default="*", help="glob applied under a directory")
        if name == "load":
            sp.add_argument("--bars", help="optional SPY price-bar file")
            sp.add_argument("--limit", type=int, help="stop after N snapshots")
            sp.add_argument(
                "--local-time",
                action="store_true",
                help="treat naive timestamps as local rather than UTC",
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


if __name__ == "__main__":
    main()
