"""Import and replay SPY-DER VPS market recordings.

The dedicated Iron-Spyder CPU VPS stages legacy tapes under::

    /var/lib/iron-spyder/imports/spy-der/market/*.jsonl

Inspect first, then stream a backtest through the same DecisionPipeline the
live path uses:

    python -m scripts.import_spyder inspect \\
        --path /var/lib/iron-spyder/imports/spy-der/market

    python -m scripts.import_spyder backtest \\
        --path /var/lib/iron-spyder/imports/spy-der/market \\
        --from-date 2026-07-14 --output /var/lib/iron-spyder/reports/imports/history.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import date, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from spy_der.backtest.engine import BacktestResult, run_backtest
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


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _filter_files(
    files: list[Path],
    *,
    session: str | None,
    from_date: date | None,
    to_date: date | None,
    require_chain: bool,
) -> list[Path]:
    selected = files
    if session:
        selected = [p for p in selected if p.stem == session]
        if not selected:
            raise SystemExit(f"no tape for session {session!r}")
    if from_date is not None:
        selected = [p for p in selected if date.fromisoformat(p.stem) >= from_date]
    if to_date is not None:
        selected = [p for p in selected if date.fromisoformat(p.stem) <= to_date]
    if require_chain:
        report = inspect_recordings(selected, sample=0)
        keep = {
            row["file"]
            for row in report["sessions"]
            if int(row.get("max_open_chain_contracts") or 0) > 0
        }
        selected = [p for p in selected if p.name in keep]
    if not selected:
        raise SystemExit("no session tapes matched the filters")
    return selected


def _trade_row(position: Any) -> dict[str, Any]:
    family = position.family.value if hasattr(position.family, "value") else str(position.family)
    return {
        "position_id": position.position_id,
        "family": family,
        "contracts": position.contracts,
        "opened_at": position.opened_at.isoformat(),
        "closed_at": position.closed_at.isoformat() if position.closed_at else None,
        "entry_price": position.entry_price,
        "realized_pnl": position.realized_pnl,
        "exit_reason": position.exit_reason.value if position.exit_reason else None,
        "max_loss_per_contract": position.max_loss_per_contract,
        "legs": [
            {
                "right": leg.right.value,
                "strike": leg.strike,
                "expiry": leg.expiry.isoformat(),
                "quantity": leg.quantity,
            }
            for leg in position.legs
        ],
    }


def _daily_breakdown(result: BacktestResult) -> list[dict[str, Any]]:
    by_day: dict[str, dict[str, Any]] = {}
    for decision in result.decisions:
        day = decision.timestamp.date().isoformat()
        row = by_day.setdefault(
            day,
            {
                "date": day,
                "decisions": 0,
                "trades_opened": 0,
                "no_trade": 0,
                "dq_sum": 0.0,
                "states": Counter(),
            },
        )
        row["decisions"] += 1
        row["dq_sum"] += decision.quality.score
        if decision.is_no_trade:
            row["no_trade"] += 1
        if decision.fill is not None:
            row["trades_opened"] += 1
        if decision.forecast is not None:
            row["states"][decision.forecast.dominant_state.value] += 1

    # Not a Counter: that is dict[str, int], and realized P&L is a float, so
    # accumulating into one silently claims whole-dollar amounts.
    pnl_by_day: defaultdict[str, float] = defaultdict(float)
    for position in result.closed_positions:
        if position.closed_at is None:
            continue
        pnl_by_day[position.closed_at.date().isoformat()] += position.realized_pnl

    out: list[dict[str, Any]] = []
    for day in sorted(by_day):
        row = by_day[day]
        decisions = int(row["decisions"])
        states: Counter[str] = row.pop("states")
        dq_sum = float(row.pop("dq_sum"))
        out.append(
            {
                "date": day,
                "decisions": decisions,
                "trades_opened": row["trades_opened"],
                "no_trade": row["no_trade"],
                "no_trade_rate": row["no_trade"] / decisions if decisions else 1.0,
                "avg_data_quality": dq_sum / decisions if decisions else 0.0,
                "realized_pnl": float(pnl_by_day.get(day, 0.0)),
                "top_state": states.most_common(1)[0][0] if states else None,
            }
        )
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def build_report(
    *,
    files: list[Path],
    result: BacktestResult,
    stats: ImportStats,
    mode: str,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    equity = result.equity_curve
    stride = max(1, len(equity) // 200) if equity else 1
    return _jsonable(
        {
            "generated_at": finished_at.isoformat(),
            "mode": mode,
            "sessions": [p.stem for p in files],
            "session_count": len(files),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
            "import": stats.summary(),
            "import_detail": {
                "files": stats.files,
                "records": stats.records,
                "snapshots": stats.snapshots,
                "contracts_kept": stats.contracts_kept,
                "contracts_dropped": stats.contracts_dropped,
                "iv_derived": stats.iv_derived,
                "iv_unavailable": stats.iv_unavailable,
                "skipped_status": stats.skipped_status,
                "parse_errors": stats.parse_errors,
                "notes": list(stats.notes),
            },
            "summary": {
                "decisions": result.decision_count,
                "trades": result.trade_count,
                "starting_equity": equity[0][1] if equity else None,
                "ending_equity": equity[-1][1] if equity else None,
                "metrics": result.metrics().as_dict(),
            },
            "daily": _daily_breakdown(result),
            "trades": [_trade_row(p) for p in result.closed_positions],
            "equity_curve": [
                {"timestamp": ts.isoformat(), "equity": eq} for ts, eq in equity[::stride]
            ],
        }
    )


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
        sp.add_argument("--session", help="restrict to YYYY-MM-DD filename stem")
        sp.add_argument("--from-date", type=_parse_date, help="inclusive YYYY-MM-DD")
        sp.add_argument("--to-date", type=_parse_date, help="inclusive YYYY-MM-DD")
        if name != "inspect":
            sp.add_argument("--limit", type=int, help="stop after N snapshots")
            sp.add_argument("--progress", type=int, default=50)
            sp.add_argument(
                "--require-chain",
                action="store_true",
                help="skip session tapes whose OPEN chain never carried contracts",
            )
        if name == "backtest":
            sp.add_argument("--mode", default="backtest")
            sp.add_argument(
                "--output",
                type=Path,
                help="write full JSON report to this path (also prints a short summary)",
            )

    args = parser.parse_args(argv)
    root = Path(args.path)
    files = discover_session_files(root, pattern=args.pattern)
    if not files:
        raise SystemExit(f"no session tapes under {root}")

    files = _filter_files(
        files,
        session=getattr(args, "session", None),
        from_date=getattr(args, "from_date", None),
        to_date=getattr(args, "to_date", None),
        require_chain=bool(getattr(args, "require_chain", False)),
    )

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

    print(
        f"backtest sessions={len(files)} "
        f"({files[0].stem} .. {files[-1].stem}) mode={args.mode}",
        flush=True,
    )
    started = datetime.now().astimezone()
    pipeline = build_pipeline(args.mode)
    result = run_backtest(pipeline, counted(stream, args.progress, stats))
    finished = datetime.now().astimezone()
    report = build_report(
        files=files,
        result=result,
        stats=stats,
        mode=args.mode,
        started_at=started,
        finished_at=finished,
    )

    summary = {
        "sessions": report["sessions"],
        "summary": report["summary"],
        "daily": [
            {
                "date": d["date"],
                "decisions": d["decisions"],
                "trades_opened": d["trades_opened"],
                "realized_pnl": d["realized_pnl"],
                "top_state": d["top_state"],
            }
            for d in report["daily"]
        ],
        "trades": report["trades"],
        "output": str(args.output) if args.output else None,
    }
    # Always emit the summary first — the live API container mounts state read-only,
    # so a report path under /var/lib/iron-spyder can fail after a long replay.
    print(json.dumps(summary, indent=2, default=str))

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print(f"wrote {args.output}", flush=True)
        except OSError as exc:
            print(f"warning: could not write {args.output}: {exc}", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
