"""Process queued backtest-optimize jobs from the VPS state root.

    python -m scripts.optimize_worker --state-root /var/lib/iron-spyder

Each job:
  1. Runs a baseline backtest on the newest stored SPY-DER session tapes.
  2. Sweeps a small optimizer config grid looking for better expectancy.
  3. Writes a run report with metrics + delta vs the prior configuration.
  4. Promotes the winning config when it beats the baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import replace
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from spy_der.backtest.engine import run_backtest
from spy_der.config.settings import load_settings
from spy_der.data.providers.spyder_recordings import (
    ImportStats,
    discover_session_files,
    iter_snapshots,
)
from spy_der.execution.paper_broker import PaperBroker, PaperBrokerConfig
from spy_der.models.forecast_engine import ForecastEngine
from spy_der.optimizer.engine import StrategyOptimizer
from spy_der.pipeline import DecisionPipeline, PipelineConfig
from spy_der.risk.engine import RiskEngine
from spy_der.vps.files import atomic_write_json
from spy_der.vps.optimize import (
    enqueue_run,
    ensure_optimize_dirs,
    load_schedule,
    optimize_paths,
    save_schedule,
    update_job_progress,
)
from spy_der.vps.paths import state_paths

log = logging.getLogger("optimize_worker")

# Small, deliberate grid — overnight desk runs must finish, not explore forever.
GRID: tuple[dict[str, float], ...] = (
    {},  # baseline (active / defaults)
    {"min_return_on_risk": 0.03, "no_trade_margin": 0.5},
    {"min_return_on_risk": 0.05, "no_trade_margin": 1.0},
    {"min_return_on_risk": 0.06, "no_trade_margin": 1.5},
    {"family_probability_threshold": 0.12, "min_return_on_risk": 0.04},
)


def build_pipeline(mode: str, optimizer_overrides: dict[str, Any]) -> DecisionPipeline:
    settings = load_settings(mode)
    opt = settings.optimizer
    no_trade = opt.no_trade
    if "no_trade_margin" in optimizer_overrides:
        no_trade = replace(no_trade, margin=float(optimizer_overrides["no_trade_margin"]))
    opt_kwargs: dict[str, Any] = {}
    for key in (
        "min_return_on_risk",
        "family_probability_threshold",
        "min_fill_probability",
        "max_spread_ratio",
    ):
        if key in optimizer_overrides:
            opt_kwargs[key] = float(optimizer_overrides[key])
    optimizer = replace(opt, no_trade=no_trade, **opt_kwargs)
    return DecisionPipeline(
        forecast_engine=ForecastEngine(config=settings.forecast),
        optimizer=StrategyOptimizer(optimizer),
        risk_engine=RiskEngine(config=settings.risk),
        broker=PaperBroker(
            config=PaperBrokerConfig(starting_equity=settings.starting_equity),
            costs=settings.costs,
        ),
        config=PipelineConfig(
            data_quality=settings.data_quality, costs=settings.costs, persist=False
        ),
    )


def _config_fingerprint(overrides: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings("backtest")
    base = {
        "min_return_on_risk": settings.optimizer.min_return_on_risk,
        "family_probability_threshold": settings.optimizer.family_probability_threshold,
        "min_fill_probability": settings.optimizer.min_fill_probability,
        "max_spread_ratio": settings.optimizer.max_spread_ratio,
        "no_trade_margin": settings.optimizer.no_trade.margin,
    }
    base.update(overrides)
    return base


def _score(metrics: dict[str, Any]) -> float:
    """Single scalar for picking a winner: expectancy, with light trade-count prior."""
    expectancy = float(metrics.get("expectancy") or 0.0)
    trades = float(metrics.get("trades") or 0.0)
    drawdown = float(metrics.get("max_drawdown_fraction") or 0.0)
    if trades < 1:
        return -1e9
    return expectancy - 50.0 * drawdown + 0.01 * math.log1p(trades)


def _load_snapshots(
    files: list[Path],
    *,
    limit: int,
) -> tuple[list[Any], ImportStats]:
    stats = ImportStats()
    stream = iter_snapshots(files, stats=stats, open_only=True)
    if limit > 0:
        stream = islice(stream, limit)
    return list(stream), stats


def _run_one(
    files: list[Path],
    snapshots: list[Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    pipeline = build_pipeline("backtest", overrides)
    result = run_backtest(pipeline, iter(snapshots))
    metrics = result.metrics().as_dict()
    # JSON-safe: replace inf
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            clean[key] = None
        else:
            clean[key] = value
    return {
        "config": _config_fingerprint(overrides),
        "metrics": clean,
        "score": _score(clean),
        "decisions": result.decision_count,
        "sessions": [p.stem for p in files],
    }


def process_job(job_path: Path, root: Path) -> dict[str, Any]:
    ops = ensure_optimize_dirs(root)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    started = datetime.now(tz=UTC).isoformat()
    job["status"] = "running"
    job["started_at"] = started
    atomic_write_json(job_path, job)
    # Load + N config evals. Total is refined once the grid is known.
    update_job_progress(
        job_path,
        phase="loading",
        message="Loading session tapes",
        current=0,
        total=1 + len(GRID),
        percent=2.0,
        status="running",
        detail="Discovering stored SPY sessions",
    )

    try:
        import_root = Path(job.get("import_path") or ops["imports"])
        files = discover_session_files(import_root, pattern="*.jsonl")
        if not files:
            raise FileNotFoundError(f"no session tapes under {import_root}")
        session_count = max(1, int(job.get("session_count") or 3))
        files = files[-session_count:]
        limit = max(20, int(job.get("snapshot_limit") or 120))
        update_job_progress(
            job_path,
            phase="loading",
            message=f"Reading {len(files)} session tape(s)",
            current=0,
            total=1 + len(GRID),
            percent=5.0,
            detail=", ".join(p.stem for p in files[-3:]),
        )
        snapshots, stats = _load_snapshots(files, limit=limit)
        if not snapshots:
            raise RuntimeError("import produced zero snapshots")

        active = {}
        if ops["active_config"].is_file():
            try:
                active = json.loads(ops["active_config"].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                active = {}
        if not isinstance(active, dict):
            active = {}

        candidates: list[dict[str, Any]] = []
        # Always evaluate the currently active config first as the prior baseline.
        baseline_overrides = {
            k: active[k]
            for k in (
                "min_return_on_risk",
                "family_probability_threshold",
                "min_fill_probability",
                "max_spread_ratio",
                "no_trade_margin",
            )
            if k in active
        }
        grid = (baseline_overrides, *[g for g in GRID if g != baseline_overrides])
        seen: set[str] = set()
        unique_grid: list[dict[str, Any]] = []
        for overrides in grid:
            key = json.dumps(overrides, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            unique_grid.append(overrides)

        # 1 load step + one step per unique config candidate.
        total_steps = 1 + len(unique_grid)
        update_job_progress(
            job_path,
            phase="loading",
            message=f"Loaded {len(snapshots)} snapshots",
            current=1,
            total=total_steps,
            detail=f"{len(unique_grid)} configs to evaluate",
        )

        for index, overrides in enumerate(unique_grid, start=1):
            label = "baseline" if index == 1 else f"candidate {index}"
            detail = (
                "defaults"
                if not overrides
                else ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
            )
            update_job_progress(
                job_path,
                phase="evaluating",
                message=f"Evaluating {label}",
                current=1 + index - 1,
                total=total_steps,
                detail=detail,
            )
            log.info("evaluate config %s", overrides or "defaults")
            candidates.append(_run_one(files, snapshots, overrides))
            update_job_progress(
                job_path,
                phase="evaluating",
                message=f"Finished {label}",
                current=1 + index,
                total=total_steps,
                detail=detail,
            )

        baseline = candidates[0]
        winner = max(candidates, key=lambda c: c["score"])
        improved = winner["score"] > baseline["score"] + 1e-9
        update_job_progress(
            job_path,
            phase="promoting" if improved else "finalizing",
            message="Promoting winning config" if improved else "No lift — keeping baseline",
            current=total_steps,
            total=total_steps,
            percent=95.0,
        )
        if improved:
            atomic_write_json(ops["active_config"], winner["config"])

        finished = datetime.now(tz=UTC)
        report = {
            "id": job["id"],
            "status": "completed",
            "reason": job.get("reason"),
            "started_at": job["started_at"],
            "finished_at": finished.isoformat(),
            "sessions": [p.stem for p in files],
            "snapshot_count": len(snapshots),
            "import": stats.summary(),
            "prior_config": baseline["config"],
            "prior_metrics": baseline["metrics"],
            "config": winner["config"],
            "metrics": winner["metrics"],
            "improved": improved,
            "config_changes": {
                key: {"prior": baseline["config"].get(key), "current": winner["config"].get(key)}
                for key in sorted(set(baseline["config"]) | set(winner["config"]))
                if baseline["config"].get(key) != winner["config"].get(key)
            },
            "candidates": [
                {
                    "config": c["config"],
                    "metrics": c["metrics"],
                    "score": c["score"],
                }
                for c in sorted(candidates, key=lambda c: c["score"], reverse=True)
            ],
        }
        atomic_write_json(ops["runs"] / f"{job['id']}.json", report)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job.update(
            {
                "status": "completed",
                "finished_at": finished.isoformat(),
                "run_id": job["id"],
                "improved": improved,
                "progress": {
                    "phase": "done",
                    "message": "Backtest complete",
                    "current": total_steps,
                    "total": total_steps,
                    "percent": 100.0,
                    "detail": "Improved" if improved else "No lift",
                    "updated_at": finished.isoformat(),
                },
            }
        )
        atomic_write_json(job_path, job)

        schedule = load_schedule(root)
        schedule["last_run_at"] = finished.isoformat()
        schedule["last_job_id"] = job["id"]
        if schedule.get("enabled"):
            schedule = save_schedule(schedule, root, now=finished)
        else:
            atomic_write_json(ops["schedule"], schedule)
        return report
    except Exception as exc:
        log.exception("optimize job %s failed", job.get("id"))
        try:
            failed = json.loads(job_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failed = job if isinstance(job, dict) else {}
        if not isinstance(failed, dict):
            failed = {}
        failed["status"] = "failed"
        failed["finished_at"] = datetime.now(tz=UTC).isoformat()
        failed["error"] = str(exc)
        failed["progress"] = {
            "phase": "failed",
            "message": "Backtest failed",
            "current": int((failed.get("progress") or {}).get("current") or 0),
            "total": int((failed.get("progress") or {}).get("total") or 0),
            "percent": float((failed.get("progress") or {}).get("percent") or 0.0),
            "detail": str(exc)[:160],
            "updated_at": failed["finished_at"],
        }
        atomic_write_json(job_path, failed)
        raise


def claim_next_job(root: Path) -> Path | None:
    ops = ensure_optimize_dirs(root)
    for path in sorted(ops["jobs"].glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("status") == "queued":
            return path
    return None


def maybe_enqueue_scheduled(root: Path, *, now: datetime) -> None:
    schedule = load_schedule(root)
    if not schedule.get("enabled"):
        return
    next_raw = schedule.get("next_run_at")
    if not next_raw:
        save_schedule(schedule, root, now=now)
        schedule = load_schedule(root)
        next_raw = schedule.get("next_run_at")
    if not next_raw:
        return
    try:
        next_at = datetime.fromisoformat(str(next_raw))
    except ValueError:
        return
    if next_at.tzinfo is None:
        next_at = next_at.replace(tzinfo=UTC)
    if now < next_at:
        return
    # Don't stack if something is already queued/running.
    ops = optimize_paths(root)
    for path in ops["jobs"].glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("status") in {"queued", "running"}:
            return
    enqueue_run(root, reason="schedule", now=now)
    save_schedule(schedule, root, now=now)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", default=None)
    parser.add_argument(
        "--once",
        action="store_true",
        help="process at most one job (default)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    root = Path(args.state_root or state_paths().root)
    now = datetime.now(tz=UTC)
    maybe_enqueue_scheduled(root, now=now)
    job_path = claim_next_job(root)
    if job_path is None:
        log.info("no queued optimize jobs under %s", root)
        return 0
    report = process_job(job_path, root)
    log.info(
        "completed %s improved=%s expectancy=%s",
        report.get("id"),
        report.get("improved"),
        (report.get("metrics") or {}).get("expectancy"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
