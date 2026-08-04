"""Backtest-optimizer job queue, schedule, and metrics comparison.

Jobs and run reports live under the VPS state root so the status API (RW) can
enqueue work and the optimize worker can process it without touching the live
supervisor decision loop.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from spy_der.vps.files import atomic_write_json
from spy_der.vps.paths import StatePaths, state_paths

__all__ = [
    "METRIC_KEYS",
    "build_optimize_status",
    "compare_metrics",
    "enqueue_run",
    "list_runs",
    "load_schedule",
    "optimize_paths",
    "save_schedule",
]

METRIC_KEYS: tuple[str, ...] = (
    "expectancy",
    "win_rate",
    "profit_factor",
    "total_pnl",
    "max_drawdown_fraction",
    "sharpe_like",
    "no_trade_rate",
    "trades",
)

DEFAULT_SCHEDULE: dict[str, Any] = {
    "enabled": False,
    "cadence": "daily",
    "hour_utc": 6,
    "session_count": 3,
    "snapshot_limit": 120,
    "next_run_at": None,
    "last_run_at": None,
    "last_job_id": None,
}


def optimize_paths(root: str | Path | None = None) -> dict[str, Path]:
    paths = state_paths(root)
    base = paths.reports / "optimize"
    return {
        "root": base,
        "jobs": base / "jobs",
        "runs": base / "runs",
        "schedule": base / "schedule.json",
        "active_config": paths.configs / "optimizer.json",
        "imports": paths.root / "imports" / "spy-der" / "market",
    }


def ensure_optimize_dirs(root: str | Path | None = None) -> dict[str, Path]:
    ops = optimize_paths(root)
    for key in ("root", "jobs", "runs"):
        ops[key].mkdir(parents=True, exist_ok=True)
    state_paths(root).configs.mkdir(parents=True, exist_ok=True)
    return ops


def load_schedule(root: str | Path | None = None) -> dict[str, Any]:
    ops = optimize_paths(root)
    if not ops["schedule"].is_file():
        return dict(DEFAULT_SCHEDULE)
    try:
        data = json.loads(ops["schedule"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_SCHEDULE)
    if not isinstance(data, dict):
        return dict(DEFAULT_SCHEDULE)
    out = dict(DEFAULT_SCHEDULE)
    out.update(data)
    return out


def save_schedule(
    updates: dict[str, Any],
    root: str | Path | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    ensure_optimize_dirs(root)
    ops = optimize_paths(root)
    current = load_schedule(root)
    allowed = {
        "enabled",
        "cadence",
        "hour_utc",
        "session_count",
        "snapshot_limit",
    }
    for key, value in updates.items():
        if key in allowed:
            current[key] = value
    stamp = now or datetime.now(tz=UTC)
    if current.get("enabled"):
        current["next_run_at"] = _next_run_at(current, after=stamp).isoformat()
    else:
        current["next_run_at"] = None
    atomic_write_json(ops["schedule"], current)
    return current


def enqueue_run(
    root: str | Path | None = None,
    *,
    reason: str = "manual",
    session_count: int | None = None,
    snapshot_limit: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Queue a backtest-optimize job for the worker."""
    ensure_optimize_dirs(root)
    ops = optimize_paths(root)
    schedule = load_schedule(root)
    stamp = now or datetime.now(tz=UTC)
    job_id = f"opt-{stamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    job = {
        "id": job_id,
        "status": "queued",
        "reason": reason,
        "created_at": stamp.isoformat(),
        "started_at": None,
        "finished_at": None,
        "session_count": int(session_count or schedule.get("session_count") or 3),
        "snapshot_limit": int(snapshot_limit or schedule.get("snapshot_limit") or 120),
        "import_path": str(ops["imports"]),
        "error": None,
    }
    atomic_write_json(ops["jobs"] / f"{job_id}.json", job)
    return job


def list_runs(root: str | Path | None = None, *, limit: int = 20) -> list[dict[str, Any]]:
    ops = optimize_paths(root)
    if not ops["runs"].is_dir():
        return []
    files = sorted(ops["runs"].glob("*.json"), key=lambda p: p.name, reverse=True)
    out: list[dict[str, Any]] = []
    for path in files[: max(1, limit)]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def active_job(root: str | Path | None = None) -> dict[str, Any] | None:
    ops = optimize_paths(root)
    if not ops["jobs"].is_dir():
        return None
    queued: list[dict[str, Any]] = []
    for path in sorted(ops["jobs"].glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        status = data.get("status")
        if status == "running":
            return data
        if status == "queued":
            queued.append(data)
    return queued[0] if queued else None


def compare_metrics(
    current: dict[str, Any] | None,
    prior: dict[str, Any] | None,
) -> dict[str, dict[str, float | None]]:
    """Field-by-field delta. Lower is better for drawdown and no_trade_rate."""
    out: dict[str, dict[str, float | None]] = {}
    cur = current or {}
    prv = prior or {}
    for key in METRIC_KEYS:
        c = _as_float(cur.get(key))
        p = _as_float(prv.get(key))
        delta = None if c is None or p is None else c - p
        out[key] = {"prior": p, "current": c, "delta": delta}
    return out


def build_optimize_status(root: str | Path | None = None) -> dict[str, Any]:
    ops = optimize_paths(root)
    schedule = load_schedule(root)
    runs = list_runs(root, limit=20)
    latest = runs[0] if runs else None
    prior = runs[1] if len(runs) > 1 else None
    latest_metrics = (latest or {}).get("metrics") if latest else None
    prior_metrics = (prior or {}).get("metrics") if prior else None
    # Prefer explicit prior_metrics stored on the run (baseline of that job).
    if latest and isinstance(latest.get("prior_metrics"), dict):
        prior_metrics = latest["prior_metrics"]
    sessions = _count_sessions(ops["imports"])
    active_cfg = _load_json(ops["active_config"]) or {}
    return {
        "schedule": schedule,
        "active_job": active_job(root),
        "latest_run": latest,
        "prior_run": prior,
        "deltas": compare_metrics(
            latest_metrics if isinstance(latest_metrics, dict) else None,
            prior_metrics if isinstance(prior_metrics, dict) else None,
        ),
        "active_config": active_cfg,
        "data": {
            "sessions": sessions,
            "path": str(ops["imports"]),
            "available": sessions > 0,
        },
        "runs": runs[:10],
    }


def _count_sessions(path: Path) -> int:
    if not path.is_dir():
        return 0
    return len(list(path.glob("*.jsonl")))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _next_run_at(schedule: dict[str, Any], *, after: datetime) -> datetime:
    hour = int(schedule.get("hour_utc") or 6)
    cadence = str(schedule.get("cadence") or "daily")
    candidate = after.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    candidate = candidate.replace(hour=hour)
    if candidate <= after.astimezone(UTC):
        candidate += timedelta(days=1)
    if cadence == "weekly":
        # Next Monday (weekday 0) at hour_utc.
        while candidate.weekday() != 0:
            candidate += timedelta(days=1)
    return candidate
