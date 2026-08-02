"""Service heartbeats published under the state root.

Answering "is it running?" used to mean ``systemctl status`` over SSH. That is
the wrong place for the answer: the dashboard already reads this state root,
and a heartbeat is a better signal than unit state — a unit can be
``active (running)`` while its loop is wedged, and a heartbeat that stopped
advancing says so.

Each service writes ``<state_root>/health/<service>.json`` on every pass.
Nothing here talks to systemd, so it works unprivileged and through the same
read-only file surface the dashboard API uses.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spy_der.vps.files import atomic_write_json

__all__ = [
    "HEALTH_DIRNAME",
    "STALE_INTERVAL_MULTIPLE",
    "classify_age",
    "read_heartbeats",
    "write_heartbeat",
]

HEALTH_DIRNAME = "health"

#: Beyond this multiple of a service's own interval, its heartbeat is stale.
STALE_INTERVAL_MULTIPLE = 3.0


def health_dir(state_root: str | Path) -> Path:
    return Path(state_root) / HEALTH_DIRNAME


def write_heartbeat(
    state_root: str | Path,
    service: str,
    *,
    interval_seconds: float,
    detail: str = "",
    extra: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Path:
    """Publish one service's liveness. Best-effort: never raises upward.

    A heartbeat failure must not take down the service it describes — a full
    disk should degrade the dashboard, not stop the decision loop.
    """
    stamp = now or datetime.now(tz=UTC)
    payload: dict[str, Any] = {
        "service": service,
        "updated_at": stamp.isoformat(),
        "interval_seconds": float(interval_seconds),
        "pid": os.getpid(),
        "detail": detail,
    }
    if extra:
        payload.update(extra)
    target = health_dir(state_root) / f"{service}.json"
    with contextlib.suppress(OSError):
        atomic_write_json(target, payload)
    return target


def classify_age(age_seconds: float | None, interval_seconds: float) -> str:
    """``ok`` | ``late`` | ``stale`` | ``unknown``."""
    if age_seconds is None:
        return "unknown"
    if interval_seconds <= 0:
        return "ok"
    if age_seconds <= interval_seconds:
        return "ok"
    if age_seconds <= interval_seconds * STALE_INTERVAL_MULTIPLE:
        return "late"
    return "stale"


def read_heartbeats(
    state_root: str | Path, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Every published heartbeat with age classification, name-sorted."""
    directory = health_dir(state_root)
    if not directory.is_dir():
        return []
    stamp = now or datetime.now(tz=UTC)
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        entry: dict[str, Any] = {"service": path.stem}
        try:
            with open(path, encoding="utf-8") as handle:
                body = json.load(handle)
        except PermissionError:
            entry.update(state="unknown", note="permission denied reading heartbeat")
            out.append(entry)
            continue
        except (OSError, json.JSONDecodeError):
            entry.update(state="unknown", note="heartbeat unreadable")
            out.append(entry)
            continue
        if not isinstance(body, dict):
            entry.update(state="unknown", note="heartbeat is not an object")
            out.append(entry)
            continue

        entry.update(body)
        interval = float(body.get("interval_seconds") or 0.0)
        age = _age_seconds(body.get("updated_at"), stamp)
        entry["age_seconds"] = age
        if body.get("health") == "failed":
            entry["state"] = "failed"
        else:
            entry["state"] = classify_age(age, interval)
        out.append(entry)
    return out


def _age_seconds(updated_at: Any, now: datetime) -> float | None:
    if not isinstance(updated_at, str) or not updated_at:
        return None
    try:
        parsed = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (now - parsed).total_seconds())
