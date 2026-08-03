"""One view of everything an operator would otherwise SSH to check.

Four questions, four sections:

``services``  is each service alive?     — heartbeats
``pipeline``  is the decision loop dry?  — live_state.json
``deploy``    did my change land?        — deploy.json
``overall``   worst-of the above

Read-only and defensive: a missing file becomes a note, never an exception.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spy_der.vps.heartbeat import classify_age, read_heartbeats

__all__ = ["EXPECTED_SERVICES", "build_system_status"]

#: Services expected to publish a heartbeat. Silence is reported as never_seen.
EXPECTED_SERVICES: dict[str, str] = {
    "supervisor": "decision pipeline → live_state + runtime.db",
    "dashboard-api": "read-only status HTTP → :8788",
}


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None, "not found"
    except PermissionError:
        return None, "permission denied"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable ({type(exc).__name__})"
    return (data, None) if isinstance(data, dict) else (None, "not a JSON object")


def _age_seconds(value: Any, now: datetime) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (now - parsed).total_seconds())


def _services(state_root: Path, now: datetime) -> list[dict[str, Any]]:
    published = {entry["service"]: entry for entry in read_heartbeats(state_root, now=now)}
    out: list[dict[str, Any]] = []
    for name, purpose in EXPECTED_SERVICES.items():
        entry = published.pop(name, None)
        if entry is None:
            out.append(
                {
                    "service": name,
                    "purpose": purpose,
                    "state": "never_seen",
                    "detail": "no heartbeat has ever been published",
                }
            )
            continue
        entry["purpose"] = purpose
        out.append(entry)
    for entry in published.values():
        entry.setdefault("purpose", "")
        out.append(entry)
    return out


def _pipeline(state_root: Path, now: datetime) -> dict[str, Any]:
    live, note = _read_json(state_root / "live_state.json")
    if live is None:
        return {"state": "unavailable", "note": f"live_state {note}"}

    system = live.get("system") or {}
    age = _age_seconds(live.get("generated_at"), now)

    # A readable file is not a live pipeline. The last write a stopping
    # supervisor makes says so, and a process that died without one leaves a
    # file that simply stops ageing — so both the declared status and the age
    # have to be consulted, or a week-old snapshot reports "ok".
    if str(system.get("status")) == "stopped":
        state = "stopped"
    else:
        state = classify_age(age, float(live.get("refresh_interval_seconds") or 0.0))

    return {
        "state": state,
        "mode": live.get("mode"),
        "generated_at": live.get("generated_at"),
        "age_seconds": age,
        "open_positions": live.get("open_positions"),
        "equity": live.get("equity"),
        "kill_switches": live.get("kill_switches") or [],
        "last_decision": live.get("decision"),
        "system": system,
    }


def _deploy(state_root: Path, now: datetime) -> dict[str, Any]:
    data, note = _read_json(state_root / "deploy.json")
    if data is None:
        return {"state": "unknown", "note": f"deploy.json {note}"}
    data["deployed_age_seconds"] = _age_seconds(data.get("deployed_at"), now)
    data["state"] = "ok"
    return data


def _overall(services: list[dict[str, Any]], pipeline: dict[str, Any]) -> str:
    states = {str(s.get("state")) for s in services}
    if states & {"stale", "never_seen", "failed"}:
        return "degraded"
    # A stopped or stale pipeline is degraded even when every heartbeat is
    # fresh: the dashboard API can be perfectly healthy while the thing that
    # makes decisions is not running at all.
    if str(pipeline.get("state")) in {"stopped", "stale"}:
        return "degraded"
    if pipeline.get("state") == "unavailable":
        return "warn"
    if states & {"late", "unknown"}:
        return "warn"
    if str(pipeline.get("state")) in {"late", "unknown"}:
        return "warn"
    if pipeline.get("kill_switches"):
        return "warn"
    return "ok"


def build_system_status(
    state_root: str | Path, *, now: datetime | None = None
) -> dict[str, Any]:
    """Everything the dashboard needs to replace an SSH session."""
    root = Path(state_root)
    stamp = now or datetime.now(tz=UTC)
    services = _services(root, stamp)
    pipeline = _pipeline(root, stamp)
    return {
        "generated_at": stamp.isoformat(),
        "state_root": str(root),
        "overall": _overall(services, pipeline),
        "services": services,
        "pipeline": pipeline,
        "deploy": _deploy(root, stamp),
    }
