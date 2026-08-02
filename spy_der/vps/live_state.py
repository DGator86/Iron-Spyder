"""Publish ``live_state.json`` for the dashboard and status API.

The file is Iron-Spyder-owned and lives under the VPS state root. Nothing here
writes into another project's tree. Callers build a payload from a
:class:`~spy_der.pipeline.PipelineResult` (or a heartbeat-only snapshot when
the market is closed) and this module writes it atomically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spy_der.pipeline import PipelineResult
from spy_der.vps.files import atomic_write_json
from spy_der.vps.paths import state_paths

__all__ = ["SCHEMA_VERSION", "build_live_state", "write_live_state"]

SCHEMA_VERSION = "iron_spyder.live.v1"


def build_live_state(
    *,
    mode: str,
    stats: dict[str, Any] | None = None,
    result: PipelineResult | None = None,
    kill_switches: list[str] | None = None,
    open_positions: int = 0,
    equity: float | None = None,
    note: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assemble a dashboard-facing live-state payload."""
    stamp = now or datetime.now(tz=UTC)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": stamp.isoformat(),
        "mode": mode,
        "role": "decision_pipeline",
        "live_execution_enabled": False,
        "open_positions": open_positions,
        "equity": equity,
        "kill_switches": list(kill_switches or ()),
        "stats": stats or {},
        "system": {
            "status": "running" if result is not None or stats else "heartbeat",
            "note": note or "VPS supervisor alive; live trading disabled",
        },
    }
    if result is None:
        payload["decision"] = None
        return payload

    forecast = result.forecast
    candidate = result.candidate
    payload["decision"] = {
        "action": "TRADE" if result.traded else "NO_TRADE",
        "label": result.decision,
        "reasons": list(result.reasons),
        "timestamp": result.timestamp.isoformat(),
        "data_quality": result.quality.score,
        "tradeable": result.quality.is_tradeable,
        "spot": result.snapshot.spot,
        "state": forecast.dominant_state.value if forecast is not None else None,
        "confidence": forecast.confidence if forecast is not None else None,
        "family": candidate.family.value if candidate is not None else None,
        "utility": candidate.utility if candidate is not None else None,
    }
    return payload


def write_live_state(
    payload: dict[str, Any],
    *,
    state_root: str | Path | None = None,
    path: str | Path | None = None,
) -> Path:
    """Atomically write live state under the state root (or an explicit path)."""
    target = Path(path) if path is not None else state_paths(state_root).live_state
    return atomic_write_json(target, payload)
