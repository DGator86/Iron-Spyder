"""Trade journal assembled from the supervisor's runtime + audit stores.

The FastAPI ``AppState`` process has its own empty paper book. The live desk
must read the supervisor's ``runtime.db`` (open) and ``audit.db`` (closed) so
the journal matches what the VPS is actually managing.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from spy_der.domain.enums import CONTRACT_MULTIPLIER
from spy_der.vps.paths import StatePaths, state_paths

__all__ = ["build_journal"]


def build_journal(
    root: str | Path | None = None,
    *,
    limit: int = 100,
) -> dict[str, Any]:
    """Return open + closed journal rows for the dashboard."""
    paths = state_paths(root)
    open_rows = _load_open_positions(paths)
    closed_rows = _load_closed_trades(paths, limit=max(1, limit))
    entries = [*open_rows, *closed_rows]
    realized = sum(float(e["result"] or 0.0) for e in closed_rows)
    unrealized = sum(float(e["unrealizedPnl"] or 0.0) for e in open_rows)
    return {
        "source": "supervisor",
        "open_count": len(open_rows),
        "closed_count": len(closed_rows),
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "entries": entries,
        "runtime_db": str(paths.runtime_db),
        "audit_db": str(paths.audit_db),
    }


def _load_open_positions(paths: StatePaths) -> list[dict[str, Any]]:
    if not paths.runtime_db.is_file():
        return []
    try:
        with sqlite3.connect(f"file:{paths.runtime_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT payload FROM runtime_state WHERE id = 1").fetchone()
    except sqlite3.Error:
        return []
    if row is None:
        return []
    try:
        payload = json.loads(row["payload"])
    except (TypeError, json.JSONDecodeError):
        return []
    out: list[dict[str, Any]] = []
    for raw in payload.get("positions") or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("closed_at"):
            continue
        out.append(_open_entry(raw))
    out.sort(key=lambda e: e.get("openedAt") or "", reverse=True)
    return out


def _load_closed_trades(paths: StatePaths, *, limit: int) -> list[dict[str, Any]]:
    if not paths.audit_db.is_file():
        return []
    try:
        with sqlite3.connect(f"file:{paths.audit_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [_closed_entry(dict(row)) for row in rows]


def _open_entry(raw: dict[str, Any]) -> dict[str, Any]:
    contracts = int(raw.get("contracts") or 1)
    entry_price = float(raw.get("entry_price") or 0.0)
    current_price = raw.get("current_price")
    current = float(current_price) if current_price is not None else None
    commission = float(raw.get("entry_commission") or 0.0)
    # Premium outlay / credit received (debits positive per Position convention).
    cost = abs(entry_price) * contracts * CONTRACT_MULTIPLIER + commission
    unrealized = None
    current_value = None
    if current is not None:
        current_value = current * contracts * CONTRACT_MULTIPLIER
        unrealized = (current - entry_price) * contracts * CONTRACT_MULTIPLIER
    max_loss = float(raw.get("max_loss_per_contract") or 0.0) * contracts
    return {
        "id": str(raw.get("position_id") or ""),
        "status": "open",
        "strategy": str(raw.get("family") or "Unknown"),
        "openedAt": raw.get("opened_at"),
        "closedAt": None,
        "contracts": contracts,
        "entryPrice": entry_price,
        "cost": cost,
        "currentPrice": current,
        "currentValue": current_value,
        "realizedPnl": None,
        "unrealizedPnl": unrealized,
        "result": unrealized,
        "exitReason": None,
        "maxLoss": max_loss,
        "legs": list(raw.get("legs") or []),
    }


def _closed_entry(row: dict[str, Any]) -> dict[str, Any]:
    contracts = int(row.get("contracts") or 1)
    entry_price = float(row.get("entry_price") or 0.0)
    exit_price = row.get("exit_price")
    exit = float(exit_price) if exit_price is not None else None
    realized = row.get("realized_pnl")
    pnl = float(realized) if realized is not None else None
    cost = entry_price * contracts * CONTRACT_MULTIPLIER
    legs: list[Any] = []
    payload_raw = row.get("payload")
    if payload_raw:
        try:
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            if isinstance(payload, dict):
                legs = list(payload.get("legs") or [])
        except (TypeError, json.JSONDecodeError):
            legs = []
    return {
        "id": str(row.get("position_id") or ""),
        "status": "closed",
        "strategy": str(row.get("family") or "Unknown"),
        "openedAt": row.get("opened_at"),
        "closedAt": row.get("closed_at"),
        "contracts": contracts,
        "entryPrice": entry_price,
        "cost": cost,
        "currentPrice": exit,
        "currentValue": None,
        "realizedPnl": pnl,
        "unrealizedPnl": None,
        "result": pnl,
        "exitReason": row.get("exit_reason"),
        "maxLoss": float(row.get("max_loss") or 0.0),
        "legs": legs,
    }
