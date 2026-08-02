"""Decision audit trail (spec section 32).

Every decision — including every *rejection* — is persisted with the full set
of versions that produced it. Spec 32 requires the rejected alternatives and
the risk vetoes, not just the trade that was taken, because the question asked
after a bad month is "what did it decline, and why", and that is unanswerable
if only fills are stored.

Backed by stdlib :mod:`sqlite3` with JSON payloads. The schema is deliberately
shallow: a handful of indexed scalar columns for querying, and the complete
record as JSON so that adding a field to the decision does not require a
migration.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    trading_day       TEXT    NOT NULL,
    decision          TEXT    NOT NULL,
    traded            INTEGER NOT NULL,
    market_state      TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    data_quality      REAL    NOT NULL,
    spot              REAL    NOT NULL,
    selected_family   TEXT,
    expected_value    REAL,
    utility           REAL,
    max_loss          REAL,
    contracts         INTEGER,
    veto_reasons      TEXT    NOT NULL,
    payload           TEXT    NOT NULL,
    schema_version    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_day ON decisions(trading_day);
CREATE INDEX IF NOT EXISTS idx_decisions_traded ON decisions(traded);

CREATE TABLE IF NOT EXISTS trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id       TEXT    NOT NULL UNIQUE,
    opened_at         TEXT    NOT NULL,
    closed_at         TEXT,
    family            TEXT    NOT NULL,
    contracts         INTEGER NOT NULL,
    entry_price       REAL    NOT NULL,
    exit_price        REAL,
    realized_pnl      REAL,
    exit_reason       TEXT,
    max_loss          REAL    NOT NULL,
    payload           TEXT    NOT NULL,
    schema_version    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades(opened_at);
"""


@dataclass(frozen=True)
class ComponentVersions:
    """Versions recorded with every decision (spec section 32)."""

    data: str = "1.0.0"
    features: str = "1.0.0"
    model: str = "1.0.0"
    calibration: str = "1.0.0"
    strategy_generator: str = "1.0.0"
    risk_engine: str = "1.0.0"
    broker_adapter: str = "paper-1.0.0"
    schema: str = SCHEMA_VERSION


@dataclass
class DecisionRecord:
    """One complete decision, traded or not."""

    timestamp: datetime
    spot: float
    decision: str
    traded: bool
    market_state: str
    state_probabilities: dict[str, float]
    confidence: float
    data_quality: float
    forecast: dict[str, Any]
    candidates: list[dict[str, Any]]
    rejected_candidates: list[dict[str, Any]]
    registry_rejections: dict[str, int]
    risk_decision: dict[str, Any]
    veto_reasons: list[str]
    selected: dict[str, Any] | None = None
    sizing: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    fill: dict[str, Any] | None = None
    versions: ComponentVersions = field(default_factory=ComponentVersions)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def trading_day(self) -> date:
        return self.timestamp.date()


class AuditStore:
    """SQLite-backed store for decisions and completed trades."""

    def __init__(self, path: str | Path = "spyder_audit.db") -> None:
        self.path = str(path)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def record_decision(self, record: DecisionRecord) -> int:
        selected = record.selected or {}
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO decisions (
                    timestamp, trading_day, decision, traded, market_state, confidence,
                    data_quality, spot, selected_family, expected_value, utility,
                    max_loss, contracts, veto_reasons, payload, schema_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.timestamp.isoformat(),
                    record.trading_day.isoformat(),
                    record.decision,
                    int(record.traded),
                    record.market_state,
                    record.confidence,
                    record.data_quality,
                    record.spot,
                    selected.get("family"),
                    selected.get("expected_value"),
                    selected.get("utility"),
                    selected.get("max_loss"),
                    (record.sizing or {}).get("contracts"),
                    json.dumps(record.veto_reasons),
                    json.dumps(_encode(record)),
                    record.versions.schema,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def record_trade(self, payload: dict[str, Any]) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO trades (
                    position_id, opened_at, closed_at, family, contracts, entry_price,
                    exit_price, realized_pnl, exit_reason, max_loss, payload, schema_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    payload["position_id"],
                    payload["opened_at"],
                    payload.get("closed_at"),
                    payload["family"],
                    payload["contracts"],
                    payload["entry_price"],
                    payload.get("exit_price"),
                    payload.get("realized_pnl"),
                    payload.get("exit_reason"),
                    payload["max_loss"],
                    json.dumps(_encode(payload)),
                    SCHEMA_VERSION,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def decisions(self, *, limit: int = 100, traded_only: bool = False) -> list[dict[str, Any]]:
        clause = "WHERE traded = 1 " if traded_only else ""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM decisions {clause}ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def trades(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def load_payload(self, decision_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT payload FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def iter_decisions(self, *, batch: int = 500) -> Iterator[dict[str, Any]]:
        offset = 0
        while True:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT * FROM decisions ORDER BY id LIMIT ? OFFSET ?", (batch, offset)
                ).fetchall()
            if not rows:
                return
            yield from (dict(row) for row in rows)
            offset += len(rows)

    def counts(self) -> dict[str, int]:
        with closing(self._connect()) as conn:
            decisions = conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"]
            traded = conn.execute(
                "SELECT COUNT(*) AS n FROM decisions WHERE traded = 1"
            ).fetchone()["n"]
            trades = conn.execute("SELECT COUNT(*) AS n FROM trades").fetchone()["n"]
        return {
            "decisions": int(decisions),
            "traded": int(traded),
            "no_trade": int(decisions) - int(traded),
            "trades": int(trades),
        }


def _encode(value: Any) -> Any:
    """JSON-safe conversion for datetimes, enums, dataclasses, and tuples."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _encode(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(_encode_key(k)): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_encode(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and value != value:  # NaN is not valid JSON
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return str(value)
    return value


def _encode_key(key: Any) -> Any:
    if isinstance(key, (datetime, date)):
        return key.isoformat()
    if isinstance(key, Enum):
        return key.value
    if isinstance(key, tuple):
        return "|".join(str(_encode_key(k)) for k in key)
    return key
