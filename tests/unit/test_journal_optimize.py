"""Trade journal + backtest-optimize desk surfaces."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from spy_der.vps.dashboard_api import create_app
from spy_der.vps.journal import build_journal
from spy_der.vps.optimize import (
    build_optimize_status,
    compare_metrics,
    enqueue_run,
    ensure_optimize_dirs,
    load_schedule,
    save_schedule,
    update_job_progress,
)
from spy_der.vps.paths import ensure_state_tree


def _seed_runtime(root, *, positions):
    ensure_state_tree(root)
    db = root / "runtime.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE runtime_state ("
        "id INTEGER PRIMARY KEY, saved_at TEXT, schema_version TEXT, payload TEXT)"
    )
    conn.execute(
        "INSERT INTO runtime_state VALUES (1, ?, ?, ?)",
        (
            datetime(2026, 8, 4, tzinfo=UTC).isoformat(),
            "iron_spyder.runtime.v1",
            json.dumps({"positions": positions, "equity": 100_000}),
        ),
    )
    conn.commit()
    conn.close()


def _seed_audit(root, *, trades):
    ensure_state_tree(root)
    db = root / "audit" / "audit.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            position_id TEXT,
            opened_at TEXT,
            closed_at TEXT,
            family TEXT,
            contracts INTEGER,
            entry_price REAL,
            exit_price REAL,
            realized_pnl REAL,
            exit_reason TEXT,
            max_loss REAL,
            payload TEXT,
            schema_version TEXT
        );
        """
    )
    for trade in trades:
        conn.execute(
            "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trade["position_id"],
                trade["opened_at"],
                trade["closed_at"],
                trade["family"],
                trade["contracts"],
                trade["entry_price"],
                trade["exit_price"],
                trade["realized_pnl"],
                trade["exit_reason"],
                trade["max_loss"],
                json.dumps({"legs": []}),
                "1",
            ),
        )
    conn.commit()
    conn.close()


def test_journal_merges_open_and_closed(tmp_path):
    _seed_runtime(
        tmp_path,
        positions=[
            {
                "position_id": "P-open",
                "opened_at": "2026-08-04T14:00:00+00:00",
                "family": "IronCondor",
                "contracts": 1,
                "entry_price": 1.20,
                "current_price": 0.80,
                "entry_commission": 2.0,
                "max_loss_per_contract": 3.80,
                "legs": [],
            }
        ],
    )
    _seed_audit(
        tmp_path,
        trades=[
            {
                "position_id": "P-closed",
                "opened_at": "2026-08-03T15:00:00+00:00",
                "closed_at": "2026-08-03T18:00:00+00:00",
                "family": "BullCallDebitSpread",
                "contracts": 2,
                "entry_price": 1.50,
                "exit_price": 2.10,
                "realized_pnl": 120.0,
                "exit_reason": "profit_target",
                "max_loss": 700.0,
            }
        ],
    )
    journal = build_journal(tmp_path)
    assert journal["open_count"] == 1
    assert journal["closed_count"] == 1
    open_row = next(e for e in journal["entries"] if e["status"] == "open")
    assert open_row["strategy"] == "IronCondor"
    assert open_row["unrealizedPnl"] == pytest_approx(-40.0)
    closed = next(e for e in journal["entries"] if e["status"] == "closed")
    assert closed["result"] == 120.0


def pytest_approx(value, rel=1e-6):
    # tiny local helper so we don't need pytest.approx import aliasing
    class _A:
        def __eq__(self, other):
            return abs(float(other) - value) <= rel * max(1.0, abs(value))

    return _A()


def test_optimize_enqueue_and_schedule(tmp_path):
    ensure_optimize_dirs(tmp_path)
    job = enqueue_run(tmp_path, reason="manual", session_count=2, snapshot_limit=40)
    assert job["status"] == "queued"
    assert job["progress"]["phase"] == "queued"
    assert job["progress"]["percent"] == 0.0
    job_path = tmp_path / "reports" / "optimize" / "jobs" / f"{job['id']}.json"
    assert job_path.is_file()

    updated = update_job_progress(
        job_path,
        phase="evaluating",
        message="Evaluating candidate 2",
        current=3,
        total=6,
        detail="min_return_on_risk=0.05",
        status="running",
    )
    assert updated["status"] == "running"
    assert updated["progress"]["percent"] == 50.0
    assert updated["progress"]["message"] == "Evaluating candidate 2"

    schedule = save_schedule(
        {"enabled": True, "cadence": "daily", "hour_utc": 7},
        tmp_path,
        now=datetime(2026, 8, 4, 8, 0, tzinfo=UTC),
    )
    assert schedule["enabled"] is True
    assert schedule["next_run_at"]

    status = build_optimize_status(tmp_path)
    assert status["active_job"]["id"] == job["id"]
    assert status["active_job"]["progress"]["percent"] == 50.0
    assert status["schedule"]["enabled"] is True


def test_compare_metrics_delta():
    deltas = compare_metrics(
        {"expectancy": 12.0, "win_rate": 0.6, "trades": 10},
        {"expectancy": 8.0, "win_rate": 0.5, "trades": 10},
    )
    assert deltas["expectancy"]["delta"] == 4.0
    assert deltas["win_rate"]["delta"] == pytest_approx(0.1)


def test_desk_api_journal_and_optimize_routes(tmp_path):
    ensure_state_tree(tmp_path)
    _seed_runtime(tmp_path, positions=[])
    client = TestClient(create_app(tmp_path))
    journal = client.get("/journal")
    assert journal.status_code == 200
    assert journal.json()["entries"] == []

    queued = client.post("/optimize/run", json={"session_count": 2, "snapshot_limit": 40})
    assert queued.status_code == 200
    body = queued.json()
    assert body["queued"] is True
    assert body["job"]["status"] == "queued"

    scheduled = client.post(
        "/optimize/schedule",
        json={"enabled": True, "cadence": "weekly", "hour_utc": 5},
    )
    assert scheduled.status_code == 200
    assert scheduled.json()["schedule"]["cadence"] == "weekly"
    assert load_schedule(tmp_path)["enabled"] is True

    status = client.get("/optimize")
    assert status.status_code == 200
    assert status.json()["active_job"] is not None
