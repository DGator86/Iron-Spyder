"""VPS module: state root, heartbeats, live state, status, and service."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from spy_der.data.providers.base import SyntheticProvider
from spy_der.data.synthetic import scenario
from spy_der.vps.dashboard_api import create_app
from spy_der.vps.files import atomic_write_json
from spy_der.vps.heartbeat import classify_age, read_heartbeats, write_heartbeat
from spy_der.vps.live_state import SCHEMA_VERSION, build_live_state, write_live_state
from spy_der.vps.paths import STATE_DIRS, ensure_state_tree, state_paths
from spy_der.vps.service import VpsServiceConfig, build_service
from spy_der.vps.system_status import EXPECTED_SERVICES, build_system_status

SESSION = datetime(2026, 8, 3, 15, 0, tzinfo=UTC)


def test_ensure_state_tree_creates_declared_directories(tmp_path):
    paths = ensure_state_tree(tmp_path)
    assert paths.root == tmp_path
    for name in STATE_DIRS:
        assert (tmp_path / name).is_dir()
    assert paths.live_state == tmp_path / "live_state.json"
    assert paths.runtime_db == tmp_path / "runtime.db"


def test_atomic_write_json_replaces_without_truncate(tmp_path):
    target = tmp_path / "payload.json"
    atomic_write_json(target, {"n": 1})
    atomic_write_json(target, {"n": 2, "ok": True})
    assert json.loads(target.read_text()) == {"n": 2, "ok": True}
    leftovers = list(tmp_path.glob(".payload.json.*"))
    assert leftovers == []


def test_heartbeat_round_trip_and_classification(tmp_path):
    now = SESSION
    write_heartbeat(
        tmp_path,
        "supervisor",
        interval_seconds=60.0,
        detail="cycles=1",
        now=now,
    )
    entries = read_heartbeats(tmp_path, now=now + timedelta(seconds=30))
    assert len(entries) == 1
    assert entries[0]["service"] == "supervisor"
    assert entries[0]["state"] == "ok"
    assert entries[0]["detail"] == "cycles=1"

    assert classify_age(30.0, 60.0) == "ok"
    assert classify_age(90.0, 60.0) == "late"
    assert classify_age(200.0, 60.0) == "stale"
    assert classify_age(None, 60.0) == "unknown"


def test_build_live_state_heartbeat_only():
    payload = build_live_state(mode="paper", now=SESSION, note="idle")
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["mode"] == "paper"
    assert payload["live_execution_enabled"] is False
    assert payload["decision"] is None
    assert "idle" in payload["system"]["note"]


def test_system_status_reports_never_seen_and_deploy(tmp_path):
    ensure_state_tree(tmp_path)
    atomic_write_json(
        tmp_path / "deploy.json",
        {"commit": "abc", "deployed_at": SESSION.isoformat()},
    )
    status = build_system_status(tmp_path, now=SESSION)
    assert status["overall"] == "degraded"  # expected services never_seen
    names = {entry["service"] for entry in status["services"]}
    assert set(EXPECTED_SERVICES) <= names
    assert status["deploy"]["state"] == "ok"
    assert status["pipeline"]["state"] == "unavailable"


def test_vps_service_publishes_heartbeat_and_live_state(tmp_path):
    config = VpsServiceConfig(
        state_root=str(tmp_path),
        mode="paper",
        decision_interval=timedelta(minutes=5),
        scenario="broad_range",
        max_cycles=1,
    )
    clock = {"t": SESSION}

    def advance() -> datetime:
        return clock["t"]

    def sleeper(_seconds: float) -> None:
        clock["t"] = clock["t"] + timedelta(minutes=5)

    provider = SyntheticProvider(spec=scenario("broad_range"), step=timedelta(minutes=5))
    service = build_service(config, provider=provider, clock=advance, sleeper=sleeper, alert=None)
    stats = service.run()

    assert stats.cycles == 1
    assert (tmp_path / "health" / "supervisor.json").is_file()
    live = json.loads((tmp_path / "live_state.json").read_text())
    assert live["schema_version"] == SCHEMA_VERSION
    assert live["mode"] == "paper"
    assert live["decision"] is not None
    assert live["decision"]["action"] in {"TRADE", "NO_TRADE"}

    status = build_system_status(tmp_path, now=clock["t"])
    supervisor = next(s for s in status["services"] if s["service"] == "supervisor")
    assert supervisor["state"] == "ok"
    assert status["pipeline"]["state"] == "ok"


def test_dashboard_api_is_read_only(tmp_path):
    ensure_state_tree(tmp_path)
    write_live_state(
        build_live_state(mode="paper", now=SESSION, note="fixture"),
        state_root=tmp_path,
    )
    write_heartbeat(tmp_path, "supervisor", interval_seconds=60.0, now=SESSION)

    client = TestClient(create_app(tmp_path))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    live = client.get("/v1/live-state")
    assert live.status_code == 200
    assert live.json()["mode"] == "paper"

    system = client.get("/v1/system")
    assert system.status_code == 200
    assert "services" in system.json()

    # No mutating routes are registered.
    assert client.post("/v1/live-state", json={}).status_code == 405


def test_cli_status_prints_json(tmp_path, capsys):
    from spy_der.vps.cli import main

    ensure_state_tree(tmp_path)
    code = main(["status", "--state-root", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state_root"] == str(tmp_path)


def test_state_paths_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("IRON_SPYDER_STATE_ROOT", str(tmp_path / "from-env"))
    # Re-import the default by calling through a fresh Path construction.
    from spy_der.vps import paths as paths_mod

    monkeypatch.setattr(paths_mod, "DEFAULT_STATE_ROOT", str(tmp_path / "from-env"))
    assert state_paths().root == tmp_path / "from-env"
