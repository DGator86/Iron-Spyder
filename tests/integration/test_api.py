"""API surface tests (spec section 35)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from spy_der.app.api import app  # noqa: E402
from spy_der.app.state import STATE  # noqa: E402

#: Mid-session weekday — avoids after-hours 0DTE expiry flaking CI.
_API_SESSION_TS = datetime(2026, 8, 3, 15, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from spy_der.data.persistence.audit import AuditStore

    STATE.audit = AuditStore(tmp_path_factory.mktemp("api") / "audit.db")
    STATE.reset()
    STATE.set_scenario("broad_range")
    STATE.run_once(_API_SESSION_TS)
    with fastapi_testclient.TestClient(app) as test_client:
        yield test_client


REQUIRED_GET_ENDPOINTS = [
    "/health",
    "/market/current",
    "/analytics/current",
    "/analytics/gamma-profile",
    "/analytics/walls",
    "/analytics/volatility-surface",
    "/forecast/current",
    "/forecast/history",
    "/strategies/candidates",
    "/strategies/ranking",
    "/positions",
    "/orders",
    "/performance",
]


@pytest.mark.parametrize("path", REQUIRED_GET_ENDPOINTS)
def test_required_get_endpoints_respond(client, path):
    response = client.get(path)
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), dict)


def test_health_reports_mode_and_kill_switch(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"]
    assert isinstance(body["kill_switch"], list)


def test_analysis_run_returns_a_decision(client):
    body = client.post("/analysis/run").json()
    assert "decision" in body
    assert isinstance(body["traded"], bool)


def test_forecast_state_probabilities_sum_to_one(client):
    body = client.get("/forecast/current").json()
    total = sum(body["state_probabilities"].values())
    assert total == pytest.approx(1.0, abs=1e-6)
    assert body["dominant_state"] in body["state_probabilities"]


def test_candidates_expose_the_no_trade_comparison(client):
    body = client.get("/strategies/candidates").json()
    assert "no_trade_utility" in body
    assert "no_trade_margin" in body
    assert body["no_trade_margin"] > 0.0
    for candidate in body["candidates"]:
        assert candidate["max_loss"] > 0.0
        assert "exit_plan" in candidate


def test_gamma_profile_returns_a_curve(client):
    body = client.get("/analytics/gamma-profile").json()
    assert len(body["curve"]) > 10
    assert all("spot" in point and "gex" in point for point in body["curve"])


def test_kill_switch_toggles(client):
    tripped = client.post("/risk/kill-switch", params={"enabled": True}).json()
    assert tripped["tripped"] is True
    assert tripped["reasons"]

    reset = client.post("/risk/kill-switch", params={"enabled": False}).json()
    assert reset["tripped"] is False


def test_scenario_switch_validates_the_name(client):
    assert client.post("/scenario", params={"name": "strong_pin"}).status_code == 200
    assert client.post("/scenario", params={"name": "nope"}).status_code == 404
    client.post("/scenario", params={"name": "broad_range"})


def test_no_live_order_endpoints_exist():
    """Spec 35, 40: live-order routes must not be reachable at all."""
    paths = {route.path for route in app.routes}
    for forbidden in ("/live/orders", "/live/close", "/orders/submit", "/live/trade"):
        assert forbidden not in paths
    assert "/paper/orders" in paths


def test_live_mode_refuses_without_the_spec_gates():
    from spy_der.config.settings import LiveTradingDisabled, Mode, Settings

    settings = Settings(mode=Mode.LIVE)
    with pytest.raises(LiveTradingDisabled) as excinfo:
        settings.assert_live_allowed()
    assert "paper trading" in str(excinfo.value)


def test_shadow_mode_does_not_permit_order_submission():
    from spy_der.config.settings import Mode, Settings

    assert Settings(mode=Mode.SHADOW).may_submit_orders is False
    assert Settings(mode=Mode.PAPER).may_submit_orders is True
    assert Settings(mode=Mode.RESEARCH).may_submit_orders is False
