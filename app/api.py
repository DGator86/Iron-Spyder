from __future__ import annotations

from fastapi import FastAPI

from app.state import STATE

app = FastAPI(title="SPY-DER")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/analysis/current")
def analysis_current() -> dict[str, object]:
    s = STATE.snapshot()
    return {
        "spy_price": s["quote"].last,
        "expected_move": s["expected_move"],
        "gamma_flip": s["gamma_flip"],
        "call_wall": s["call_wall"],
        "put_wall": s["put_wall"],
        "market_state": s["market_state"].state,
        "market_state_probabilities": s["market_state"].probabilities,
    }


@app.get("/analytics/gamma-profile")
def gamma_profile_endpoint() -> dict[str, object]:
    s = STATE.snapshot()
    return {"profile": s["gamma_profile"]}


@app.get("/forecast/current")
def forecast_current() -> dict[str, object]:
    s = STATE.snapshot()
    return {"expected_move": s["expected_move"], "market_state_probabilities": s["market_state"].probabilities}


@app.get("/strategies/candidates")
def strategies_candidates() -> dict[str, object]:
    s = STATE.snapshot()
    return {"best": s["best"], "candidates": [c.model_dump() for c in s["ranked"]]}


@app.get("/strategies/ranking")
def strategies_ranking() -> dict[str, object]:
    s = STATE.snapshot()
    return {"ranking": [c.model_dump() for c in s["ranked"]]}


@app.get("/positions")
def positions() -> dict[str, object]:
    return {"positions": [p.model_dump() for p in STATE.broker.positions.values()]}


@app.get("/orders")
def orders() -> dict[str, object]:
    return {"orders": [o.model_dump() for o in STATE.broker.orders.values()]}


@app.post("/paper/orders")
def paper_orders() -> dict[str, object]:
    s = STATE.snapshot()
    if not s["ranked"]:
        return {"status": "no_trade"}
    order, fill, reason = STATE.broker.execute_candidate(s["ranked"][0], len(STATE.broker.positions))
    return {"order": order.model_dump(), "fill": None if fill is None else fill.model_dump(), "reason": reason}


@app.post("/risk/kill-switch")
def kill_switch(enabled: bool) -> dict[str, object]:
    STATE.broker.kill_switch.set(enabled)
    return {"enabled": STATE.broker.kill_switch.enabled}
