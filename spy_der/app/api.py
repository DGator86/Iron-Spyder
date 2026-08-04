"""FastAPI surface (spec section 35).

Live-order endpoints are absent by design, not merely disabled. Spec 35 says
they must be off unless explicitly enabled through configuration and broker
authorization, and spec 40 gates live trading behind six months of paper
trading and a shadow phase. Rather than ship a route guarded by a flag someone
could flip, the routes do not exist; :meth:`Settings.assert_live_allowed`
explains what would have to be true first.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from spy_der.app.state import STATE
from spy_der.data.synthetic import SCENARIOS
from spy_der.domain.enums import ExitReason, KillSwitchReason
from spy_der.pipeline import PipelineResult, force_close_all

app = FastAPI(
    title="SPY-DER",
    version="0.2.0",
    description="SPY defined-risk options intelligence and strategy optimization",
)


# --------------------------------------------------------------------------
# Health and market
# --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    result = STATE.last_result
    return {
        "status": "ok",
        "mode": STATE.settings.mode.value,
        "feed": STATE.feed,
        # Scenario only applies to the seeded generator; null when on Tradier.
        "scenario": STATE.scenario_name if STATE.feed == "synthetic" else None,
        "kill_switch": list(STATE.kill_switch.reasons),
        "open_positions": STATE.pipeline.portfolio.open_count if STATE.pipeline else 0,
        "last_decision_at": result.timestamp.isoformat() if result else None,
    }


@app.get("/market/current")
def market_current() -> dict[str, Any]:
    result = STATE.current()
    snapshot = result.snapshot
    return {
        "timestamp": snapshot.timestamp.isoformat(),
        "spot": snapshot.spot,
        "bid": snapshot.spy_quote.bid,
        "ask": snapshot.spy_quote.ask,
        "contracts": len(snapshot.option_chain),
        "expirations": [e.isoformat() for e in snapshot.expirations],
        "data_quality": result.quality.score,
        "data_quality_issues": [
            {"code": i.code, "severity": i.severity.value, "affected": i.affected}
            for i in result.quality.issues
        ],
        "tradeable": result.quality.is_tradeable,
    }


@app.get("/analytics/current")
def analytics_current() -> dict[str, Any]:
    result = STATE.current()
    if result.analytics is None:
        raise HTTPException(503, "analytics unavailable: data quality lockout")
    features = result.analytics.features
    return {
        "timestamp": features.timestamp.isoformat(),
        "spot": features.spot,
        "gex_total": features.gex_total,
        "dex_total": features.dex_total,
        "vex_total": features.vex_total,
        "cex_total": features.cex_total,
        "dealer_agreement": features.gex_agreement,
        "gamma_flip": features.gamma_flip,
        "vol_trigger": features.vol_trigger,
        "call_wall": features.call_wall,
        "put_wall": features.put_wall,
        "wall_interaction": features.wall_interaction,
        "pin_strike": features.pin_strike,
        "pin_concentration": features.pin_concentration,
        "expected_move": features.expected_move,
        "expected_move_utilization": features.em_utilization,
        "atm_iv": features.atm_iv,
        "realized_vol": features.realized_vol,
        "iv_rv_spread": features.iv_rv_spread,
        "put_skew": features.put_skew,
        "risk_reversal": features.risk_reversal,
        "term_slope": features.term_slope,
        "flow_tilt": features.flow_tilt,
        "liquidity": features.liquidity,
        "feature_version": features.feature_version,
    }


@app.get("/analytics/gamma-profile")
def gamma_profile() -> dict[str, Any]:
    result = STATE.current()
    if result.analytics is None:
        raise HTTPException(503, "analytics unavailable")
    profile = result.analytics.gamma_profile
    return {
        "convention": profile.convention.value,
        "spot": profile.spot,
        "curve": [
            {"spot": float(s), "gex": float(g)}
            for s, g in zip(profile.spots.tolist(), profile.gex.tolist())
        ],
        "flip_points": list(profile.flip_points),
        "nearest_flip": profile.nearest_flip,
        "vol_trigger": profile.vol_trigger,
        "positive_zones": [list(z) for z in profile.positive_zones],
        "negative_zones": [list(z) for z in profile.negative_zones],
    }


@app.get("/analytics/walls")
def walls() -> dict[str, Any]:
    result = STATE.current()
    if result.analytics is None:
        raise HTTPException(503, "analytics unavailable")
    wall_set = result.analytics.walls
    return {
        "spot": wall_set.spot,
        "call_wall": wall_set.call_wall.strike if wall_set.call_wall else None,
        "put_wall": wall_set.put_wall.strike if wall_set.put_wall else None,
        "concentration": wall_set.concentration,
        "inside_walls": wall_set.inside_walls,
        "position_in_range": wall_set.position_in_range,
    }


@app.get("/analytics/volatility-surface")
def volatility_surface() -> dict[str, Any]:
    result = STATE.current()
    if result.analytics is None:
        raise HTTPException(503, "analytics unavailable")
    term = result.analytics.term_structure
    skew = result.analytics.skew
    return {
        "term_structure": [
            {"expiration": e.isoformat(), "tenor_years": t, "atm_iv": iv}
            for e, t, iv in zip(term.expirations, term.tenors, term.atm_ivs)
        ],
        "term_slope": term.slope(),
        "curvature": term.curvature,
        "inverted": term.is_inverted,
        "skew": (
            {
                "atm_iv": skew.atm_iv,
                "put_skew": skew.put_skew,
                "call_skew": skew.call_skew,
                "risk_reversal": skew.risk_reversal,
                "butterfly": skew.butterfly,
            }
            if skew
            else None
        ),
    }


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------


@app.get("/forecast/current")
def forecast_current() -> dict[str, Any]:
    result = STATE.current()
    if result.forecast is None:
        raise HTTPException(503, "forecast unavailable: data quality lockout")
    return _forecast_payload(result)


@app.get("/forecast/history")
def forecast_history(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    if STATE.audit is None:
        raise HTTPException(503, "audit store unavailable")
    rows = STATE.audit.decisions(limit=limit)
    return {
        "count": len(rows),
        "history": [
            {
                "timestamp": row["timestamp"],
                "market_state": row["market_state"],
                "confidence": row["confidence"],
                "data_quality": row["data_quality"],
                "decision": row["decision"],
            }
            for row in rows
        ],
    }


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


@app.get("/strategies/candidates")
def strategy_candidates(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    result = STATE.current()
    optimization = result.optimization
    if optimization is None:
        return {"decision": "no_trade", "candidates": [], "rejected": []}
    return {
        "decision": optimization.decision,
        "trade_selected": optimization.trade_selected,
        "no_trade_utility": optimization.no_trade_utility,
        "no_trade_margin": optimization.no_trade_margin,
        "eligible_families": sorted(f.value for f in optimization.eligible_families),
        "evaluated": optimization.evaluated_count,
        "candidates": [_candidate_payload(c) for c in optimization.top(limit)],
        "rejected": [
            {"strategy_id": sid, "reasons": list(reasons)}
            for sid, reasons in optimization.rejected[:limit]
        ],
        "registry_rejections": optimization.registry_rejections,
    }


@app.get("/strategies/ranking")
def strategy_ranking(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    result = STATE.current()
    if result.optimization is None:
        return {"ranking": []}
    return {
        "ranking": [
            {
                "rank": i + 1,
                "strategy_id": c.strategy_id,
                "family": c.family.value,
                "utility": c.utility,
                "expected_value": c.expected_value,
                "expected_return_on_risk": c.expected_return_on_risk,
                "max_loss": c.max_loss,
                "beats_no_trade": c.utility
                > result.optimization.no_trade_utility + result.optimization.no_trade_margin,
            }
            for i, c in enumerate(result.optimization.top(limit))
        ]
    }


# --------------------------------------------------------------------------
# Positions, orders, performance
# --------------------------------------------------------------------------


@app.get("/positions")
def positions() -> dict[str, Any]:
    if STATE.pipeline is None:
        return {"positions": []}
    return {
        "open_risk": STATE.pipeline.portfolio.total_open_risk,
        "positions": [
            {
                "position_id": p.position_id,
                "family": p.family.value,
                "contracts": p.contracts,
                "opened_at": p.opened_at.isoformat(),
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "max_loss": p.total_max_loss,
                "legs": [
                    {
                        "expiry": leg.expiry.isoformat(),
                        "strike": leg.strike,
                        "right": leg.right.value,
                        "quantity": leg.quantity,
                    }
                    for leg in p.legs
                ],
            }
            for p in STATE.pipeline.portfolio.open_positions
        ],
    }


@app.get("/orders")
def orders(limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
    if STATE.pipeline is None:
        return {"orders": []}
    all_orders = list(STATE.pipeline.broker.orders.values())[-limit:]
    return {
        "orders": [
            {
                "order_id": o.order_id,
                "status": o.status.value,
                "intent": o.intent.value,
                "contracts": o.contracts,
                "filled": o.filled_contracts,
                "limit_price": o.limit_price,
                "reprice_attempts": o.reprice_attempts,
                "reject_reason": o.reject_reason,
            }
            for o in all_orders
        ]
    }


@app.get("/performance")
def performance() -> dict[str, Any]:
    if STATE.pipeline is None or STATE.audit is None:
        raise HTTPException(503, "pipeline unavailable")

    session = STATE.pipeline.risk_engine.session
    counts = STATE.audit.counts()
    return {
        "equity": STATE.pipeline.broker.equity,
        "realized_pnl_today": session.realized_pnl,
        "trades_today": session.trades_today,
        "consecutive_losses": session.consecutive_losses,
        "peak_equity": session.peak_equity,
        "drawdown_fraction": session.drawdown_fraction(STATE.pipeline.broker.equity),
        "slippage_ratio": session.slippage_ratio,
        "decisions": counts["decisions"],
        "traded": counts["traded"],
        "no_trade": counts["no_trade"],
        "no_trade_rate": (
            counts["no_trade"] / counts["decisions"] if counts["decisions"] else 1.0
        ),
    }


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


@app.post("/analysis/run")
def run_analysis(at: datetime | None = None) -> dict[str, Any]:
    """Run one decision cycle and return the outcome."""
    result = STATE.run_once(at)
    return {
        "timestamp": result.timestamp.isoformat(),
        "decision": result.decision,
        "traded": result.traded,
        "reasons": list(result.reasons),
        "veto_reasons": list(result.risk.veto_reasons),
        "exits": [
            {"position_id": s.position_id, "reason": s.reason.value, "detail": s.detail}
            for s in result.exits
        ],
    }


@app.post("/paper/orders")
def paper_order() -> dict[str, Any]:
    """Run a cycle in paper mode; the pipeline submits only if risk approves."""
    if not STATE.settings.may_submit_orders:
        raise HTTPException(
            403, f"mode {STATE.settings.mode.value} does not permit order submission"
        )
    result = STATE.run_once()
    return {
        "decision": result.decision,
        "traded": result.traded,
        "order": (
            {
                "order_id": result.order.order_id,
                "status": result.order.status.value,
                "limit_price": result.order.limit_price,
                "contracts": result.order.contracts,
            }
            if result.order
            else None
        ),
        "fill": (
            {
                "fill_price": result.fill.fill_price,
                "contracts": result.fill.contracts,
                "commission": result.fill.commission,
            }
            if result.fill
            else None
        ),
        "reasons": list(result.reasons) + list(result.risk.veto_reasons),
    }


@app.post("/paper/close")
def paper_close() -> dict[str, Any]:
    """Close every open paper position immediately."""
    if STATE.pipeline is None:
        raise HTTPException(503, "pipeline unavailable")
    result = STATE.current()
    closed = force_close_all(STATE.pipeline, result.snapshot, ExitReason.KILL_SWITCH)
    return {
        "closed": [
            {"position_id": p.position_id, "realized_pnl": p.realized_pnl} for p in closed
        ]
    }


@app.post("/risk/kill-switch")
def kill_switch(
    enabled: bool = Query(..., description="True trips the manual switch, False resets it"),
    reason: str = Query("manual", description="Reason label recorded with the trip"),
) -> dict[str, Any]:
    switch = STATE.kill_switch
    if enabled:
        switch.trip(KillSwitchReason.MANUAL, reason)
    else:
        switch.reset()
    return {"tripped": switch.is_tripped, "reasons": list(switch.reasons)}


@app.post("/models/reload")
def reload_models() -> dict[str, Any]:
    """Rebuild the pipeline, clearing all temporal model state."""
    STATE.reset()
    return {"status": "reloaded", "mode": STATE.settings.mode.value}


@app.post("/scenario")
def set_scenario(name: str = Query(..., description="Synthetic scenario name")) -> dict[str, Any]:
    """Point the synthetic provider at a different scenario (research aid)."""
    if name not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {name!r}; have {sorted(SCENARIOS)}")
    STATE.set_scenario(name)
    return {"scenario": name}


# --------------------------------------------------------------------------
# Payload helpers
# --------------------------------------------------------------------------


def _forecast_payload(result: PipelineResult) -> dict[str, Any]:
    forecast = result.forecast
    assert forecast is not None
    return {
        "timestamp": forecast.timestamp.isoformat(),
        "spot": forecast.spot,
        "state_probabilities": {
            state.value: p for state, p in forecast.state_probabilities.items()
        },
        "dominant_state": forecast.dominant_state.value,
        "confidence": forecast.confidence,
        "model_disagreement": forecast.model_disagreement,
        "state_entropy": forecast.state_entropy,
        "state_stability": forecast.state_stability,
        "dealer_agreement": forecast.dealer_agreement,
        "calibration_score": forecast.calibration_score,
        "expected_state_duration_minutes": forecast.expected_state_duration_minutes,
        "horizons": {
            name: {
                "minutes": h.minutes,
                "expected_price": h.expected_price,
                "median_price": h.median_price,
                "price_quantiles": {str(q): v for q, v in h.price_quantiles.items()},
                "iv_quantiles": {str(q): v for q, v in h.iv_quantiles.items()},
                "expected_mfe": h.expected_mfe,
                "expected_mae": h.expected_mae,
                "touch_probabilities": {str(k): v for k, v in h.touch_probabilities.items()},
            }
            for name, h in forecast.horizons.items()
        },
        "pin_probabilities": {str(k): v for k, v in forecast.pin_probabilities.items()},
        "range_probabilities": {
            f"{lo}-{hi}": v for (lo, hi), v in forecast.range_probabilities.items()
        },
    }


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    return {
        "strategy_id": candidate.strategy_id,
        "family": candidate.family.value,
        "entry_price": candidate.entry_price,
        "expected_value": candidate.expected_value,
        "expected_return_on_risk": candidate.expected_return_on_risk,
        "utility": candidate.utility,
        "max_loss": candidate.max_loss,
        "max_profit": candidate.max_profit,
        "breakevens": list(candidate.breakevens),
        "cvar": candidate.cvar,
        "probability_of_profit": candidate.probability_of_profit,
        "liquidity": candidate.liquidity.score,
        "fill_probability": candidate.fill_probability,
        "expected_slippage": candidate.expected_slippage,
        "assignment_risk": candidate.assignment_risk,
        "greeks": {
            "delta": candidate.greeks.delta,
            "gamma": candidate.greeks.gamma,
            "theta": candidate.greeks.theta,
            "vega": candidate.greeks.vega,
        },
        "exit_plan": {
            "profit_target_fraction": candidate.exit_plan.profit_target_fraction,
            "stop_loss_fraction": candidate.exit_plan.stop_loss_fraction,
            "max_holding_minutes": candidate.exit_plan.max_holding_minutes,
        },
        "legs": [
            {
                "expiry": leg.expiry.isoformat(),
                "strike": leg.strike,
                "right": leg.right.value,
                "quantity": leg.quantity,
                "entry_price": leg.entry_price,
            }
            for leg in candidate.legs
        ],
        "diagnostics": candidate.diagnostics,
    }
