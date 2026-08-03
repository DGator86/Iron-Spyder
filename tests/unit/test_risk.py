"""Risk engine, sizing, and kill switches (spec sections 28, 29, 42, 44)."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from spy_der.domain.enums import (
    CONTRACT_MULTIPLIER,
    KillSwitchReason,
    MarketState,
    OptionRight,
    StrategyFamily,
)
from spy_der.domain.execution import BrokerState
from spy_der.domain.forecast import HORIZON_MINUTES, ForecastBundle, HorizonForecast
from spy_der.domain.strategy import (
    ExitPlan,
    LiquidityProfile,
    Strategy,
    StrategyCandidate,
    StrategyGreeks,
)
from spy_der.risk.engine import PortfolioState, RiskEngine
from spy_der.risk.limits import KillSwitch, RiskConfig, SessionState
from spy_der.risk.sizing import MIN_CONFIDENCE_FACTOR, confidence_factor, size_position
from tests.conftest import leg

CALL = OptionRight.CALL
PUT = OptionRight.PUT
NOW = datetime(2026, 8, 3, 15, 30, tzinfo=UTC)


def make_forecast(**overrides) -> ForecastBundle:
    states = overrides.pop("state_probabilities", None) or {
        state: (0.6 if state is MarketState.BROAD_RANGE else 0.04) for state in MarketState
    }
    total = sum(states.values())
    states = {k: v / total for k, v in states.items()}

    horizon = HorizonForecast(
        horizon="eod",
        minutes=HORIZON_MINUTES["eod"],
        price_quantiles={0.1: 545.0, 0.5: 550.0, 0.9: 555.0},
        iv_quantiles={0.1: 0.12, 0.5: 0.14, 0.9: 0.18},
        expected_price=550.0,
        median_price=550.0,
        price_std=4.0,
        skewness=0.0,
        kurtosis=0.0,
    )
    defaults = dict(
        timestamp=NOW,
        spot=550.0,
        horizons={"eod": horizon, "expiration": horizon},
        state_probabilities=states,
        confidence=0.6,
        model_disagreement=0.05,
        calibration_score=0.9,
        state_stability=0.95,
        dealer_agreement=0.85,
        data_quality=0.95,
    )
    defaults.update(overrides)
    return ForecastBundle(**defaults)


def make_candidate(**overrides) -> StrategyCandidate:
    strategy = overrides.pop(
        "strategy",
        Strategy(
            StrategyFamily.BULL_CALL_DEBIT_SPREAD,
            (leg(1, 550, CALL, 4.0), leg(-1, 555, CALL, 2.0)),
        ),
    )
    defaults = dict(
        strategy_id="test-1",
        family=strategy.family,
        strategy=strategy,
        entry_price=2.0,
        max_profit=300.0,
        max_loss=200.0,
        breakevens=(552.0,),
        greeks=StrategyGreeks(delta=25.0, gamma=3.0, theta=-4.0, vega=6.0),
        liquidity=LiquidityProfile(
            score=0.85,
            worst_spread_ratio=0.03,
            total_open_interest=8_000,
            total_volume=1_200,
            min_quote_size=60,
        ),
        fill_probability=0.85,
        expected_slippage=4.0,
        expected_value=30.0,
        expected_return_on_risk=0.15,
        cvar=120.0,
        utility=22.0,
        assignment_risk=0.05,
        exit_plan=ExitPlan(
            profit_target_fraction=0.6,
            stop_loss_fraction=0.7,
            max_holding_minutes=180.0,
        ),
        capital_usage=200.0,
    )
    defaults.update(overrides)
    return StrategyCandidate(**defaults)


def make_snapshot(scored_snapshot, **context):
    snapshot = scored_snapshot("broad_range")
    return replace(
        snapshot,
        timestamp=NOW,
        context=replace(snapshot.context, **context) if context else snapshot.context,
    )


# --------------------------------------------------------------------------
# Approval path
# --------------------------------------------------------------------------


def test_clean_candidate_is_approved_and_sized(scored_snapshot):
    engine = RiskEngine()
    decision, sizing = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(equity=100_000.0, buying_power=100_000.0),
    )
    assert decision.approved, decision.veto_reasons
    assert decision.max_contracts >= 1
    assert decision.approved_risk > 0.0
    assert sizing is not None and sizing.is_tradeable


def test_missing_candidate_is_vetoed(scored_snapshot):
    engine = RiskEngine()
    decision, _ = engine.evaluate(
        None,
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(),
    )
    assert not decision.approved
    assert "no candidate submitted" in decision.veto_reasons


# --------------------------------------------------------------------------
# Vetoes (spec 28.1, 29)
# --------------------------------------------------------------------------

VETO_CASES = {
    "risk too large": ({"max_loss": 5_000.0}, "exceeds"),
    "return on risk too low": ({"expected_return_on_risk": 0.001}, "return on risk"),
    "illiquid": (
        {
            "liquidity": LiquidityProfile(
                score=0.05,
                worst_spread_ratio=0.9,
                total_open_interest=10,
                total_volume=1,
                min_quote_size=1,
            )
        },
        "liquidity",
    ),
    "assignment risk": ({"assignment_risk": 0.9}, "assignment risk"),
    "excess slippage": ({"expected_slippage": 150.0}, "slippage"),
    "gamma exposure": (
        {"greeks": StrategyGreeks(gamma=99_999.0)},
        "gamma exposure",
    ),
    "vega exposure": ({"greeks": StrategyGreeks(vega=99_999.0)}, "vega exposure"),
}


@pytest.mark.parametrize("name", sorted(VETO_CASES))
def test_candidate_vetoes(name, scored_snapshot):
    overrides, fragment = VETO_CASES[name]
    engine = RiskEngine()
    decision, _ = engine.evaluate(
        make_candidate(**overrides),
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(),
    )
    assert not decision.approved
    assert any(fragment in reason for reason in decision.veto_reasons), decision.veto_reasons


def test_low_confidence_is_vetoed(scored_snapshot):
    engine = RiskEngine()
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(confidence=0.05),
        portfolio=PortfolioState(),
        broker=BrokerState(),
    )
    assert not decision.approved
    assert any("confidence" in r for r in decision.veto_reasons)


def test_transition_state_triggers_structural_veto(scored_snapshot):
    engine = RiskEngine()
    states = dict.fromkeys(MarketState, 0.02)
    states[MarketState.TRANSITION] = 0.7
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(state_probabilities=states),
        portfolio=PortfolioState(),
        broker=BrokerState(),
    )
    assert not decision.approved
    assert any("structural veto" in r for r in decision.veto_reasons)


def test_scheduled_event_blocks_entry(scored_snapshot):
    engine = RiskEngine()
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot, scheduled_events=("FOMC",)),
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(),
    )
    assert not decision.approved
    assert any("event lockout" in r for r in decision.veto_reasons)


def test_late_session_blocks_entry(scored_snapshot):
    engine = RiskEngine()
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot, minutes_to_close=5.0),
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(),
    )
    assert not decision.approved
    assert any("minutes to close" in r for r in decision.veto_reasons)


def test_broker_problems_block_entry(scored_snapshot):
    engine = RiskEngine()
    for broker, fragment in (
        (BrokerState(connected=False), "disconnected"),
        (BrokerState(reconciled=False), "not reconciled"),
        (BrokerState(discrepancies=("550C mismatch",)), "discrepancies"),
    ):
        decision, _ = engine.evaluate(
            make_candidate(),
            snapshot=make_snapshot(scored_snapshot),
            forecast=make_forecast(),
            portfolio=PortfolioState(),
            broker=broker,
        )
        assert not decision.approved
        assert any(fragment in r for r in decision.veto_reasons)


def test_data_quality_lockout(scored_snapshot):
    engine = RiskEngine()
    snapshot = replace(make_snapshot(scored_snapshot), data_quality_score=0.2)
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=snapshot,
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(),
    )
    assert not decision.approved
    assert KillSwitchReason.DATA_QUALITY in engine.kill_switch


# --------------------------------------------------------------------------
# Portfolio limits (spec 28.2)
# --------------------------------------------------------------------------


def test_open_position_limit_blocks_a_second_trade(scored_snapshot):
    from spy_der.domain.execution import Position

    engine = RiskEngine()
    portfolio = PortfolioState()
    portfolio.add(
        Position(
            position_id="P1",
            opened_at=NOW,
            family=StrategyFamily.IRON_CONDOR,
            strategy=Strategy(
                StrategyFamily.BULL_CALL_DEBIT_SPREAD,
                (leg(1, 550, CALL, 4.0), leg(-1, 555, CALL, 2.0)),
            ),
            contracts=1,
            entry_price=2.0,
            max_loss_per_contract=200.0,
            max_profit_per_contract=300.0,
            exit_plan=ExitPlan(0.6, 0.7, 180.0),
        )
    )
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(),
        portfolio=portfolio,
        broker=BrokerState(),
    )
    assert not decision.approved
    assert any("open strategies" in r for r in decision.veto_reasons)


# --------------------------------------------------------------------------
# Kill switches (spec 42)
# --------------------------------------------------------------------------


def test_kill_switch_latches_until_reset():
    switch = KillSwitch()
    switch.trip(KillSwitchReason.SLIPPAGE, "3.5x expected")
    assert switch.is_tripped
    # The condition clearing does not clear the switch.
    switch.trip(KillSwitchReason.SLIPPAGE, "back to normal")
    assert "3.5x expected" in switch.reasons[0]
    switch.reset(KillSwitchReason.SLIPPAGE)
    assert not switch.is_tripped


def test_daily_loss_trips_the_kill_switch(scored_snapshot):
    engine = RiskEngine()
    engine.session.roll_to(NOW.date(), 100_000.0)
    engine.session.record_trade(-2_000.0, 98_000.0)
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(equity=98_000.0),
    )
    assert not decision.approved
    assert KillSwitchReason.DAILY_LOSS in engine.kill_switch


def test_consecutive_losses_trip_the_kill_switch(scored_snapshot):
    engine = RiskEngine()
    engine.session.roll_to(NOW.date(), 100_000.0)
    for _ in range(3):
        engine.session.record_trade(-50.0, 99_850.0)
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(equity=99_850.0),
    )
    assert not decision.approved
    assert KillSwitchReason.CONSECUTIVE_LOSSES in engine.kill_switch


def test_drawdown_trips_the_kill_switch(scored_snapshot):
    engine = RiskEngine()
    engine.session.peak_equity = 100_000.0
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(equity=90_000.0),
    )
    assert not decision.approved
    assert KillSwitchReason.DRAWDOWN in engine.kill_switch


def test_slippage_drift_trips_the_kill_switch(scored_snapshot):
    engine = RiskEngine()
    for _ in range(10):
        engine.session.record_slippage(expected=2.0, realized=12.0)
    decision, _ = engine.evaluate(
        make_candidate(),
        snapshot=make_snapshot(scored_snapshot),
        forecast=make_forecast(),
        portfolio=PortfolioState(),
        broker=BrokerState(),
    )
    assert not decision.approved
    assert KillSwitchReason.SLIPPAGE in engine.kill_switch


def test_session_state_rolls_daily_but_keeps_the_high_water_mark():
    session = SessionState()
    session.roll_to(NOW.date(), 100_000.0)
    session.record_trade(-500.0, 99_500.0)
    assert session.trades_today == 1

    session.roll_to((NOW + timedelta(days=1)).date(), 99_500.0)
    assert session.trades_today == 0
    assert session.realized_pnl == 0.0
    assert session.peak_equity == 100_000.0  # drawdown measured from the peak
    assert session.drawdown_fraction(95_000.0) == pytest.approx(0.05)


# --------------------------------------------------------------------------
# Sizing (spec 28.4)
# --------------------------------------------------------------------------


def test_sizing_scales_with_quality():
    config = RiskConfig()
    high = size_position(
        make_candidate(),
        equity=100_000.0,
        confidence=0.9,
        stability=0.95,
        calibration_quality=0.95,
        config=config,
    )
    low = size_position(
        make_candidate(),
        equity=100_000.0,
        confidence=0.2,
        stability=0.5,
        calibration_quality=0.5,
        config=config,
    )
    assert high.adjusted_risk > low.adjusted_risk
    assert high.contracts >= low.contracts


def test_confidence_below_the_floor_sizes_nothing():
    """That range is vetoed upstream; the mapping must not disagree."""
    assert confidence_factor(0.05, 0.20) == 0.0
    assert confidence_factor(0.20, 0.20) == 0.0
    assert confidence_factor(float("nan"), 0.20) == 0.0


def test_confidence_just_above_the_floor_can_afford_a_contract():
    """The dead band: admitted by the veto, but unsizeable under raw confidence.

    Raw confidence double-counts the ``min_confidence`` veto — it scales size
    from zero across a range already excluded — so a forecast at 0.25 got 25%
    of base risk and could not buy one contract of the cheapest SPY vertical.
    """
    factor = confidence_factor(0.25, 0.20)
    assert factor >= MIN_CONFIDENCE_FACTOR
    # $100k equity, base fraction 0.005, other quality terms at 0.8.
    budget = 100_000.0 * 0.005 * factor * (0.8**3)
    assert budget > 100.0  # clears a typical $73-95 vertical
    assert 100_000.0 * 0.005 * 0.25 * (0.8**3) < 73.0  # what it was before


def test_confidence_mapping_is_monotone_and_bounded():
    floor = 0.20
    values = [confidence_factor(c / 100.0, floor) for c in range(21, 101)]
    assert all(a <= b for a, b in zip(values[:-1], values[1:], strict=True))
    assert values[-1] == pytest.approx(1.0)
    assert all(MIN_CONFIDENCE_FACTOR <= v <= 1.0 for v in values)


def test_full_confidence_is_unchanged_by_the_mapping():
    """The fix must not inflate size where confidence was already high."""
    assert confidence_factor(1.0, 0.20) == pytest.approx(1.0)


def test_the_risk_budget_and_sizing_agree_on_confidence():
    """A filter that scaled confidence differently would discard sizeable trades."""
    config = RiskConfig()
    engine = RiskEngine(config=config)
    candidate = make_candidate()
    budget = engine.risk_budget(
        broker=BrokerState(equity=100_000.0),
        portfolio=PortfolioState(),
        confidence=0.30,
        stability=0.8,
        calibration_quality=0.8,
        liquidity=candidate.liquidity.score,
    )
    sized = size_position(
        candidate,
        equity=100_000.0,
        confidence=0.30,
        stability=0.8,
        calibration_quality=0.8,
        config=config,
    )
    # The budget is the pre-filter for exactly this sizing call, so anything it
    # admits must survive sizing.
    assert budget == pytest.approx(sized.adjusted_risk)


def test_sizing_respects_the_hard_contract_cap():
    config = RiskConfig()
    result = size_position(
        make_candidate(max_loss=1.0),
        equity=10_000_000.0,
        confidence=1.0,
        stability=1.0,
        calibration_quality=1.0,
        config=config,
    )
    assert result.contracts == config.trade.max_contracts
    assert result.binding_constraint == "max contracts cap"


def test_sizing_never_exceeds_the_dollar_cap():
    config = RiskConfig()
    result = size_position(
        make_candidate(max_loss=100.0),
        equity=10_000_000.0,
        confidence=1.0,
        stability=1.0,
        calibration_quality=1.0,
        config=config,
    )
    assert result.total_risk <= config.trade.max_dollar_risk + 1e-9


def test_sizing_refuses_undefined_risk():
    result = size_position(
        make_candidate(max_loss=float("inf")),
        equity=100_000.0,
        confidence=1.0,
        stability=1.0,
        calibration_quality=1.0,
        config=RiskConfig(),
    )
    assert result.contracts == 0
    assert "undefined risk" in result.binding_constraint


def test_sizing_is_blind_to_realized_pnl():
    """Spec 28.4 forbids increasing size after a loss.

    Enforced structurally: nothing in the sizing signature can carry P&L.
    """
    import inspect

    parameters = set(inspect.signature(size_position).parameters)
    assert not parameters & {"pnl", "realized_pnl", "session", "losses", "equity_change"}


def test_max_loss_is_immutable_after_entry():
    """Spec 28.4: the maximum permitted loss cannot be altered after entry."""
    from spy_der.domain.execution import Position

    position = Position(
        position_id="P1",
        opened_at=NOW,
        family=StrategyFamily.BULL_CALL_DEBIT_SPREAD,
        strategy=Strategy(
            StrategyFamily.BULL_CALL_DEBIT_SPREAD,
            (leg(1, 550, CALL, 4.0), leg(-1, 555, CALL, 2.0)),
        ),
        contracts=2,
        entry_price=2.0,
        max_loss_per_contract=200.0,
        max_profit_per_contract=300.0,
        exit_plan=ExitPlan(0.6, 0.7, 180.0),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        position.max_loss_per_contract = 50.0  # type: ignore[misc]
    assert position.total_max_loss == pytest.approx(400.0)

    marked = position.marked(3.0, NOW)
    assert marked.max_loss_per_contract == 200.0
    assert marked.unrealized_pnl == pytest.approx(1.0 * CONTRACT_MULTIPLIER * 2)
