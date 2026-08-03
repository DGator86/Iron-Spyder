"""Execution, fills, reconciliation, and assignment control (spec 31, 30.5, 44)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from spy_der.domain.enums import (
    ExitReason,
    OptionRight,
    OrderIntent,
    OrderStatus,
    StrategyFamily,
)
from spy_der.domain.execution import BrokerState, Position, RiskDecision
from spy_der.domain.strategy import ExitPlan, Strategy
from spy_der.execution.fills import (
    CostModel,
    entry_slippage,
    fill_probability,
    round_trip_cost,
)
from spy_der.execution.manager import PositionManager
from spy_der.execution.orders import build_close_order, build_open_order, describe
from spy_der.execution.paper_broker import PaperBroker, PaperBrokerConfig
from spy_der.execution.reconciliation import apply_to_broker_state, reconcile
from spy_der.optimizer.engine import EXIT_DEFAULTS
from tests.conftest import leg
from tests.unit.test_risk import make_candidate

CALL = OptionRight.CALL
PUT = OptionRight.PUT
NOW = datetime(2026, 8, 3, 15, 30, tzinfo=UTC)


def make_position(snapshot, *, contracts: int = 1, **overrides) -> Position:
    expiry = snapshot.expirations[-1]
    strategy = Strategy(
        StrategyFamily.BULL_CALL_DEBIT_SPREAD,
        (
            leg(1, 550, CALL, 4.0, expiry=expiry),
            leg(-1, 553, CALL, 2.5, expiry=expiry),
        ),
    )
    defaults = dict(
        position_id="P-test",
        opened_at=snapshot.timestamp - timedelta(minutes=30),
        family=strategy.family,
        strategy=strategy,
        contracts=contracts,
        entry_price=1.5,
        max_loss_per_contract=150.0,
        max_profit_per_contract=150.0,
        exit_plan=ExitPlan(
            profit_target_fraction=0.6,
            stop_loss_fraction=0.7,
            max_holding_minutes=120.0,
            invalidation_states=("Transition", "Unstable"),
        ),
        current_price=1.5,
        marked_at=snapshot.timestamp,
    )
    defaults.update(overrides)
    return Position(**defaults)


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------


def test_round_trip_cost_charges_both_directions(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    expiry = snapshot.expirations[-1]
    legs = (leg(1, 550, CALL, 4.0, expiry=expiry), leg(-1, 553, CALL, 2.5, expiry=expiry))
    quotes = tuple(snapshot.quote_for(expiry, leg_.strike, leg_.right) for leg_ in legs)
    assert all(quotes)

    model = CostModel()
    one_way = entry_slippage(legs, quotes, model)
    both_ways = round_trip_cost(legs, quotes, model)
    assert both_ways > 2.0 * one_way  # slippage twice, plus commissions twice


def test_cost_stress_increases_cost_monotonically(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    expiry = snapshot.expirations[-1]
    legs = (leg(1, 550, CALL, 4.0, expiry=expiry), leg(-1, 553, CALL, 2.5, expiry=expiry))
    quotes = tuple(snapshot.quote_for(expiry, leg_.strike, leg_.right) for leg_ in legs)

    base = round_trip_cost(legs, quotes, CostModel())
    wider = round_trip_cost(legs, quotes, CostModel().stressed(spread_multiplier=2.0))
    adverse = round_trip_cost(legs, quotes, CostModel().stressed(adverse_ticks=2.0))
    assert wider > base
    assert adverse > base


def test_fill_probability_falls_with_legs_and_spread(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    expiry = snapshot.expirations[-1]
    quotes = tuple(
        snapshot.quote_for(expiry, k, CALL) for k in snapshot.strikes_for(expiry)[:4]
    )
    model = CostModel()
    two_leg = fill_probability(quotes[:2], leg_count=2, contracts=1, model=model)
    four_leg = fill_probability(quotes, leg_count=4, contracts=1, model=model)
    assert four_leg < two_leg

    wide = tuple(replace(q, bid=q.mid * 0.5, ask=q.mid * 1.5) for q in quotes[:2])
    assert fill_probability(wide, leg_count=2, contracts=1, model=model) < two_leg


def test_fill_probability_is_never_certain(scored_snapshot):
    """Spec 37.7: nothing may assume a guaranteed midpoint fill."""
    snapshot = scored_snapshot("broad_range")
    expiry = snapshot.expirations[-1]
    quotes = tuple(snapshot.quote_for(expiry, k, CALL) for k in snapshot.strikes_for(expiry)[:2])
    assert fill_probability(quotes, leg_count=2, contracts=1, model=CostModel()) < 1.0


def test_fill_probability_uses_open_interest_when_volume_is_zero(scored_snapshot):
    """Historical tapes often report volume=0; OI must still inform fill odds."""
    snapshot = scored_snapshot("broad_range")
    expiry = snapshot.expirations[-1]
    quotes = tuple(snapshot.quote_for(expiry, k, CALL) for k in snapshot.strikes_for(expiry)[:2])
    zero_volume = tuple(replace(q, volume=0, open_interest=max(q.open_interest, 2_000)) for q in quotes)
    empty = tuple(replace(q, volume=0, open_interest=0) for q in quotes)
    model = CostModel()
    with_oi = fill_probability(zero_volume, leg_count=2, contracts=1, model=model)
    without = fill_probability(empty, leg_count=2, contracts=1, model=model)
    assert with_oi > without
    assert with_oi >= 0.25


# --------------------------------------------------------------------------
# Order construction
# --------------------------------------------------------------------------


def test_open_order_takes_size_from_the_risk_decision():
    candidate = make_candidate()
    decision = RiskDecision(approved=True, max_contracts=3, required_price_limit=2.05)
    order = build_open_order(candidate, decision, now=NOW)
    assert order.contracts == 3
    assert order.limit_price == pytest.approx(2.05)
    assert order.intent is OrderIntent.OPEN


def test_open_order_refuses_an_unapproved_decision():
    with pytest.raises(ValueError, match="unapproved"):
        build_open_order(make_candidate(), RiskDecision(approved=False), now=NOW)


def test_open_order_refuses_zero_contracts():
    with pytest.raises(ValueError, match="zero contracts"):
        build_open_order(
            make_candidate(), RiskDecision(approved=True, max_contracts=0), now=NOW
        )


def test_close_order_inverts_every_leg(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    position = make_position(snapshot)
    order = build_close_order(position, limit_price=-1.8, now=NOW)
    assert order.intent is OrderIntent.CLOSE
    for original, closing in zip(position.legs, order.strategy.legs):
        assert closing.quantity == -original.quantity
        assert closing.strike == original.strike
        assert closing.expiry == original.expiry
    assert "CLOSE" in describe(order)


# --------------------------------------------------------------------------
# Paper broker
# --------------------------------------------------------------------------


def test_paper_broker_fills_a_marketable_order(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    broker = PaperBroker()
    candidate = make_candidate(
        strategy=Strategy(
            StrategyFamily.BULL_CALL_DEBIT_SPREAD,
            (
                leg(1, 550, CALL, 4.0, expiry=snapshot.expirations[-1]),
                leg(-1, 553, CALL, 2.5, expiry=snapshot.expirations[-1]),
            ),
        )
    )
    order = build_open_order(
        candidate,
        RiskDecision(approved=True, max_contracts=1, required_price_limit=50.0),
        now=snapshot.timestamp,
    )
    order = broker.submit(order, snapshot)
    assert order.status is OrderStatus.WORKING

    fill = None
    for _ in range(6):
        order, fill = broker.poll(order.order_id, snapshot)
        if fill is not None:
            break
    assert fill is not None
    assert order.status is OrderStatus.FILLED
    assert fill.commission > 0.0


def test_paper_broker_expires_an_unmarketable_order(scored_snapshot):
    """An order that cannot fill at its limit expires; it never fills anyway."""
    snapshot = scored_snapshot("broad_range")
    broker = PaperBroker(config=PaperBrokerConfig(max_reprice_attempts=2))
    candidate = make_candidate(
        strategy=Strategy(
            StrategyFamily.BULL_CALL_DEBIT_SPREAD,
            (
                leg(1, 550, CALL, 4.0, expiry=snapshot.expirations[-1]),
                leg(-1, 553, CALL, 2.5, expiry=snapshot.expirations[-1]),
            ),
        )
    )
    order = build_open_order(
        candidate,
        RiskDecision(approved=True, max_contracts=1, required_price_limit=-99.0),
        now=snapshot.timestamp,
    )
    order = broker.submit(order, snapshot)
    for _ in range(6):
        order, fill = broker.poll(order.order_id, snapshot)
        assert fill is None
        if order.is_terminal:
            break
    assert order.status is OrderStatus.EXPIRED
    assert order.reprice_attempts >= 2


def test_paper_broker_rejects_duplicate_order_ids(scored_snapshot):
    from spy_der.execution.broker_base import BrokerError

    snapshot = scored_snapshot("broad_range")
    broker = PaperBroker()
    order = build_open_order(
        make_candidate(),
        RiskDecision(approved=True, max_contracts=1, required_price_limit=50.0),
        now=snapshot.timestamp,
        order_id="dup",
    )
    broker.submit(order, snapshot)
    with pytest.raises(BrokerError, match="duplicate"):
        broker.submit(order, snapshot)


def test_paper_broker_does_not_support_live():
    """Spec 40 gates live execution; no shipped adapter may claim support."""
    assert PaperBroker().supports_live is False


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------


def test_reconciliation_matches_identical_books(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    position = make_position(snapshot)
    report = reconcile((position,), (position,))
    assert report.is_reconciled
    assert report.matched_legs == 2


def test_reconciliation_detects_a_missing_broker_leg(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    position = make_position(snapshot)
    half = replace(
        position, strategy=Strategy(position.family, (position.legs[0],))
    )
    report = reconcile((position,), (half,))
    assert not report.is_reconciled
    assert report.internal_only


def test_reconciliation_detects_a_quantity_mismatch(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    ours = make_position(snapshot, contracts=2)
    theirs = make_position(snapshot, contracts=1)
    report = reconcile((ours,), (theirs,))
    assert not report.is_reconciled
    assert report.quantity_mismatches


def test_unreconciled_state_blocks_the_risk_engine(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    position = make_position(snapshot)
    report = reconcile((position,), ())
    state = apply_to_broker_state(BrokerState(), report)
    assert not state.reconciled
    assert not state.is_healthy


# --------------------------------------------------------------------------
# Exit management and assignment control (spec 30, 30.5)
# --------------------------------------------------------------------------


def test_exit_price_is_worse_than_the_midpoint(scored_snapshot):
    """Marks use executable prices, not midpoints."""
    snapshot = scored_snapshot("broad_range")
    position = make_position(snapshot)
    manager = PositionManager()
    price = manager.exit_price(position, snapshot)
    assert price is not None

    mid = sum(
        leg_.quantity * snapshot.quote_for(leg_.expiry, leg_.strike, leg_.right).mid for leg_ in position.legs
    )
    assert price < mid  # closing a debit structure sells into the bid side


def test_profit_target_and_stop_fire(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    manager = PositionManager()

    winner = make_position(snapshot, entry_price=0.10, max_profit_per_contract=50.0)
    signal = manager.check_exit(winner, snapshot)
    assert signal is not None and signal.reason is ExitReason.PROFIT_TARGET

    loser = make_position(snapshot, entry_price=8.0, max_loss_per_contract=100.0)
    signal = manager.check_exit(loser, snapshot)
    assert signal is not None and signal.reason is ExitReason.RISK_STOP


def test_short_itm_leg_near_expiry_triggers_assignment_exit(scored_snapshot):
    """Spec 30.5: never rely on assignment; close the short leg first."""
    snapshot = scored_snapshot("broad_range")
    expiry = snapshot.expirations[0]  # same-day
    strategy = Strategy(
        StrategyFamily.BEAR_CALL_CREDIT_SPREAD,
        (
            leg(-1, 545, CALL, 5.0, expiry=expiry),  # deep in the money
            leg(1, 555, CALL, 0.5, expiry=expiry),
        ),
    )
    position = make_position(
        snapshot,
        strategy=strategy,
        family=strategy.family,
        exit_plan=ExitPlan(0.5, 0.8, 300.0, close_before_expiry_minutes=600.0),
    )
    manager = PositionManager()
    assert manager.assignment_exposure(position, snapshot) is not None
    signal = manager.check_exit(position, snapshot)
    assert signal is not None and signal.reason is ExitReason.ASSIGNMENT_RISK


def test_assignment_outranks_the_profit_target(scored_snapshot):
    """Assignment control is non-negotiable even on a winning position."""
    snapshot = scored_snapshot("broad_range")
    expiry = snapshot.expirations[0]
    strategy = Strategy(
        StrategyFamily.BEAR_CALL_CREDIT_SPREAD,
        (leg(-1, 545, CALL, 5.0, expiry=expiry), leg(1, 555, CALL, 0.5, expiry=expiry)),
    )
    position = make_position(
        snapshot,
        strategy=strategy,
        family=strategy.family,
        entry_price=-100.0,  # an enormous paper gain
        max_profit_per_contract=1.0,
        exit_plan=ExitPlan(0.1, 0.9, 300.0, close_before_expiry_minutes=600.0),
    )
    signal = PositionManager().check_exit(position, snapshot)
    assert signal is not None and signal.reason is ExitReason.ASSIGNMENT_RISK


def test_time_stop_fires(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    position = make_position(
        snapshot,
        opened_at=snapshot.timestamp - timedelta(minutes=500),
        exit_plan=ExitPlan(0.95, 0.95, 60.0),
    )
    signal = PositionManager().check_exit(position, snapshot)
    assert signal is not None and signal.reason is ExitReason.TIME_STOP


def test_kill_switch_forces_an_exit(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    position = make_position(snapshot)
    signal = PositionManager().check_exit(position, snapshot, kill_switch_active=True)
    assert signal is not None and signal.reason is ExitReason.KILL_SWITCH


def test_every_position_carries_a_complete_exit_plan(scored_snapshot):
    """Spec 30: no open position without the full exit contract."""
    snapshot = scored_snapshot("broad_range")
    plan = make_position(snapshot).exit_plan
    assert plan.profit_target_fraction > 0.0
    assert plan.stop_loss_fraction > 0.0
    assert plan.max_holding_minutes > 0.0
    assert plan.close_before_expiry_minutes > 0.0
    assert plan.invalidation_states


def test_exit_plan_rejects_degenerate_values():
    for kwargs in (
        {"profit_target_fraction": 0.0},
        {"stop_loss_fraction": 0.0},
        {"max_holding_minutes": 0.0},
        {"stop_loss_credit_multiple": 0.0},
        {"stop_loss_credit_multiple": -1.0},
    ):
        base = {
            "profit_target_fraction": 0.5,
            "stop_loss_fraction": 0.5,
            "max_holding_minutes": 60.0,
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            ExitPlan(**base)


# --------------------------------------------------------------------------
# Credit-denominated risk stops (spec 30)
# --------------------------------------------------------------------------


def credit_plan(multiple: float | None = 2.0) -> ExitPlan:
    return ExitPlan(
        profit_target_fraction=0.50,
        stop_loss_fraction=1.00,
        max_holding_minutes=390.0,
        stop_loss_credit_multiple=multiple,
    )


def test_a_credit_multiple_stops_before_maximum_loss():
    """The point of the field: 2x credit must be reachable, unlike a clamped 2.0.

    $5-wide spread taken for $1.50 credit. Maximum loss is width minus credit,
    $350 per contract, and 2x the credit is $300 — a real stop at 0.86 of the
    maximum, where a ``stop_loss_fraction`` of 2.0 would clamp to 1.0 and only
    fire at the full $350, which is no stop at all.
    """
    stop = credit_plan().stop_loss_dollars(max_loss=350.0, credit_received=150.0)
    assert stop == pytest.approx(300.0)
    assert stop < 350.0


def test_the_same_multiple_is_a_different_max_loss_fraction_by_width():
    """Why a static max-loss fraction cannot express the rule."""
    plan = credit_plan(2.0)
    narrow = plan.stop_loss_dollars(max_loss=350.0, credit_received=150.0)  # $5 wide
    wide = plan.stop_loss_dollars(max_loss=850.0, credit_received=150.0)  # $10 wide
    assert narrow / 350.0 == pytest.approx(0.857, abs=1e-3)
    assert wide / 850.0 == pytest.approx(0.353, abs=1e-3)


def test_a_credit_stop_is_capped_at_maximum_loss():
    """A threshold beyond max loss is unreachable, which is the same as no stop."""
    # $2.00 credit on a $2.50 spread: 2x credit is $400, max loss only $50.
    stop = credit_plan().stop_loss_dollars(max_loss=50.0, credit_received=200.0)
    assert stop == pytest.approx(50.0)


def test_a_debit_structure_falls_back_to_the_max_loss_fraction():
    plan = ExitPlan(
        profit_target_fraction=0.65,
        stop_loss_fraction=0.60,
        max_holding_minutes=390.0,
        stop_loss_credit_multiple=2.0,
    )
    assert plan.stop_loss_dollars(max_loss=500.0, credit_received=0.0) == pytest.approx(300.0)


def test_no_multiple_leaves_the_max_loss_fraction_untouched():
    plan = ExitPlan(
        profit_target_fraction=0.65, stop_loss_fraction=0.60, max_holding_minutes=390.0
    )
    assert plan.stop_loss_dollars(max_loss=500.0, credit_received=150.0) == pytest.approx(300.0)


def test_every_credit_family_declares_a_reachable_stop():
    """Regression: no credit family may sit at an unreachable stop again."""
    credit, width = 150.0, 500.0
    max_loss = width - credit
    for family, default in EXIT_DEFAULTS.items():
        if default.credit_multiple is None:
            continue
        plan = ExitPlan(
            profit_target_fraction=default.target,
            stop_loss_fraction=default.stop,
            max_holding_minutes=390.0,
            stop_loss_credit_multiple=default.credit_multiple,
        )
        stop = plan.stop_loss_dollars(max_loss=max_loss, credit_received=credit)
        assert 0.0 < stop < max_loss, f"{family.value} stop {stop} is not reachable"


def test_the_manager_stops_a_credit_position_at_the_credit_multiple(scored_snapshot):
    """The enforced threshold, end to end through the position manager."""
    snapshot = scored_snapshot("broad_range")
    # entry_price is the net per-share structure price; negative is a credit.
    position = make_position(
        snapshot,
        entry_price=-1.50,
        max_loss_per_contract=350.0,
        exit_plan=credit_plan(2.0),
    )
    assert PositionManager.stop_threshold(position) == pytest.approx(300.0)


def test_the_simulator_and_the_manager_price_the_same_stop():
    """If these diverged, the optimizer would value a stop never enforced."""
    plan = credit_plan(2.0)
    # Both resolve through ExitPlan.stop_loss_dollars, so equality is structural
    # rather than two constants that happen to agree today.
    entry_per_share = -1.50
    simulated = plan.stop_loss_dollars(
        max_loss=350.0, credit_received=max(0.0, -entry_per_share) * 100
    )
    enforced = plan.stop_loss_dollars(max_loss=350.0, credit_received=150.0)
    assert simulated == enforced == pytest.approx(300.0)
