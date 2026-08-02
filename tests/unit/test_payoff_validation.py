"""Payoff geometry and the defined-risk proof (spec sections 37.1, 37.2, 44)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from spy_der.domain.enums import OptionRight, StrategyFamily
from spy_der.domain.strategy import OptionLeg, Strategy
from spy_der.strategies.payoff import (
    breakevens,
    payoff_profile,
    tail_slopes,
    terminal_payoff,
)
from spy_der.strategies.registry import StrategyRegistry
from spy_der.strategies.validation import (
    UndefinedRiskError,
    assert_defined_risk,
    coverage_violations,
    is_defined_risk,
    validate_defined_risk,
)
from tests.conftest import FAR, NEAR, leg

CALL = OptionRight.CALL
PUT = OptionRight.PUT


# --------------------------------------------------------------------------
# Exact payoff geometry
# --------------------------------------------------------------------------


def test_bull_call_spread_geometry(bull_call_spread):
    profile = payoff_profile(bull_call_spread)
    assert profile.max_loss == pytest.approx(200.0)  # the 2.00 debit
    assert profile.max_profit == pytest.approx(300.0)  # width 5 minus debit
    assert profile.breakevens == pytest.approx((552.0,))  # long strike + debit
    assert terminal_payoff(bull_call_spread, 545.0) == pytest.approx(-200.0)
    assert terminal_payoff(bull_call_spread, 560.0) == pytest.approx(300.0)


def test_iron_condor_geometry(iron_condor):
    profile = payoff_profile(iron_condor)
    assert profile.net_entry_cost == pytest.approx(-160.0)  # net credit
    assert profile.max_profit == pytest.approx(160.0)  # keep the credit
    assert profile.max_loss == pytest.approx(340.0)  # width 5 minus credit
    assert profile.breakevens == pytest.approx((538.4, 561.6))


def test_butterfly_geometry():
    fly = Strategy(
        StrategyFamily.CALL_BUTTERFLY,
        (leg(1, 545, CALL, 7.0), leg(-2, 550, CALL, 4.0), leg(1, 555, CALL, 2.0)),
    )
    profile = payoff_profile(fly)
    assert profile.max_loss == pytest.approx(100.0)  # the 1.00 debit
    assert profile.max_profit == pytest.approx(400.0)  # wing width minus debit
    assert profile.breakevens == pytest.approx((546.0, 554.0))


def test_long_straddle_has_unbounded_profit():
    straddle = Strategy(
        StrategyFamily.LONG_STRADDLE, (leg(1, 550, CALL, 4.0), leg(1, 550, PUT, 4.0))
    )
    profile = payoff_profile(straddle)
    assert profile.profit_is_unbounded
    assert math.isinf(profile.max_profit)
    assert profile.max_loss == pytest.approx(800.0)
    assert profile.breakevens == pytest.approx((542.0, 558.0))


def test_calendar_max_loss_is_the_debit(long_calendar):
    """A long calendar's worst case is losing the debit (spec 23).

    Measured at the near expiration with the far leg floored at intrinsic,
    which is a genuine lower bound on its value.
    """
    profile = payoff_profile(long_calendar)
    assert profile.max_loss == pytest.approx(200.0)
    assert profile.worst_case_expiry == NEAR
    # The terminal diagram is undefined across expirations.
    assert profile.profit_is_model_dependent
    assert math.isnan(profile.max_profit)
    assert profile.breakevens == ()


def test_breakevens_ignore_flat_zero_regions():
    """A zero-cost backspread breaks even across its lower range.

    Only the boundaries of the profitable region are breakevens; interior
    points of a flat zero-P&L stretch are not.
    """
    backspread = Strategy(
        StrategyFamily.CALL_BACKSPREAD, (leg(-1, 550, CALL, 4.0), leg(2, 555, CALL, 2.0))
    )
    assert backspread.net_entry_cost == pytest.approx(0.0)
    assert breakevens(backspread) == pytest.approx((550.0, 560.0))


def test_payoff_extrema_are_exact_not_sampled():
    """Extrema sit at kinks, so the reported worst case must be a strike."""
    fly = Strategy(
        StrategyFamily.BROKEN_WING_CALL_BUTTERFLY,
        (leg(1, 545, CALL, 7.0), leg(-2, 550, CALL, 4.0), leg(1, 558, CALL, 1.2)),
    )
    profile = payoff_profile(fly)
    assert profile.worst_case_spot in {0.0, *fly.strikes}


# --------------------------------------------------------------------------
# Defined-risk proof
# --------------------------------------------------------------------------

ADMISSIBLE = {
    "bull call spread": (
        StrategyFamily.BULL_CALL_DEBIT_SPREAD,
        (leg(1, 550, CALL, 4.0), leg(-1, 555, CALL, 2.0)),
    ),
    "bull put credit spread": (
        StrategyFamily.BULL_PUT_CREDIT_SPREAD,
        (leg(-1, 545, PUT, 3.0), leg(1, 540, PUT, 1.5)),
    ),
    "bear call credit spread": (
        StrategyFamily.BEAR_CALL_CREDIT_SPREAD,
        (leg(-1, 555, CALL, 3.0), leg(1, 560, CALL, 1.5)),
    ),
    "iron butterfly": (
        StrategyFamily.IRON_BUTTERFLY,
        (
            leg(1, 540, PUT, 1.0),
            leg(-1, 550, PUT, 4.0),
            leg(-1, 550, CALL, 4.0),
            leg(1, 560, CALL, 1.0),
        ),
    ),
    "call backspread": (
        StrategyFamily.CALL_BACKSPREAD,
        (leg(-1, 550, CALL, 4.0), leg(2, 555, CALL, 2.0)),
    ),
    "put backspread": (
        StrategyFamily.PUT_BACKSPREAD,
        (leg(-1, 550, PUT, 4.0), leg(2, 545, PUT, 2.0)),
    ),
    "long calendar": (
        StrategyFamily.BULLISH_CALENDAR,
        (leg(-1, 550, CALL, 3.0, expiry=NEAR), leg(1, 550, CALL, 5.0, expiry=FAR)),
    ),
    "bullish diagonal": (
        StrategyFamily.BULLISH_DIAGONAL,
        (leg(-1, 553, CALL, 2.0, expiry=NEAR), leg(1, 549, CALL, 6.0, expiry=FAR)),
    ),
    "broken wing butterfly": (
        StrategyFamily.BROKEN_WING_CALL_BUTTERFLY,
        (leg(1, 545, CALL, 7.0), leg(-2, 550, CALL, 4.0), leg(1, 558, CALL, 1.2)),
    ),
}

INADMISSIBLE = {
    "naked short call": (
        StrategyFamily.BEAR_CALL_CREDIT_SPREAD,
        (leg(-1, 555, CALL, 2.0),),
    ),
    "naked short put": (
        StrategyFamily.BULL_PUT_CREDIT_SPREAD,
        (leg(-1, 545, PUT, 3.0),),
    ),
    "2x1 call ratio write": (
        StrategyFamily.CALL_BACKSPREAD,
        (leg(-2, 555, CALL, 2.0), leg(1, 550, CALL, 4.0)),
    ),
    "2x1 put ratio write": (
        StrategyFamily.PUT_BACKSPREAD,
        (leg(-2, 545, PUT, 2.0), leg(1, 550, PUT, 4.0)),
    ),
    "short strangle": (
        StrategyFamily.LONG_STRANGLE,
        (leg(-1, 560, CALL, 1.5), leg(-1, 540, PUT, 1.5)),
    ),
    "short straddle": (
        StrategyFamily.LONG_STRADDLE,
        (leg(-1, 550, CALL, 4.0), leg(-1, 550, PUT, 4.0)),
    ),
    "short calendar": (
        StrategyFamily.BULLISH_CALENDAR,
        (leg(1, 550, CALL, 3.0, expiry=NEAR), leg(-1, 550, CALL, 5.0, expiry=FAR)),
    ),
    "3x2 call ratio": (
        StrategyFamily.CALL_BACKSPREAD,
        (leg(-3, 552, CALL, 3.0), leg(2, 556, CALL, 1.5)),
    ),
}


@pytest.mark.parametrize("name", sorted(ADMISSIBLE))
def test_admissible_structures_pass(name):
    family, legs = ADMISSIBLE[name]
    result = validate_defined_risk(Strategy(family, legs))
    assert result.is_valid, f"{name} was rejected: {result.violations}"
    assert result.profile is not None
    assert math.isfinite(result.profile.max_loss)
    assert result.profile.max_loss > 0.0


@pytest.mark.parametrize("name", sorted(INADMISSIBLE))
def test_inadmissible_structures_are_rejected(name):
    """No undefined-risk structure may pass validation (spec 22, 44)."""
    family, legs = INADMISSIBLE[name]
    result = validate_defined_risk(Strategy(family, legs))
    assert not result.is_valid, f"{name} was wrongly admitted"


def test_short_calendar_rejected_where_net_quantity_would_not_catch_it():
    """The short calendar nets to zero yet leaves a naked far-dated short.

    This is the case a plain net-quantity rule misses, and the reason coverage
    is checked as a suffix sum by expiration.
    """
    short_calendar = Strategy(
        StrategyFamily.BULLISH_CALENDAR,
        (leg(1, 550, CALL, 3.0, expiry=NEAR), leg(-1, 550, CALL, 5.0, expiry=FAR)),
    )
    net = sum(leg_.quantity for leg_ in short_calendar.legs)
    assert net == 0  # a naive net-quantity check would pass this
    violations = coverage_violations(short_calendar)
    assert violations and "uncovered short call" in violations[0]


def test_ratio_write_rejected_but_backspread_admitted():
    """Direction of the ratio is what matters, not that a ratio exists."""
    write = Strategy(
        StrategyFamily.CALL_BACKSPREAD,
        (leg(-2, 555, CALL, 2.0), leg(1, 550, CALL, 4.0)),
    )
    backspread = Strategy(
        StrategyFamily.CALL_BACKSPREAD,
        (leg(-1, 550, CALL, 4.0), leg(2, 555, CALL, 2.0)),
    )
    assert not is_defined_risk(write)
    assert is_defined_risk(backspread)


def test_assert_defined_risk_raises_on_naked_short():
    naked = Strategy(StrategyFamily.BEAR_CALL_CREDIT_SPREAD, (leg(-1, 555, CALL, 2.0),))
    with pytest.raises(UndefinedRiskError, match="uncovered short call"):
        assert_defined_risk(naked)


def test_registry_admits_only_defined_risk():
    """The registry is the admission chokepoint (spec 44)."""
    registry = StrategyRegistry()
    for family, legs in ADMISSIBLE.values():
        registry.submit(Strategy(family, legs))
    for family, legs in INADMISSIBLE.values():
        registry.submit(Strategy(family, legs))

    assert registry.admitted_count == len(ADMISSIBLE)
    assert registry.rejected_count == len(INADMISSIBLE)
    for entry in registry.admitted:
        assert math.isfinite(entry.profile.max_loss)
        assert entry.profile.upside_slope >= 0.0


def test_registry_require_raises_rather_than_recording():
    registry = StrategyRegistry()
    naked = Strategy(StrategyFamily.BEAR_CALL_CREDIT_SPREAD, (leg(-1, 555, CALL, 2.0),))
    with pytest.raises(UndefinedRiskError):
        registry.require(naked)


def test_multi_expiry_rejected_for_non_calendar_family():
    mixed = Strategy(
        StrategyFamily.BULL_CALL_DEBIT_SPREAD,
        (leg(1, 550, CALL, 4.0, expiry=NEAR), leg(-1, 555, CALL, 2.0, expiry=FAR)),
    )
    result = validate_defined_risk(mixed)
    assert not result.is_valid
    assert any("not a calendarized family" in v for v in result.violations)


def test_zero_quantity_leg_rejected_at_construction():
    with pytest.raises(ValueError, match="non-zero"):
        OptionLeg(
            expiry=datetime(2026, 8, 3, tzinfo=UTC),
            strike=550.0,
            right=CALL,
            quantity=0,
        )


def test_tail_slopes_identify_unbounded_upside():
    naked = Strategy(StrategyFamily.BEAR_CALL_CREDIT_SPREAD, (leg(-1, 555, CALL, 2.0),))
    upside, _ = tail_slopes(naked)
    assert upside < 0.0
    assert math.isinf(payoff_profile(naked).max_loss)
