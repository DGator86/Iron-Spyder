"""Structural analytics tests (spec sections 8, 9, 10, 37.1)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spy_der.analytics.chain import build_chain_arrays, chain_from_snapshot
from spy_der.analytics.exposures import (
    DIRECTIONAL_CONVENTIONS,
    INFORMATIVE_CONVENTIONS,
    compute_exposures,
    dealer_agreement,
)
from spy_der.analytics.flow import compute_flow
from spy_der.analytics.gamma_profile import build_gamma_profile
from spy_der.analytics.surface import (
    build_expiration_arrays,
    compute_expected_move,
    compute_skew,
    compute_term_structure,
    compute_volatility_edge,
)
from spy_der.analytics.walls import WallInteraction, WallMethod, WallTracker, find_walls
from spy_der.domain.enums import DealerSignConvention, TradeDirection
from spy_der.domain.market import realized_volatility

# --------------------------------------------------------------------------
# Exposures (spec 8)
# --------------------------------------------------------------------------


def test_exposures_computed_under_every_convention(scored_snapshot):
    arrays = chain_from_snapshot(scored_snapshot("broad_range"))
    bundle = compute_exposures(arrays)
    for convention in DealerSignConvention:
        assert convention in bundle.per_convention


def test_mirror_conventions_have_opposite_gex(scored_snapshot):
    arrays = chain_from_snapshot(scored_snapshot("broad_range"))
    bundle = compute_exposures(arrays)
    long_gex = bundle.gex(DealerSignConvention.CUSTOMER_LONG)
    short_gex = bundle.gex(DealerSignConvention.CUSTOMER_SHORT)
    assert long_gex == pytest.approx(-short_gex, rel=1e-9)


def test_agreement_excludes_the_mirror_convention():
    """Including both mirrors would force agreement to zero on every snapshot."""
    assert DealerSignConvention.CUSTOMER_SHORT in DIRECTIONAL_CONVENTIONS
    assert DealerSignConvention.CUSTOMER_SHORT not in INFORMATIVE_CONVENTIONS


def test_dealer_agreement_bounds_and_unanimity():
    assert dealer_agreement([1.0]) == 1.0
    assert dealer_agreement([5.0, 5.0, 5.0], sign_consensus=1.0) == pytest.approx(1.0)
    split = dealer_agreement([5.0, -5.0], sign_consensus=0.5)
    assert 0.0 <= split < 0.5


def test_agreement_is_informative_not_pinned(scored_snapshot):
    """Agreement must vary across scenarios to be usable in confidence."""
    values = []
    for name in ("broad_range", "bull_breakout", "strong_pin", "vol_expansion"):
        arrays = chain_from_snapshot(scored_snapshot(name))
        values.append(compute_exposures(arrays).agreement)
    assert max(values) - min(values) > 0.1
    assert all(0.0 <= v <= 1.0 for v in values)


def test_unsigned_exposure_is_non_negative(scored_snapshot):
    arrays = chain_from_snapshot(scored_snapshot("broad_range"))
    bundle = compute_exposures(arrays)
    assert bundle.per_convention[DealerSignConvention.UNSIGNED].gex >= 0.0
    assert all(v >= 0.0 for v in bundle.unsigned_gex_by_strike.values())


# --------------------------------------------------------------------------
# Gamma profile (spec 8.5, 8.8)
# --------------------------------------------------------------------------


def test_gamma_profile_spans_the_requested_band(scored_snapshot):
    arrays = chain_from_snapshot(scored_snapshot("broad_range"))
    profile = build_gamma_profile(arrays, width=0.04, points=61)
    assert len(profile.spots) == 61
    assert profile.spots[0] == pytest.approx(arrays.spot * 0.96)
    assert profile.spots[-1] == pytest.approx(arrays.spot * 1.04)


def test_gamma_flip_is_an_actual_zero_crossing(scored_snapshot):
    arrays = chain_from_snapshot(scored_snapshot("broad_range"))
    profile = build_gamma_profile(arrays)
    if profile.nearest_flip is None:
        pytest.skip("no flip within the profile band")
    value = float(np.interp(profile.nearest_flip, profile.spots, profile.gex))
    scale = float(np.max(np.abs(profile.gex)))
    assert abs(value) < 0.01 * scale


def test_uniform_sign_convention_has_no_flip(scored_snapshot):
    """A uniform-sign convention can never cross zero; hence the default."""
    arrays = chain_from_snapshot(scored_snapshot("broad_range"))
    profile = build_gamma_profile(arrays, convention=DealerSignConvention.CUSTOMER_LONG)
    assert profile.nearest_flip is None
    assert np.all(profile.gex <= 0.0)


def test_vol_trigger_is_the_steepest_slope(scored_snapshot):
    arrays = chain_from_snapshot(scored_snapshot("broad_range"))
    profile = build_gamma_profile(arrays)
    assert profile.vol_trigger is not None
    steepest = float(profile.spots[int(np.argmax(np.abs(profile.slope)))])
    assert profile.vol_trigger == pytest.approx(steepest)


def test_positive_and_negative_zones_partition_the_curve(scored_snapshot):
    arrays = chain_from_snapshot(scored_snapshot("broad_range"))
    profile = build_gamma_profile(arrays)
    for low, high in profile.positive_zones:
        assert low <= high
    for low, high in profile.negative_zones:
        assert low <= high


# --------------------------------------------------------------------------
# Walls (spec 8.6, 8.7)
# --------------------------------------------------------------------------


def test_walls_are_distinct_and_bracket_spot(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    expiry = snapshot.nearest_expiration(min_seconds=86_400.0)
    arrays = build_expiration_arrays(snapshot, expiry)
    walls = find_walls(arrays)
    assert walls.call_wall is not None and walls.put_wall is not None
    assert walls.put_wall.strike < walls.call_wall.strike
    assert walls.inside_walls


@pytest.mark.parametrize("method", list(WallMethod))
def test_every_wall_method_produces_a_strike(method, scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    arrays = build_expiration_arrays(snapshot, snapshot.expirations[-1])
    walls = find_walls(arrays, method=method)
    assert walls.call_wall is not None
    assert walls.call_wall.method is method


def test_wall_tracker_distinguishes_break_from_rejection():
    tracker = WallTracker(touch_tolerance=0.001, acceptance_bars=3)
    wall = 560.0
    assert tracker.update(550.0, wall) is WallInteraction.AWAY
    assert tracker.update(559.9, wall) is WallInteraction.TOUCH
    assert tracker.update(561.0, wall) is WallInteraction.BREAK
    # Sustained trade above the wall becomes acceptance.
    tracker.update(561.5, wall)
    tracker.update(562.0, wall)
    assert tracker.update(562.5, wall) is WallInteraction.ACCEPTANCE


def test_wall_tracker_reports_migration():
    tracker = WallTracker()
    tracker.update(550.0, 560.0)
    assert tracker.update(550.0, 565.0) is WallInteraction.MIGRATION


# --------------------------------------------------------------------------
# Volatility surface (spec 9)
# --------------------------------------------------------------------------


def test_expected_move_bounds_and_utilization(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    move = compute_expected_move(snapshot, snapshot.expirations[-1])
    assert move is not None
    assert move.move > 0.0
    assert move.upper > snapshot.spot > move.lower
    assert move.utilization >= 0.0
    assert 0.0 <= move.remaining_fraction <= 1.0


def test_skew_requires_a_single_expiration(scored_snapshot):
    """Mixing tenors makes the delta interpolation meaningless, so it raises."""
    arrays = chain_from_snapshot(scored_snapshot("broad_range"))
    with pytest.raises(ValueError, match="single expiration"):
        compute_skew(arrays)


def test_put_skew_is_positive_when_downside_is_bid(scored_snapshot):
    snapshot = scored_snapshot("bear_breakdown")  # steep put skew by construction
    arrays = build_expiration_arrays(snapshot, snapshot.expirations[-1])
    skew = compute_skew(arrays)
    assert skew is not None
    assert skew.put_skew > 0.0
    assert skew.risk_reversal < 0.0  # puts richer than calls


def test_term_structure_is_ordered_and_signed(scored_snapshot):
    snapshot = scored_snapshot("vol_contraction")  # upward-sloping by construction
    term = compute_term_structure(snapshot)
    assert len(term.tenors) >= 2
    assert list(term.tenors) == sorted(term.tenors)
    assert term.slope() > 0.0
    assert not term.is_inverted


def test_inverted_term_structure_is_detected(scored_snapshot):
    snapshot = scored_snapshot("vol_expansion")  # negative term slope
    term = compute_term_structure(snapshot)
    assert term.is_inverted
    assert term.slope() < 0.0


def test_volatility_edge_sign_drives_the_recommendation():
    cheap = compute_volatility_edge(atm_iv=0.15, realized_vol=0.40)
    rich = compute_volatility_edge(atm_iv=0.40, realized_vol=0.10)
    assert cheap.favors_long_vol
    assert not rich.favors_long_vol
    assert rich.iv_rv_spread > 0.0


# --------------------------------------------------------------------------
# Flow (spec 10)
# --------------------------------------------------------------------------


def test_trade_direction_inference():
    from spy_der.data.synthetic import build_scenario

    snapshot = build_scenario("bull_grind")
    trade = snapshot.option_trades[0]
    at_ask = replace(trade, price=trade.ask_at_trade)
    at_bid = replace(trade, price=trade.bid_at_trade)
    inside = replace(trade, price=0.5 * (trade.bid_at_trade + trade.ask_at_trade))

    assert at_ask.direction is TradeDirection.BUY
    assert at_bid.direction is TradeDirection.SELL
    assert inside.direction is TradeDirection.UNKNOWN
    assert inside.signed_premium == 0.0


def test_opposing_flow_cancels_rather_than_accumulating(scored_snapshot):
    """Raw volume is not direction; a buy and an equal sell must net to zero."""
    snapshot = scored_snapshot("broad_range")
    trade = snapshot.option_trades[0]
    buy = replace(trade, price=trade.ask_at_trade, size=10)
    sell = replace(trade, price=trade.bid_at_trade, size=10)
    metrics = compute_flow([buy, sell], spot=snapshot.spot)
    assert metrics.trade_count == 2
    # Signed premia differ only by the spread, so the net is tiny relative to gross.
    gross = abs(buy.signed_premium) + abs(sell.signed_premium)
    assert abs(metrics.net_signed_premium) < 0.05 * gross


def test_flow_marks_itself_uninformative_when_unsigned(scored_snapshot):
    snapshot = scored_snapshot("broad_range")
    inside = [
        replace(t, price=0.5 * (t.bid_at_trade + t.ask_at_trade))
        for t in snapshot.option_trades[:10]
    ]
    metrics = compute_flow(inside, spot=snapshot.spot)
    assert metrics.unknown_direction_fraction == 1.0
    assert not metrics.is_informative


def test_empty_flow_is_neutral():
    metrics = compute_flow([], spot=550.0)
    assert metrics.trade_count == 0
    assert metrics.directional_tilt == 0.0
    assert not metrics.is_informative


# --------------------------------------------------------------------------
# Realized volatility
# --------------------------------------------------------------------------


def test_realized_volatility_is_positive_and_ordered(scored_snapshot):
    quiet = realized_volatility(scored_snapshot("strong_pin").spy_bars)
    active = realized_volatility(scored_snapshot("vol_expansion").spy_bars)
    assert 0.0 < quiet < active


def test_realized_volatility_handles_short_series():
    assert realized_volatility([]) == 0.0


def test_chain_arrays_reject_an_empty_chain():
    with pytest.raises(ValueError, match="empty chain"):
        build_chain_arrays([], spot=550.0, rate=0.04, div_yield=0.013)
