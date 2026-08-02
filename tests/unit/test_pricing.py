"""Pricing and Greek tests (spec section 37.1)."""

from __future__ import annotations

import numpy as np
import pytest

from spy_der.analytics import pricing
from spy_der.domain.enums import OptionRight

CALL = OptionRight.CALL
PUT = OptionRight.PUT

# (spot, strike, tau, rate, div_yield, vol)
CASES = [
    (550.0, 545.0, 0.25, 0.04, 0.013, 0.18),
    (550.0, 550.0, 1.0 / 365.0, 0.04, 0.013, 0.12),
    (430.0, 500.0, 0.5, 0.05, 0.02, 0.35),
    (600.0, 520.0, 0.08, 0.03, 0.00, 0.22),
]


@pytest.mark.parametrize("spot,strike,tau,rate,q,vol", CASES)
def test_put_call_parity(spot, strike, tau, rate, q, vol):
    """``C - P = S e^{-qT} - K e^{-rT}`` (spec 37.1)."""
    call = float(pricing.price(spot, strike, tau, rate, q, vol, CALL))
    put = float(pricing.price(spot, strike, tau, rate, q, vol, PUT))
    assert pricing.put_call_parity_gap(call, put, spot, strike, tau, rate, q) == pytest.approx(
        0.0, abs=1e-9
    )


@pytest.mark.parametrize("spot,strike,tau,rate,q,vol", CASES)
def test_delta_matches_finite_difference(spot, strike, tau, rate, q, vol):
    h = 1e-4
    for right in (CALL, PUT):
        up = float(pricing.price(spot + h, strike, tau, rate, q, vol, right))
        down = float(pricing.price(spot - h, strike, tau, rate, q, vol, right))
        analytic = float(pricing.delta(spot, strike, tau, rate, q, vol, right))
        assert analytic == pytest.approx((up - down) / (2 * h), rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("spot,strike,tau,rate,q,vol", CASES)
def test_gamma_matches_derivative_of_delta(spot, strike, tau, rate, q, vol):
    """Gamma is checked as ``d(delta)/dS``.

    A second difference of price is dominated by cancellation error at any step
    small enough to be accurate, so the well-conditioned first difference of
    delta is the right numerical check.
    """
    h = 1e-4
    up = float(pricing.delta(spot + h, strike, tau, rate, q, vol, CALL))
    down = float(pricing.delta(spot - h, strike, tau, rate, q, vol, CALL))
    analytic = float(pricing.gamma(spot, strike, tau, rate, q, vol))
    assert analytic == pytest.approx((up - down) / (2 * h), rel=1e-5, abs=1e-10)


@pytest.mark.parametrize("spot,strike,tau,rate,q,vol", CASES)
def test_vega_and_vomma_match_finite_difference(spot, strike, tau, rate, q, vol):
    h = 1e-6
    up = float(pricing.price(spot, strike, tau, rate, q, vol + h, CALL))
    down = float(pricing.price(spot, strike, tau, rate, q, vol - h, CALL))
    assert float(pricing.vega(spot, strike, tau, rate, q, vol)) == pytest.approx(
        (up - down) / (2 * h), rel=1e-5
    )

    vega_up = float(pricing.vega(spot, strike, tau, rate, q, vol + h))
    vega_down = float(pricing.vega(spot, strike, tau, rate, q, vol - h))
    assert float(pricing.vomma(spot, strike, tau, rate, q, vol)) == pytest.approx(
        (vega_up - vega_down) / (2 * h), rel=1e-4, abs=1e-6
    )


@pytest.mark.parametrize("spot,strike,tau,rate,q,vol", CASES)
def test_theta_charm_and_color_use_calendar_time(spot, strike, tau, rate, q, vol):
    """``d/dt = -d/dtau`` for theta, charm, and color (spec 7.2)."""
    h = 1e-6
    for right in (CALL, PUT):
        up = float(pricing.price(spot, strike, tau + h, rate, q, vol, right))
        down = float(pricing.price(spot, strike, tau - h, rate, q, vol, right))
        assert float(pricing.theta(spot, strike, tau, rate, q, vol, right)) == pytest.approx(
            -(up - down) / (2 * h), rel=1e-4, abs=1e-6
        )

        delta_up = float(pricing.delta(spot, strike, tau + h, rate, q, vol, right))
        delta_down = float(pricing.delta(spot, strike, tau - h, rate, q, vol, right))
        assert float(pricing.charm(spot, strike, tau, rate, q, vol, right)) == pytest.approx(
            -(delta_up - delta_down) / (2 * h), rel=1e-4, abs=1e-6
        )

    gamma_up = float(pricing.gamma(spot, strike, tau + h, rate, q, vol))
    gamma_down = float(pricing.gamma(spot, strike, tau - h, rate, q, vol))
    assert float(pricing.color(spot, strike, tau, rate, q, vol)) == pytest.approx(
        -(gamma_up - gamma_down) / (2 * h), rel=1e-3, abs=1e-6
    )


@pytest.mark.parametrize("spot,strike,tau,rate,q,vol", CASES)
def test_vanna_and_speed_match_finite_difference(spot, strike, tau, rate, q, vol):
    hv, hs = 1e-6, 1e-3
    delta_up = float(pricing.delta(spot, strike, tau, rate, q, vol + hv, CALL))
    delta_down = float(pricing.delta(spot, strike, tau, rate, q, vol - hv, CALL))
    assert float(pricing.vanna(spot, strike, tau, rate, q, vol)) == pytest.approx(
        (delta_up - delta_down) / (2 * hv), rel=1e-4, abs=1e-8
    )

    gamma_up = float(pricing.gamma(spot + hs, strike, tau, rate, q, vol))
    gamma_down = float(pricing.gamma(spot - hs, strike, tau, rate, q, vol))
    assert float(pricing.speed(spot, strike, tau, rate, q, vol)) == pytest.approx(
        (gamma_up - gamma_down) / (2 * hs), rel=1e-4, abs=1e-10
    )


def test_gamma_is_right_independent():
    """Put and call gamma coincide; the wall logic depends on this."""
    call_gamma = float(pricing.gamma(550, 552, 0.05, 0.04, 0.013, 0.16))
    assert call_gamma > 0.0
    # Gamma takes no right argument precisely because it cannot differ.
    assert "right" not in pricing.gamma.__code__.co_varnames


@pytest.mark.parametrize("spot,strike,tau,rate,q,vol", CASES)
def test_implied_volatility_round_trip(spot, strike, tau, rate, q, vol):
    for right in (CALL, PUT):
        price = float(pricing.price(spot, strike, tau, rate, q, vol, right))
        recovered = pricing.implied_volatility(price, spot, strike, tau, rate, q, right)
        assert recovered is not None
        assert recovered == pytest.approx(vol, abs=1e-6)


def test_implied_volatility_rejects_arbitrage_violating_price():
    """A price outside the no-arbitrage band returns ``None``, not a guess."""
    assert (
        pricing.implied_volatility(1_000.0, 550.0, 550.0, 0.25, 0.04, 0.013, CALL) is None
    )


def test_expired_options_equal_intrinsic():
    """At expiry the price is intrinsic and every Greek is zero (spec 37.2)."""
    for strike, expected in ((540.0, 10.0), (560.0, 0.0)):
        assert float(pricing.price(550.0, strike, 0.0, 0.04, 0.013, 0.2, CALL)) == pytest.approx(
            expected
        )
    assert float(pricing.gamma(550.0, 550.0, 0.0, 0.04, 0.013, 0.2)) == 0.0
    assert float(pricing.vega(550.0, 550.0, 0.0, 0.04, 0.013, 0.2)) == 0.0
    assert float(pricing.theta(550.0, 550.0, 0.0, 0.04, 0.013, 0.2, CALL)) == 0.0


def test_vectorized_matches_scalar():
    """Array and scalar paths must agree; the profile engine relies on it."""
    strikes = np.array([540.0, 550.0, 560.0])
    vector = pricing.price(550.0, strikes, 0.1, 0.04, 0.013, 0.18, CALL)
    scalars = [float(pricing.price(550.0, float(k), 0.1, 0.04, 0.013, 0.18, CALL)) for k in strikes]
    assert np.allclose(vector, scalars)


def test_dividend_yield_lowers_call_and_raises_put():
    """The ``q`` term must actually be wired in (spec 7.1)."""
    call_no_div = float(pricing.price(550, 550, 0.5, 0.04, 0.0, 0.2, CALL))
    call_div = float(pricing.price(550, 550, 0.5, 0.04, 0.03, 0.2, CALL))
    put_no_div = float(pricing.price(550, 550, 0.5, 0.04, 0.0, 0.2, PUT))
    put_div = float(pricing.price(550, 550, 0.5, 0.04, 0.03, 0.2, PUT))
    assert call_div < call_no_div
    assert put_div > put_no_div
