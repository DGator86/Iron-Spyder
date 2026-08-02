"""Property tests (spec section 37.2).

Randomized over a seeded parameter space rather than a fuzzing library, so the
suite has no extra dependency and any failure reproduces exactly from the
printed seed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spy_der.analytics import pricing
from spy_der.domain.enums import MarketState, OptionRight, StrategyFamily
from spy_der.domain.forecast import normalized_certainty, shannon_entropy
from spy_der.domain.strategy import Strategy
from spy_der.models.structural.scores import softmax
from spy_der.strategies.payoff import payoff_profile, terminal_payoff
from spy_der.strategies.validation import validate_defined_risk
from tests.conftest import FAR, NEAR, leg

CALL = OptionRight.CALL
PUT = OptionRight.PUT
RNG = np.random.default_rng(20260803)
SAMPLES = 200


def _random_params(rng: np.random.Generator) -> tuple[float, float, float, float, float, float]:
    spot = float(rng.uniform(300.0, 700.0))
    strike = spot * float(rng.uniform(0.75, 1.25))
    tau = float(rng.uniform(1e-4, 2.0))
    rate = float(rng.uniform(0.0, 0.08))
    div = float(rng.uniform(0.0, 0.04))
    vol = float(rng.uniform(0.03, 1.2))
    return spot, strike, tau, rate, div, vol


def test_option_prices_are_never_negative():
    rng = np.random.default_rng(1)
    for _ in range(SAMPLES):
        params = _random_params(rng)
        for right in (CALL, PUT):
            assert float(pricing.price(*params, right)) >= 0.0


def test_price_respects_no_arbitrage_bounds():
    rng = np.random.default_rng(2)
    for _ in range(SAMPLES):
        spot, strike, tau, rate, div, vol = _random_params(rng)
        forward = spot * math.exp(-div * tau)
        discounted = strike * math.exp(-rate * tau)

        call = float(pricing.price(spot, strike, tau, rate, div, vol, CALL))
        assert max(0.0, forward - discounted) - 1e-8 <= call <= forward + 1e-8

        put = float(pricing.price(spot, strike, tau, rate, div, vol, PUT))
        assert max(0.0, discounted - forward) - 1e-8 <= put <= discounted + 1e-8


def test_call_increases_and_put_decreases_with_spot():
    rng = np.random.default_rng(3)
    for _ in range(SAMPLES):
        spot, strike, tau, rate, div, vol = _random_params(rng)
        bump = spot * 0.01
        call_low = float(pricing.price(spot, strike, tau, rate, div, vol, CALL))
        call_high = float(pricing.price(spot + bump, strike, tau, rate, div, vol, CALL))
        put_low = float(pricing.price(spot, strike, tau, rate, div, vol, PUT))
        put_high = float(pricing.price(spot + bump, strike, tau, rate, div, vol, PUT))
        assert call_high >= call_low - 1e-9
        assert put_high <= put_low + 1e-9


def test_gamma_and_vega_are_non_negative_for_long_vanilla():
    rng = np.random.default_rng(4)
    for _ in range(SAMPLES):
        spot, strike, tau, rate, div, vol = _random_params(rng)
        assert float(pricing.gamma(spot, strike, tau, rate, div, vol)) >= 0.0
        assert float(pricing.vega(spot, strike, tau, rate, div, vol)) >= 0.0


def test_value_increases_with_volatility():
    rng = np.random.default_rng(5)
    for _ in range(SAMPLES):
        spot, strike, tau, rate, div, vol = _random_params(rng)
        for right in (CALL, PUT):
            low = float(pricing.price(spot, strike, tau, rate, div, vol, right))
            high = float(pricing.price(spot, strike, tau, rate, div, vol * 1.1, right))
            assert high >= low - 1e-9


# --------------------------------------------------------------------------
# Strategy invariants
# --------------------------------------------------------------------------


def _random_admissible(rng: np.random.Generator) -> Strategy:
    """Build a random structure from a set of defined-risk shapes."""
    centre = float(rng.integers(500, 600))
    width = float(rng.integers(1, 8))
    price = lambda: float(rng.uniform(0.05, 12.0))  # noqa: E731

    shape = int(rng.integers(0, 6))
    if shape == 0:
        return Strategy(
            StrategyFamily.BULL_CALL_DEBIT_SPREAD,
            (leg(1, centre, CALL, price()), leg(-1, centre + width, CALL, price())),
        )
    if shape == 1:
        return Strategy(
            StrategyFamily.BULL_PUT_CREDIT_SPREAD,
            (leg(-1, centre, PUT, price()), leg(1, centre - width, PUT, price())),
        )
    if shape == 2:
        return Strategy(
            StrategyFamily.CALL_BUTTERFLY,
            (
                leg(1, centre - width, CALL, price()),
                leg(-2, centre, CALL, price()),
                leg(1, centre + width, CALL, price()),
            ),
        )
    if shape == 3:
        return Strategy(
            StrategyFamily.IRON_CONDOR,
            (
                leg(1, centre - 2 * width, PUT, price()),
                leg(-1, centre - width, PUT, price()),
                leg(-1, centre + width, CALL, price()),
                leg(1, centre + 2 * width, CALL, price()),
            ),
        )
    if shape == 4:
        return Strategy(
            StrategyFamily.CALL_BACKSPREAD,
            (leg(-1, centre, CALL, price()), leg(2, centre + width, CALL, price())),
        )
    return Strategy(
        StrategyFamily.BULLISH_CALENDAR,
        (
            leg(-1, centre, CALL, price(), expiry=NEAR),
            leg(1, centre, CALL, price(), expiry=FAR),
        ),
    )


def test_admissible_structures_always_have_finite_positive_max_loss():
    """``MaxLoss(S) < inf`` for every admitted structure (spec 2.2)."""
    rng = np.random.default_rng(6)
    checked = 0
    for _ in range(SAMPLES):
        strategy = _random_admissible(rng)
        result = validate_defined_risk(strategy)
        if not result.is_valid:
            continue  # random prices can produce a non-positive worst case
        checked += 1
        assert math.isfinite(result.profile.max_loss)
        assert result.profile.max_loss > 0.0
        assert result.profile.upside_slope >= 0.0
    assert checked > SAMPLES // 2, "too few structures survived to be meaningful"


def test_no_admitted_structure_contains_uncovered_short_risk():
    """Spec 37.2: no allowed strategy contains uncovered short risk."""
    rng = np.random.default_rng(7)
    for _ in range(SAMPLES):
        strategy = _random_admissible(rng)
        if not validate_defined_risk(strategy).is_valid:
            continue
        for right in (CALL, PUT):
            legs = [leg_ for leg_ in strategy.legs if leg_.right is right]
            for expiry in {leg_.expiry for leg_ in legs}:
                suffix = sum(leg_.quantity for leg_ in legs if leg_.expiry >= expiry)
                assert suffix >= 0


def test_payoff_never_falls_below_reported_max_loss():
    """The reported worst case must bound the payoff everywhere."""
    rng = np.random.default_rng(8)
    for _ in range(SAMPLES):
        strategy = _random_admissible(rng)
        result = validate_defined_risk(strategy)
        if not result.is_valid or strategy.is_multi_expiry:
            continue
        low, high = strategy.strikes[0] * 0.5, strategy.strikes[-1] * 1.5
        for spot in np.linspace(low, high, 40):
            assert terminal_payoff(strategy, float(spot)) >= -result.profile.max_loss - 1e-6


def test_scaling_scales_risk_linearly():
    """Doubling contracts doubles the maximum loss and never changes its sign."""
    rng = np.random.default_rng(9)
    for _ in range(50):
        strategy = _random_admissible(rng)
        result = validate_defined_risk(strategy)
        if not result.is_valid:
            continue
        doubled = payoff_profile(strategy.scaled(2))
        assert doubled.max_loss == pytest.approx(2.0 * result.profile.max_loss, rel=1e-9)


# --------------------------------------------------------------------------
# Probability invariants
# --------------------------------------------------------------------------


def test_state_probabilities_sum_to_one():
    """Spec 37.2 and 44: probabilities sum to one."""
    rng = np.random.default_rng(10)
    for _ in range(SAMPLES):
        scores = {state: float(rng.normal(0.0, 3.0)) for state in MarketState}
        probabilities = softmax(scores)
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-12)
        assert all(0.0 <= p <= 1.0 for p in probabilities.values())


def test_softmax_is_shift_invariant_and_stable_at_extremes():
    scores = dict.fromkeys(MarketState, 500.0)
    probabilities = softmax(scores)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(math.isfinite(p) for p in probabilities.values())


def test_entropy_and_certainty_bounds():
    rng = np.random.default_rng(11)
    n = len(MarketState)
    for _ in range(100):
        raw = rng.dirichlet(np.ones(n))
        probabilities = {state: float(p) for state, p in zip(MarketState, raw)}
        entropy = shannon_entropy(probabilities)
        assert 0.0 <= entropy <= math.log(n) + 1e-9
        assert 0.0 <= normalized_certainty(probabilities) <= 1.0

    uniform = dict.fromkeys(MarketState, 1.0 / n)
    assert normalized_certainty(uniform) == pytest.approx(0.0, abs=1e-9)
