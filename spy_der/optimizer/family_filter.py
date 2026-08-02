"""Market-state to strategy-family filtering (spec section 24).

Filtering has two jobs. It prevents meaningless comparisons — ranking a long
straddle against an iron condor in a dead pin tells you nothing useful — and it
cuts the candidate search, which is the dominant cost in the decision loop.

Transition and Unstable map to the empty set. That is spec 11.10 and 11.11: in
those states the answer is no trade, and returning no eligible families is how
that becomes structural rather than advisory.
"""

from __future__ import annotations

from spy_der.domain.enums import MarketState, StrategyFamily

#: Eligible families per state, transcribed from the spec 24 table.
STATE_FAMILIES: dict[MarketState, frozenset[StrategyFamily]] = {
    MarketState.STRONG_PIN: frozenset(
        {
            StrategyFamily.IRON_BUTTERFLY,
            StrategyFamily.CALL_BUTTERFLY,
            StrategyFamily.PUT_BUTTERFLY,
            StrategyFamily.BROKEN_WING_CALL_BUTTERFLY,
            StrategyFamily.BROKEN_WING_PUT_BUTTERFLY,
            StrategyFamily.IRON_CONDOR,
            StrategyFamily.DOUBLE_CALENDAR,
        }
    ),
    MarketState.BROAD_RANGE: frozenset(
        {
            StrategyFamily.IRON_CONDOR,
            StrategyFamily.BROKEN_WING_CONDOR,
            StrategyFamily.ASYMMETRIC_CONDOR,
            StrategyFamily.DOUBLE_CALENDAR,
            StrategyFamily.BULL_PUT_CREDIT_SPREAD,
            StrategyFamily.BEAR_CALL_CREDIT_SPREAD,
            StrategyFamily.CALL_BUTTERFLY,
            StrategyFamily.PUT_BUTTERFLY,
        }
    ),
    MarketState.BULL_GRIND: frozenset(
        {
            StrategyFamily.BULL_CALL_DEBIT_SPREAD,
            StrategyFamily.BULL_PUT_CREDIT_SPREAD,
            StrategyFamily.CALL_BUTTERFLY,
            StrategyFamily.BROKEN_WING_CALL_BUTTERFLY,
            StrategyFamily.BULLISH_CALENDAR,
            StrategyFamily.BULLISH_DIAGONAL,
        }
    ),
    MarketState.BEAR_GRIND: frozenset(
        {
            StrategyFamily.BEAR_PUT_DEBIT_SPREAD,
            StrategyFamily.BEAR_CALL_CREDIT_SPREAD,
            StrategyFamily.PUT_BUTTERFLY,
            StrategyFamily.BROKEN_WING_PUT_BUTTERFLY,
            StrategyFamily.BEARISH_CALENDAR,
            StrategyFamily.BEARISH_DIAGONAL,
        }
    ),
    MarketState.BULL_BREAKOUT: frozenset(
        {
            StrategyFamily.BULL_CALL_DEBIT_SPREAD,
            StrategyFamily.CALL_BACKSPREAD,
            StrategyFamily.LONG_STRADDLE,
            StrategyFamily.LONG_STRANGLE,
            StrategyFamily.BROKEN_WING_CALL_BUTTERFLY,
        }
    ),
    MarketState.BEAR_BREAKDOWN: frozenset(
        {
            StrategyFamily.BEAR_PUT_DEBIT_SPREAD,
            StrategyFamily.PUT_BACKSPREAD,
            StrategyFamily.LONG_STRADDLE,
            StrategyFamily.LONG_STRANGLE,
            StrategyFamily.BROKEN_WING_PUT_BUTTERFLY,
        }
    ),
    MarketState.VOL_EXPANSION: frozenset(
        {
            StrategyFamily.LONG_STRADDLE,
            StrategyFamily.LONG_STRANGLE,
            StrategyFamily.CALL_BACKSPREAD,
            StrategyFamily.PUT_BACKSPREAD,
        }
    ),
    MarketState.VOL_CONTRACTION: frozenset(
        {
            StrategyFamily.BULL_PUT_CREDIT_SPREAD,
            StrategyFamily.BEAR_CALL_CREDIT_SPREAD,
            StrategyFamily.IRON_CONDOR,
            StrategyFamily.IRON_BUTTERFLY,
            StrategyFamily.BROKEN_WING_CALL_BUTTERFLY,
            StrategyFamily.BROKEN_WING_PUT_BUTTERFLY,
            StrategyFamily.BROKEN_WING_CONDOR,
        }
    ),
    MarketState.DIRECTIONAL_RANGE: frozenset(
        {
            StrategyFamily.ASYMMETRIC_CONDOR,
            StrategyFamily.BROKEN_WING_CONDOR,
            StrategyFamily.BULL_PUT_CREDIT_SPREAD,
            StrategyFamily.BEAR_CALL_CREDIT_SPREAD,
            StrategyFamily.BROKEN_WING_CALL_BUTTERFLY,
            StrategyFamily.BROKEN_WING_PUT_BUTTERFLY,
        }
    ),
    MarketState.TRANSITION: frozenset(),
    MarketState.UNSTABLE: frozenset(),
}


def eligible_families(
    state_probabilities: dict[MarketState, float], *, threshold: float = 0.15
) -> frozenset[StrategyFamily]:
    """Union of families eligible under every sufficiently probable state.

    Taking the union across states above ``threshold`` rather than filtering on
    the modal state alone matters when the distribution is genuinely split: a
    55/45 pin-versus-range read should let both families compete instead of
    silently discarding half the evidence.

    A no-trade state above the threshold contributes nothing, so an uncertain
    read that includes Transition naturally shrinks the eligible set.
    """
    families: set[StrategyFamily] = set()
    for state, probability in state_probabilities.items():
        if probability >= threshold:
            families |= STATE_FAMILIES.get(state, frozenset())
    return frozenset(families)


def families_for_state(state: MarketState) -> frozenset[StrategyFamily]:
    return STATE_FAMILIES.get(state, frozenset())


def is_no_trade_state(state: MarketState) -> bool:
    return not STATE_FAMILIES.get(state, frozenset())
