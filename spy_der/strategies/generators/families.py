"""Chain-driven generators for the allowed strategy universe (spec section 21).

Each generator yields several parameterizations of its family — different
widths, anchors, and short-strike probabilities — so that spec 2.5 holds: no
strategy, and no particular parameterization of one, is preferred by default.
The optimizer picks the winner.

Every generator returns structures only. Validation happens in
:mod:`spy_der.strategies.validation` and admission happens in
:mod:`spy_der.strategies.registry`; a generator is never trusted to have
produced a defined-risk structure just because it intended to.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from spy_der.domain.enums import OptionRight, StrategyFamily
from spy_der.domain.strategy import Strategy
from spy_der.strategies.generators.context import GenerationContext

CALL = OptionRight.CALL
PUT = OptionRight.PUT

#: Spread widths expressed in strike steps.
NARROW_WIDTHS: tuple[int, ...] = (1, 2, 3, 5)
WING_WIDTHS: tuple[int, ...] = (2, 3, 5)
#: Short-strike finish probabilities for credit structures.
SHORT_PROBABILITIES: tuple[float, ...] = (0.10, 0.16, 0.25)

Generator = Callable[[GenerationContext], Iterator[Strategy]]
_REGISTRY: dict[StrategyFamily, Generator] = {}


def generator(family: StrategyFamily) -> Callable[[Generator], Generator]:
    def register(func: Generator) -> Generator:
        _REGISTRY[family] = func
        return func

    return register


def generate(family: StrategyFamily, ctx: GenerationContext) -> list[Strategy]:
    """Generate candidate structures for one family."""
    func = _REGISTRY.get(family)
    if func is None:
        return []
    return [s for s in func(ctx) if s is not None]


def registered_families() -> frozenset[StrategyFamily]:
    return frozenset(_REGISTRY)


# --------------------------------------------------------------------------
# Vertical spreads
# --------------------------------------------------------------------------


@generator(StrategyFamily.BULL_CALL_DEBIT_SPREAD)
def _bull_call_debit(ctx: GenerationContext) -> Iterator[Strategy]:
    step = ctx.strike_step
    for anchor in _anchors(ctx, bullish=True):
        for width in NARROW_WIDTHS:
            legs = ctx.legs((anchor, CALL, 1), (anchor + width * step, CALL, -1))
            if legs:
                yield Strategy(StrategyFamily.BULL_CALL_DEBIT_SPREAD, legs)


@generator(StrategyFamily.BEAR_PUT_DEBIT_SPREAD)
def _bear_put_debit(ctx: GenerationContext) -> Iterator[Strategy]:
    step = ctx.strike_step
    for anchor in _anchors(ctx, bullish=False):
        for width in NARROW_WIDTHS:
            legs = ctx.legs((anchor, PUT, 1), (anchor - width * step, PUT, -1))
            if legs:
                yield Strategy(StrategyFamily.BEAR_PUT_DEBIT_SPREAD, legs)


@generator(StrategyFamily.BULL_PUT_CREDIT_SPREAD)
def _bull_put_credit(ctx: GenerationContext) -> Iterator[Strategy]:
    step = ctx.strike_step
    for probability in SHORT_PROBABILITIES:
        short = ctx.strike_at_finish_probability(probability, PUT)
        if short is None or short >= ctx.spot:
            continue
        for width in NARROW_WIDTHS:
            legs = ctx.legs((short, PUT, -1), (short - width * step, PUT, 1))
            if legs:
                yield Strategy(StrategyFamily.BULL_PUT_CREDIT_SPREAD, legs)


@generator(StrategyFamily.BEAR_CALL_CREDIT_SPREAD)
def _bear_call_credit(ctx: GenerationContext) -> Iterator[Strategy]:
    step = ctx.strike_step
    for probability in SHORT_PROBABILITIES:
        short = ctx.strike_at_finish_probability(probability, CALL)
        if short is None or short <= ctx.spot:
            continue
        for width in NARROW_WIDTHS:
            legs = ctx.legs((short, CALL, -1), (short + width * step, CALL, 1))
            if legs:
                yield Strategy(StrategyFamily.BEAR_CALL_CREDIT_SPREAD, legs)


# --------------------------------------------------------------------------
# Butterflies
# --------------------------------------------------------------------------


@generator(StrategyFamily.CALL_BUTTERFLY)
def _call_butterfly(ctx: GenerationContext) -> Iterator[Strategy]:
    yield from _butterfly(ctx, CALL, StrategyFamily.CALL_BUTTERFLY)


@generator(StrategyFamily.PUT_BUTTERFLY)
def _put_butterfly(ctx: GenerationContext) -> Iterator[Strategy]:
    yield from _butterfly(ctx, PUT, StrategyFamily.PUT_BUTTERFLY)


def _butterfly(
    ctx: GenerationContext, right: OptionRight, family: StrategyFamily
) -> Iterator[Strategy]:
    step = ctx.strike_step
    for centre in _butterfly_centres(ctx):
        for wing in WING_WIDTHS:
            legs = ctx.legs(
                (centre - wing * step, right, 1),
                (centre, right, -2),
                (centre + wing * step, right, 1),
            )
            if legs:
                yield Strategy(family, legs)


@generator(StrategyFamily.BROKEN_WING_CALL_BUTTERFLY)
def _broken_wing_call(ctx: GenerationContext) -> Iterator[Strategy]:
    yield from _broken_wing(ctx, CALL, StrategyFamily.BROKEN_WING_CALL_BUTTERFLY)


@generator(StrategyFamily.BROKEN_WING_PUT_BUTTERFLY)
def _broken_wing_put(ctx: GenerationContext) -> Iterator[Strategy]:
    yield from _broken_wing(ctx, PUT, StrategyFamily.BROKEN_WING_PUT_BUTTERFLY)


def _broken_wing(
    ctx: GenerationContext, right: OptionRight, family: StrategyFamily
) -> Iterator[Strategy]:
    """Butterfly with an extended far wing.

    The wing is extended *away* from the body on the side the structure is
    willing to be wrong about, which is what lets the structure be entered for a
    credit while keeping loss capped at the wing differential.
    """
    step = ctx.strike_step
    for centre in _butterfly_centres(ctx):
        for near_wing in (1, 2, 3):
            for extra in (1, 2):
                far_wing = near_wing + extra
                if right is CALL:
                    strikes = (
                        centre - near_wing * step,
                        centre,
                        centre + far_wing * step,
                    )
                else:
                    strikes = (
                        centre + near_wing * step,
                        centre,
                        centre - far_wing * step,
                    )
                legs = ctx.legs(
                    (strikes[0], right, 1), (strikes[1], right, -2), (strikes[2], right, 1)
                )
                if legs:
                    yield Strategy(family, legs)


# --------------------------------------------------------------------------
# Condors and iron structures
# --------------------------------------------------------------------------


@generator(StrategyFamily.IRON_CONDOR)
def _iron_condor(ctx: GenerationContext) -> Iterator[Strategy]:
    step = ctx.strike_step
    for probability in SHORT_PROBABILITIES:
        short_put = ctx.strike_at_finish_probability(probability, PUT)
        short_call = ctx.strike_at_finish_probability(probability, CALL)
        if short_put is None or short_call is None or short_put >= short_call:
            continue
        for wing in WING_WIDTHS:
            legs = ctx.legs(
                (short_put - wing * step, PUT, 1),
                (short_put, PUT, -1),
                (short_call, CALL, -1),
                (short_call + wing * step, CALL, 1),
            )
            if legs:
                yield Strategy(StrategyFamily.IRON_CONDOR, legs)


@generator(StrategyFamily.ASYMMETRIC_CONDOR)
def _asymmetric_condor(ctx: GenerationContext) -> Iterator[Strategy]:
    """Condor with unequal short-strike probabilities for a directional range."""
    step = ctx.strike_step
    for put_probability, call_probability in ((0.10, 0.28), (0.28, 0.10), (0.16, 0.34)):
        short_put = ctx.strike_at_finish_probability(put_probability, PUT)
        short_call = ctx.strike_at_finish_probability(call_probability, CALL)
        if short_put is None or short_call is None or short_put >= short_call:
            continue
        for wing in (2, 3):
            legs = ctx.legs(
                (short_put - wing * step, PUT, 1),
                (short_put, PUT, -1),
                (short_call, CALL, -1),
                (short_call + wing * step, CALL, 1),
            )
            if legs:
                yield Strategy(StrategyFamily.ASYMMETRIC_CONDOR, legs)


@generator(StrategyFamily.BROKEN_WING_CONDOR)
def _broken_wing_condor(ctx: GenerationContext) -> Iterator[Strategy]:
    """Condor with unequal wing widths, skewing where the loss is taken."""
    step = ctx.strike_step
    for probability in (0.16, 0.25):
        short_put = ctx.strike_at_finish_probability(probability, PUT)
        short_call = ctx.strike_at_finish_probability(probability, CALL)
        if short_put is None or short_call is None or short_put >= short_call:
            continue
        for put_wing, call_wing in ((2, 5), (5, 2)):
            legs = ctx.legs(
                (short_put - put_wing * step, PUT, 1),
                (short_put, PUT, -1),
                (short_call, CALL, -1),
                (short_call + call_wing * step, CALL, 1),
            )
            if legs:
                yield Strategy(StrategyFamily.BROKEN_WING_CONDOR, legs)


@generator(StrategyFamily.IRON_BUTTERFLY)
def _iron_butterfly(ctx: GenerationContext) -> Iterator[Strategy]:
    step = ctx.strike_step
    for centre in _butterfly_centres(ctx):
        for wing in WING_WIDTHS:
            legs = ctx.legs(
                (centre - wing * step, PUT, 1),
                (centre, PUT, -1),
                (centre, CALL, -1),
                (centre + wing * step, CALL, 1),
            )
            if legs:
                yield Strategy(StrategyFamily.IRON_BUTTERFLY, legs)


# --------------------------------------------------------------------------
# Long volatility
# --------------------------------------------------------------------------


@generator(StrategyFamily.LONG_STRADDLE)
def _long_straddle(ctx: GenerationContext) -> Iterator[Strategy]:
    for centre in {ctx.atm_strike, ctx.snap(ctx.spot)}:
        legs = ctx.legs((centre, CALL, 1), (centre, PUT, 1))
        if legs:
            yield Strategy(StrategyFamily.LONG_STRADDLE, legs)


@generator(StrategyFamily.LONG_STRANGLE)
def _long_strangle(ctx: GenerationContext) -> Iterator[Strategy]:
    step = ctx.strike_step
    for width in (2, 3, 5):
        call_strike = ctx.snap(ctx.atm_strike + width * step)
        put_strike = ctx.snap(ctx.atm_strike - width * step)
        if call_strike <= put_strike:
            continue
        legs = ctx.legs((call_strike, CALL, 1), (put_strike, PUT, 1))
        if legs:
            yield Strategy(StrategyFamily.LONG_STRANGLE, legs)


@generator(StrategyFamily.CALL_BACKSPREAD)
def _call_backspread(ctx: GenerationContext) -> Iterator[Strategy]:
    """Short one nearer call against two further calls.

    Defined risk because the long quantity exceeds the short quantity, so the
    upside slope is positive and the worst case sits at the long strike.
    """
    step = ctx.strike_step
    for short_offset in (0, 1):
        short_strike = ctx.snap(ctx.atm_strike + short_offset * step)
        for width in (1, 2, 3):
            long_strike = ctx.snap(short_strike + width * step)
            if long_strike <= short_strike:
                continue
            legs = ctx.legs((short_strike, CALL, -1), (long_strike, CALL, 2))
            if legs:
                yield Strategy(StrategyFamily.CALL_BACKSPREAD, legs)


@generator(StrategyFamily.PUT_BACKSPREAD)
def _put_backspread(ctx: GenerationContext) -> Iterator[Strategy]:
    step = ctx.strike_step
    for short_offset in (0, 1):
        short_strike = ctx.snap(ctx.atm_strike - short_offset * step)
        for width in (1, 2, 3):
            long_strike = ctx.snap(short_strike - width * step)
            if long_strike >= short_strike:
                continue
            legs = ctx.legs((short_strike, PUT, -1), (long_strike, PUT, 2))
            if legs:
                yield Strategy(StrategyFamily.PUT_BACKSPREAD, legs)


# --------------------------------------------------------------------------
# Calendarized structures
# --------------------------------------------------------------------------


@generator(StrategyFamily.BULLISH_CALENDAR)
def _bullish_calendar(ctx: GenerationContext) -> Iterator[Strategy]:
    yield from _calendar(ctx, CALL, StrategyFamily.BULLISH_CALENDAR, offsets=(0, 1, 2))


@generator(StrategyFamily.BEARISH_CALENDAR)
def _bearish_calendar(ctx: GenerationContext) -> Iterator[Strategy]:
    yield from _calendar(ctx, PUT, StrategyFamily.BEARISH_CALENDAR, offsets=(0, -1, -2))


def _calendar(
    ctx: GenerationContext,
    right: OptionRight,
    family: StrategyFamily,
    *,
    offsets: tuple[int, ...],
) -> Iterator[Strategy]:
    """Short the near expiration, long the far one at the same strike.

    The long leg must be the far-dated one. The reverse — long near, short far —
    leaves an uncovered short after the near leg expires and is rejected by the
    suffix-coverage rule, which is the intended behaviour.
    """
    if ctx.far_expiration is None or ctx.far_expiration <= ctx.expiration:
        return
    step = ctx.strike_step
    for offset in offsets:
        strike = ctx.snap(ctx.atm_strike + offset * step)
        near = ctx.leg(strike, right, -1, expiration=ctx.expiration)
        far = ctx.leg(strike, right, 1, expiration=ctx.far_expiration)
        if near and far:
            yield Strategy(family, (near, far))


@generator(StrategyFamily.DOUBLE_CALENDAR)
def _double_calendar(ctx: GenerationContext) -> Iterator[Strategy]:
    if ctx.far_expiration is None or ctx.far_expiration <= ctx.expiration:
        return
    step = ctx.strike_step
    for width in (2, 3, 5):
        call_strike = ctx.snap(ctx.atm_strike + width * step)
        put_strike = ctx.snap(ctx.atm_strike - width * step)
        if call_strike <= put_strike:
            continue
        legs = (
            ctx.leg(call_strike, CALL, -1, expiration=ctx.expiration),
            ctx.leg(call_strike, CALL, 1, expiration=ctx.far_expiration),
            ctx.leg(put_strike, PUT, -1, expiration=ctx.expiration),
            ctx.leg(put_strike, PUT, 1, expiration=ctx.far_expiration),
        )
        if all(legs):
            yield Strategy(StrategyFamily.DOUBLE_CALENDAR, tuple(legs))  # type: ignore[arg-type]


@generator(StrategyFamily.BULLISH_DIAGONAL)
def _bullish_diagonal(ctx: GenerationContext) -> Iterator[Strategy]:
    """Long a lower-strike far call against a short higher-strike near call."""
    if ctx.far_expiration is None or ctx.far_expiration <= ctx.expiration:
        return
    step = ctx.strike_step
    for width in (1, 2, 3):
        long_strike = ctx.snap(ctx.atm_strike - step)
        short_strike = ctx.snap(long_strike + width * step)
        if short_strike <= long_strike:
            continue
        near = ctx.leg(short_strike, CALL, -1, expiration=ctx.expiration)
        far = ctx.leg(long_strike, CALL, 1, expiration=ctx.far_expiration)
        if near and far:
            yield Strategy(StrategyFamily.BULLISH_DIAGONAL, (near, far))


@generator(StrategyFamily.BEARISH_DIAGONAL)
def _bearish_diagonal(ctx: GenerationContext) -> Iterator[Strategy]:
    """Long a higher-strike far put against a short lower-strike near put."""
    if ctx.far_expiration is None or ctx.far_expiration <= ctx.expiration:
        return
    step = ctx.strike_step
    for width in (1, 2, 3):
        long_strike = ctx.snap(ctx.atm_strike + step)
        short_strike = ctx.snap(long_strike - width * step)
        if short_strike >= long_strike:
            continue
        near = ctx.leg(short_strike, PUT, -1, expiration=ctx.expiration)
        far = ctx.leg(long_strike, PUT, 1, expiration=ctx.far_expiration)
        if near and far:
            yield Strategy(StrategyFamily.BEARISH_DIAGONAL, (near, far))


# --------------------------------------------------------------------------
# Anchor selection
# --------------------------------------------------------------------------


def _anchors(ctx: GenerationContext, *, bullish: bool) -> tuple[float, ...]:
    """Long-strike anchors for a directional debit spread.

    Spans at-the-money through modestly in-the-money, plus the median forecast
    price, so the optimizer can trade off cost against probability instead of
    being handed one fixed moneyness.
    """
    step = ctx.strike_step
    atm = ctx.atm_strike
    offsets = (-1, 0, 1) if bullish else (1, 0, -1)
    anchors = {ctx.snap(atm + o * step) for o in offsets}

    horizon = ctx.forecast.horizons.get(ctx.horizon)
    if horizon is not None and horizon.price_quantiles:
        anchors.add(ctx.snap(horizon.quantile(0.5)))
    return tuple(sorted(anchors))


def _butterfly_centres(ctx: GenerationContext) -> tuple[float, ...]:
    """Body strikes for butterflies: the money, the pin, and the forecast median."""
    centres = {ctx.atm_strike}
    if ctx.features.pin_strike is not None:
        centres.add(ctx.snap(ctx.features.pin_strike))

    horizon = ctx.forecast.horizons.get(ctx.horizon)
    if horizon is not None and horizon.price_quantiles:
        centres.add(ctx.snap(horizon.quantile(0.5)))
    return tuple(sorted(centres))
