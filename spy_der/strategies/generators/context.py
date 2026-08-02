"""Shared context and leg construction for strategy generators (spec section 25).

Strike selection is anchored to forecast quantiles, walls, and pin strikes
rather than to fixed deltas or fixed widths, which spec 25 explicitly requires.
A generator asks "which strike has a 15% chance of finishing in the money"
instead of "which strike is 15 delta"; those coincide only under a lognormal
model the forecast engine is not assumed to agree with.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

from spy_der.domain.enums import OptionRight
from spy_der.domain.forecast import ForecastBundle
from spy_der.domain.market import MarketSnapshot, OptionQuote, nearest_value
from spy_der.domain.strategy import OptionLeg
from spy_der.features.builder import MarketFeatures

#: Fraction of the half-spread crossed when estimating an entry price.
#: ``0.0`` is the midpoint and ``1.0`` is the far touch. The default assumes a
#: marketable limit that gets partial price improvement; spec 37.7 requires the
#: cost stress tests to sweep this, because a strategy that only works at the
#: midpoint is not production ready.
DEFAULT_AGGRESSIVENESS = 0.5


@dataclass(frozen=True)
class GenerationContext:
    """Everything a generator needs to build legs for one expiration."""

    snapshot: MarketSnapshot
    features: MarketFeatures
    forecast: ForecastBundle
    expiration: datetime
    far_expiration: datetime | None = None
    aggressiveness: float = DEFAULT_AGGRESSIVENESS
    horizon: str = "eod"

    @property
    def spot(self) -> float:
        return self.snapshot.spot

    @cached_property
    def quotes(self) -> dict[tuple[datetime, float, OptionRight], OptionQuote]:
        return {
            (q.expiration, q.strike, q.right): q
            for q in self.snapshot.option_chain
            if q.expiration in self.expirations
        }

    @cached_property
    def expirations(self) -> frozenset[datetime]:
        out = {self.expiration}
        if self.far_expiration is not None:
            out.add(self.far_expiration)
        return frozenset(out)

    @cached_property
    def strikes(self) -> tuple[float, ...]:
        """Strikes quoted on both rights at the near expiration."""
        calls = {
            q.strike
            for q in self.snapshot.chain_for(self.expiration)
            if q.right is OptionRight.CALL
        }
        puts = {
            q.strike
            for q in self.snapshot.chain_for(self.expiration)
            if q.right is OptionRight.PUT
        }
        return tuple(sorted(calls & puts))

    @cached_property
    def strike_step(self) -> float:
        strikes = self.strikes
        if len(strikes) < 2:
            return 1.0
        gaps = [b - a for a, b in zip(strikes, strikes[1:])]
        return min(gaps)

    @property
    def atm_strike(self) -> float:
        return nearest_value(self.strikes, self.spot)

    def quote(
        self, strike: float, right: OptionRight, *, expiration: datetime | None = None
    ) -> OptionQuote | None:
        return self.quotes.get((expiration or self.expiration, strike, right))

    def snap(self, target: float) -> float:
        """Nearest listed strike to ``target``."""
        return nearest_value(self.strikes, target)

    def strike_at_finish_probability(
        self, probability: float, right: OptionRight
    ) -> float | None:
        """Strike whose forecast probability of finishing in the money is ``probability``.

        For calls this is ``P(S_T >= K) = probability``; for puts it is
        ``P(S_T <= K) = probability``. Returns the nearest listed strike.
        """
        horizon = self.forecast.horizons.get(self.horizon)
        if horizon is None or not horizon.price_quantiles:
            return None
        tau = 1.0 - probability if right is OptionRight.CALL else probability
        target = horizon.quantile(min(max(tau, 0.001), 0.999))
        return self.snap(target)

    def leg(
        self,
        strike: float,
        right: OptionRight,
        quantity: int,
        *,
        expiration: datetime | None = None,
    ) -> OptionLeg | None:
        """Build a priced leg, or ``None`` when the contract is not quotable."""
        expiry = expiration or self.expiration
        quote = self.quote(strike, right, expiration=expiry)
        if quote is None or quote.mid <= 0.0 or quote.ask < quote.bid:
            return None

        price = execution_price(quote, buying=quantity > 0, aggressiveness=self.aggressiveness)
        if price <= 0.0:
            return None

        return OptionLeg(
            expiry=expiry,
            strike=strike,
            right=right,
            quantity=quantity,
            entry_price=price,
            entry_iv=quote.implied_volatility,
        )

    def legs(self, *specs: tuple[float, OptionRight, int]) -> tuple[OptionLeg, ...] | None:
        """Build several legs, returning ``None`` if any is unquotable.

        All-or-nothing on purpose: a structure missing one leg is not a cheaper
        version of itself, it is a different and possibly undefined-risk
        position.
        """
        built: list[OptionLeg] = []
        for strike, right, quantity in specs:
            leg = self.leg(strike, right, quantity)
            if leg is None:
                return None
            built.append(leg)
        return tuple(built)


def execution_price(quote: OptionQuote, *, buying: bool, aggressiveness: float) -> float:
    """Estimated entry price, moving from the midpoint toward the touch."""
    mid = quote.mid
    edge = max(0.0, quote.ask - mid) if buying else max(0.0, mid - quote.bid)
    price = mid + aggressiveness * edge if buying else mid - aggressiveness * edge
    return max(0.01, round(price, 2))


def closing_price(quote: OptionQuote, *, was_long: bool, aggressiveness: float) -> float:
    """Estimated exit price for a leg being closed."""
    return execution_price(quote, buying=not was_long, aggressiveness=aggressiveness)
