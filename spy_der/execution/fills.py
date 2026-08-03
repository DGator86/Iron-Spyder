"""Fill probability and slippage modelling (spec section 31.1).

Spec 37.7 is blunt: a strategy that depends on midpoint fills is not production
ready. So every candidate is costed with an explicit, pessimistic-by-default
slippage estimate and a fill probability below one, and the cost stress tests
sweep both.

Slippage is modelled per leg and summed. Multi-leg structures do not get the
netting benefit a single-leg trade would: each leg crosses its own spread, and
the complex-order book for a four-leg structure is thinner than any individual
leg's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spy_der.domain.enums import CONTRACT_MULTIPLIER
from spy_der.domain.market import OptionQuote
from spy_der.domain.strategy import LiquidityProfile, OptionLeg


@dataclass(frozen=True)
class CostModel:
    """Commission and slippage assumptions (spec 37.7)."""

    commission_per_contract: float = 0.65
    exchange_fee_per_contract: float = 0.05
    spread_capture: float = 0.5
    """Fraction of the half-spread paid on each leg, each way.

    ``0.0`` is a midpoint fill and ``1.0`` pays the full half-spread. The
    default assumes no price improvement beyond meeting in the middle of the
    remaining edge.
    """

    spread_multiplier: float = 1.0
    """Stress multiplier applied to every quoted spread (spec 37.7)."""

    adverse_ticks: float = 0.0
    """Extra ticks of adverse selection per leg, each way."""

    tick_size: float = 0.01
    fill_probability_floor: float = 0.05

    def stressed(
        self, *, spread_multiplier: float = 1.0, adverse_ticks: float = 0.0
    ) -> CostModel:
        from dataclasses import replace

        return replace(
            self,
            spread_multiplier=self.spread_multiplier * spread_multiplier,
            adverse_ticks=self.adverse_ticks + adverse_ticks,
        )


def leg_slippage(quote: OptionQuote, model: CostModel) -> float:
    """Per-share slippage versus the midpoint for one leg, one way."""
    half_spread = 0.5 * max(0.0, quote.ask - quote.bid) * model.spread_multiplier
    return half_spread * model.spread_capture + model.adverse_ticks * model.tick_size


def round_trip_cost(
    legs: Sequence[OptionLeg],
    quotes: Sequence[OptionQuote],
    model: CostModel,
    *,
    contracts: int = 1,
) -> float:
    """Total dollar cost of opening and closing the structure once.

    Commissions are charged on both the open and the close; slippage is charged
    on each leg in both directions.
    """
    if len(legs) != len(quotes):
        raise ValueError("legs and quotes must correspond one to one")

    slippage = sum(
        abs(leg.quantity) * CONTRACT_MULTIPLIER * leg_slippage(quote, model) * 2.0
        for leg, quote in zip(legs, quotes)
    )
    per_contract_fees = model.commission_per_contract + model.exchange_fee_per_contract
    commissions = sum(abs(leg.quantity) for leg in legs) * per_contract_fees * 2.0
    return (slippage + commissions) * contracts


def entry_slippage(
    legs: Sequence[OptionLeg], quotes: Sequence[OptionQuote], model: CostModel
) -> float:
    """Dollar slippage on entry alone, per contract."""
    return sum(
        abs(leg.quantity) * CONTRACT_MULTIPLIER * leg_slippage(quote, model)
        for leg, quote in zip(legs, quotes)
    )


def liquidity_profile(quotes: Sequence[OptionQuote]) -> LiquidityProfile:
    """Aggregate the execution quality of the legs of one structure."""
    if not quotes:
        return LiquidityProfile(
            score=0.0,
            worst_spread_ratio=float("inf"),
            total_open_interest=0,
            total_volume=0,
            min_quote_size=0,
        )

    spread_ratios = [q.spread_ratio for q in quotes]
    worst = max(spread_ratios)
    sizes = [min(q.bid_size, q.ask_size) for q in quotes]

    spread_term = float(np.clip(1.0 - worst / 0.25, 0.0, 1.0))
    size_term = float(np.clip(np.log1p(min(sizes)) / np.log1p(100.0), 0.0, 1.0))
    oi_term = float(
        np.clip(np.log1p(min(q.open_interest for q in quotes)) / np.log1p(2_000.0), 0.0, 1.0)
    )
    score = 0.55 * spread_term + 0.25 * size_term + 0.20 * oi_term

    return LiquidityProfile(
        score=float(np.clip(score, 0.0, 1.0)),
        worst_spread_ratio=worst,
        total_open_interest=sum(q.open_interest for q in quotes),
        total_volume=sum(q.volume for q in quotes),
        min_quote_size=min(sizes),
    )


def fill_probability(
    quotes: Sequence[OptionQuote],
    *,
    leg_count: int,
    contracts: int,
    model: CostModel,
    minutes_to_close: float = 390.0,
) -> float:
    """Probability the complex order fills at the requested limit (spec 31.1).

    Falls with spread width, leg count, and requested size relative to quoted
    depth, and near the close when complex-order books thin out.
    """
    if not quotes:
        return 0.0

    worst_spread = max(q.spread_ratio for q in quotes)
    spread_term = float(np.clip(1.0 - worst_spread / 0.30, 0.05, 1.0))

    depth = min(min(q.bid_size, q.ask_size) for q in quotes)
    size_term = float(np.clip(depth / max(1.0, contracts * 2.0), 0.1, 1.0))

    # Each additional leg beyond two reduces the chance of an atomic fill.
    leg_term = float(np.clip(1.0 - 0.08 * max(0, leg_count - 2), 0.4, 1.0))

    # Historical feeds (and some live snapshots early in the session) report
    # volume=0 while open interest is meaningful. Use OI as a soft activity
    # proxy so fill probability is not structurally capped below typical
    # optimizer gates whenever volume is missing.
    activity = min(_activity_proxy(q) for q in quotes)
    volume_term = float(np.clip(np.log1p(activity) / np.log1p(200.0), 0.25, 1.0))
    close_term = 1.0 if minutes_to_close > 15.0 else 0.6

    probability = spread_term * size_term * leg_term * volume_term * close_term
    return float(np.clip(probability, model.fill_probability_floor, 0.99))


def _activity_proxy(quote: OptionQuote) -> float:
    if quote.volume > 0:
        return float(quote.volume)
    if quote.open_interest > 0:
        return float(max(1, quote.open_interest // 20))
    return 0.0
