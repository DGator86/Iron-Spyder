from __future__ import annotations

from itertools import pairwise

from data.schemas import OptionLeg, Side


def leg_payoff(leg: OptionLeg, spot: float) -> float:
    intrinsic = max(0.0, spot - leg.strike) if leg.option_type.value == "call" else max(0.0, leg.strike - spot)
    sign = 1 if leg.side == Side.BUY else -1
    return sign * leg.quantity * (intrinsic - leg.premium)


def strategy_payoff(legs: list[OptionLeg], spot: float) -> float:
    return sum(leg_payoff(leg, spot) for leg in legs)


def payoff_metrics(legs: list[OptionLeg]) -> tuple[float, float, list[float]]:
    strikes = sorted({leg.strike for leg in legs})
    grid = [0.0, *strikes, max(strikes) * 2 if strikes else 1000.0]
    vals = [strategy_payoff(legs, s) for s in grid]
    max_profit = max(vals)
    max_loss = -min(vals)
    breakevens: list[float] = []
    for a, b in pairwise(grid):
        pa = strategy_payoff(legs, a)
        pb = strategy_payoff(legs, b)
        if pa == 0:
            breakevens.append(a)
        elif pa * pb < 0:
            breakevens.append((a + b) / 2)
    return max_profit, max_loss, breakevens


def finite_max_loss(legs: list[OptionLeg]) -> bool:
    extremes = [strategy_payoff(legs, 0), strategy_payoff(legs, 10_000)]
    return all(v > -1e9 for v in extremes)
