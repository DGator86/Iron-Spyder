from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

import numpy as np

from data.schemas import OptionContractSnapshot, OptionType


def atm_expected_move(spot: float, iv: float, days: int = 30) -> float:
    return float(spot * iv * np.sqrt(days / 365))


def gamma_exposure(chain: Iterable[OptionContractSnapshot]) -> float:
    return float(sum(o.gamma * o.open_interest * 100 for o in chain))


def delta_exposure(chain: Iterable[OptionContractSnapshot]) -> float:
    return float(sum(o.delta * o.open_interest * 100 for o in chain))


def gamma_profile(chain: list[OptionContractSnapshot], price_grid: list[float]) -> list[tuple[float, float]]:
    profile: list[tuple[float, float]] = []
    for price in price_grid:
        val = sum((1 - abs(o.strike - price) / max(price, 1)) * o.gamma * o.open_interest for o in chain)
        profile.append((price, float(val)))
    return profile


def gamma_flip(profile: list[tuple[float, float]]) -> float | None:
    for (p0, g0), (p1, g1) in pairwise(profile):
        if g0 == 0:
            return p0
        if g0 * g1 < 0:
            return (p0 + p1) / 2
    return None


def call_wall(chain: Iterable[OptionContractSnapshot]) -> float:
    calls = [o for o in chain if o.option_type == OptionType.CALL]
    return max(calls, key=lambda o: o.open_interest).strike


def put_wall(chain: Iterable[OptionContractSnapshot]) -> float:
    puts = [o for o in chain if o.option_type == OptionType.PUT]
    return max(puts, key=lambda o: o.open_interest).strike


def put_call_skew(chain: Iterable[OptionContractSnapshot]) -> float:
    calls = [o.implied_volatility for o in chain if o.option_type == OptionType.CALL]
    puts = [o.implied_volatility for o in chain if o.option_type == OptionType.PUT]
    return float(np.mean(puts) - np.mean(calls))


def liquidity_score(chain: Iterable[OptionContractSnapshot]) -> float:
    vals = [min(1.0, (o.volume + o.open_interest / 10) / 1000) - min(0.5, (o.ask - o.bid)) for o in chain]
    return float(np.clip(np.mean(vals), 0, 1))
