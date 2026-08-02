from __future__ import annotations

from typing import TypedDict

from analytics.metrics import (
    atm_expected_move,
    call_wall,
    delta_exposure,
    gamma_exposure,
    gamma_flip,
    gamma_profile,
    liquidity_score,
    put_call_skew,
    put_wall,
)
from data.schemas import OptionContractSnapshot, SPYQuote, StrategyCandidate
from data.synthetic import generate_option_chain, synthetic_spy_quote
from execution.broker import PaperBroker
from models.market_state import StateClassification, classify_market_state
from optimizer.optimizer import OptimizerConfig, rank_candidates


class Snapshot(TypedDict):
    quote: SPYQuote
    chain: list[OptionContractSnapshot]
    expected_move: float
    gamma_exposure: float
    delta_exposure: float
    gamma_profile: list[tuple[float, float]]
    gamma_flip: float | None
    call_wall: float
    put_wall: float
    liquidity: float
    market_state: StateClassification
    best: str
    ranked: list[StrategyCandidate]


class AppState:
    def __init__(self) -> None:
        self.broker = PaperBroker()
        self.config = OptimizerConfig()

    def snapshot(self) -> Snapshot:
        quote = synthetic_spy_quote()
        chain = generate_option_chain(spot=quote.last)
        iv = chain[len(chain) // 2].implied_volatility
        em = atm_expected_move(quote.last, iv)
        liq = liquidity_score(chain)
        grid = [quote.last + x for x in range(-30, 31, 5)]
        gp = gamma_profile(chain, grid)
        gf = gamma_flip(gp)
        cls = classify_market_state(em, quote.last, gf, put_call_skew(chain))
        best, ranked = rank_candidates(quote.last, cls.state, liq, self.config)
        return {
            "quote": quote,
            "chain": chain,
            "expected_move": em,
            "gamma_exposure": gamma_exposure(chain),
            "delta_exposure": delta_exposure(chain),
            "gamma_profile": gp,
            "gamma_flip": gf,
            "call_wall": call_wall(chain),
            "put_wall": put_wall(chain),
            "liquidity": liq,
            "market_state": cls,
            "best": best,
            "ranked": ranked,
        }


STATE = AppState()
