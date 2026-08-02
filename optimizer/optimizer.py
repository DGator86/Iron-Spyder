from __future__ import annotations

from dataclasses import dataclass

from data.schemas import StrategyCandidate
from strategies.candidates import generate_candidates


@dataclass
class OptimizerConfig:
    min_liquidity: float = 0.3
    min_expected_return_on_risk: float = 0.08


STATE_FAMILY_FILTERS = {
    "Strong pin": {"Iron condor", "Iron butterfly", "Long call butterfly", "Long put butterfly"},
    "Broad range": {"Iron condor", "Long strangle"},
    "Bull grind": {"Bull call debit spread", "Bull put credit spread", "Defined-risk call backspread"},
    "Bear grind": {"Bear put debit spread", "Bear call credit spread", "Defined-risk put backspread"},
    "Volatility expansion": {"Long straddle", "Long strangle", "Defined-risk call backspread", "Defined-risk put backspread"},
}


def rank_candidates(spot: float, market_state: str, liquidity: float, config: OptimizerConfig) -> tuple[str, list[StrategyCandidate]]:
    family_filter = STATE_FAMILY_FILTERS.get(market_state)
    cands = generate_candidates(spot=spot, liquidity=liquidity, family_filter=family_filter)
    eligible = [
        c for c in cands
        if not c.rejection_reasons and c.max_loss > 0 and c.liquidity >= config.min_liquidity and c.expected_return_on_risk >= config.min_expected_return_on_risk
    ]
    ranked = sorted(eligible, key=lambda c: c.expected_utility, reverse=True)
    if not ranked:
        return "no_trade", []
    return ranked[0].name, ranked
