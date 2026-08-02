from __future__ import annotations

from data.schemas import StrategyCandidate
from strategies.payoff import payoff_metrics
from strategies.registry import STRATEGY_FAMILIES, make_strategy
from strategies.validator import validate_defined_risk


def generate_candidates(spot: float, liquidity: float, family_filter: set[str] | None = None) -> list[StrategyCandidate]:
    candidates: list[StrategyCandidate] = []
    families = [f for f in STRATEGY_FAMILIES if family_filter is None or f in family_filter]
    for family in families:
        legs = make_strategy(family, spot)
        reasons = validate_defined_risk(legs)
        max_profit, max_loss, bes = payoff_metrics(legs)
        eror = 0.0 if max_loss <= 0 else max_profit / max_loss * 0.25
        candidates.append(
            StrategyCandidate(
                name=family,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakevens=bes,
                expected_return_on_risk=eror,
                liquidity=liquidity,
                expected_utility=0.6 * eror + 0.4 * liquidity,
                rejection_reasons=reasons,
            )
        )
    return candidates
