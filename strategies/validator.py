from __future__ import annotations

from collections import defaultdict

from data.schemas import OptionLeg, OptionType, Side
from strategies.payoff import finite_max_loss


def validate_defined_risk(legs: list[OptionLeg]) -> list[str]:
    reasons: list[str] = []
    if any("SPY" not in leg.contract_id for leg in legs):
        reasons.append("Strategies requiring non-SPY assets are not allowed")

    short_calls = [l for l in legs if l.side == Side.SELL and l.option_type == OptionType.CALL]
    long_calls = [l for l in legs if l.side == Side.BUY and l.option_type == OptionType.CALL]
    short_puts = [l for l in legs if l.side == Side.SELL and l.option_type == OptionType.PUT]
    long_puts = [l for l in legs if l.side == Side.BUY and l.option_type == OptionType.PUT]

    if short_calls and not long_calls:
        reasons.append("Naked short calls are forbidden")
    if short_puts and not long_puts:
        reasons.append("Naked short puts are forbidden")

    by_type: dict[OptionType, int] = defaultdict(int)
    for leg in legs:
        by_type[leg.option_type] += leg.quantity if leg.side == Side.BUY else -leg.quantity
    if abs(by_type[OptionType.CALL]) > 1 or abs(by_type[OptionType.PUT]) > 1:
        reasons.append("Unlimited-risk ratio spreads are forbidden")

    if not finite_max_loss(legs):
        reasons.append("Strategy does not have finite calculable maximum loss")

    return reasons
