from datetime import UTC, datetime

from data.schemas import OptionLeg, OptionType, Side
from strategies.payoff import payoff_metrics
from strategies.validator import validate_defined_risk


def test_strategy_payoff_and_max_loss() -> None:
    legs = [
        OptionLeg(contract_id="SPY-C1", option_type=OptionType.CALL, strike=550, expiration=datetime.now(UTC), side=Side.BUY, quantity=1, premium=4),
        OptionLeg(contract_id="SPY-C2", option_type=OptionType.CALL, strike=555, expiration=datetime.now(UTC), side=Side.SELL, quantity=1, premium=2),
    ]
    max_profit, max_loss, _ = payoff_metrics(legs)
    assert max_loss >= 0
    assert max_profit >= 0


def test_reject_undefined_risk() -> None:
    legs = [
        OptionLeg(contract_id="SPY-C1", option_type=OptionType.CALL, strike=550, expiration=datetime.now(UTC), side=Side.SELL, quantity=1, premium=3),
    ]
    reasons = validate_defined_risk(legs)
    assert any("Naked short calls" in r for r in reasons)
