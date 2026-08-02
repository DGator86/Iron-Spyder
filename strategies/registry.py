from __future__ import annotations

from datetime import UTC, datetime

from data.schemas import OptionLeg, OptionType, Side

STRATEGY_FAMILIES = [
    "Bull call debit spread",
    "Bear put debit spread",
    "Bull put credit spread",
    "Bear call credit spread",
    "Iron condor",
    "Iron butterfly",
    "Long call butterfly",
    "Long put butterfly",
    "Long straddle",
    "Long strangle",
    "Defined-risk call backspread",
    "Defined-risk put backspread",
]


def _leg(contract: str, t: OptionType, strike: float, side: Side, premium: float, qty: int = 1) -> OptionLeg:
    return OptionLeg(contract_id=contract, option_type=t, strike=strike, expiration=datetime.now(UTC), side=side, quantity=qty, premium=premium)


def make_strategy(name: str, spot: float) -> list[OptionLeg]:
    s0 = round(spot / 5) * 5
    if name == "Bull call debit spread":
        return [_leg(f"SPY-{s0}-C", OptionType.CALL, s0, Side.BUY, 4.0), _leg(f"SPY-{s0+5}-C", OptionType.CALL, s0+5, Side.SELL, 2.0)]
    if name == "Bear put debit spread":
        return [_leg(f"SPY-{s0}-P", OptionType.PUT, s0, Side.BUY, 4.2), _leg(f"SPY-{s0-5}-P", OptionType.PUT, s0-5, Side.SELL, 2.1)]
    if name == "Bull put credit spread":
        return [_leg(f"SPY-{s0}-P", OptionType.PUT, s0, Side.SELL, 3.5), _leg(f"SPY-{s0-5}-P", OptionType.PUT, s0-5, Side.BUY, 1.8)]
    if name == "Bear call credit spread":
        return [_leg(f"SPY-{s0}-C", OptionType.CALL, s0, Side.SELL, 3.5), _leg(f"SPY-{s0+5}-C", OptionType.CALL, s0+5, Side.BUY, 1.7)]
    if name == "Iron condor":
        return [
            _leg(f"SPY-{s0-10}-P", OptionType.PUT, s0-10, Side.BUY, 1.0),
            _leg(f"SPY-{s0-5}-P", OptionType.PUT, s0-5, Side.SELL, 2.0),
            _leg(f"SPY-{s0+5}-C", OptionType.CALL, s0+5, Side.SELL, 2.0),
            _leg(f"SPY-{s0+10}-C", OptionType.CALL, s0+10, Side.BUY, 1.0),
        ]
    if name == "Iron butterfly":
        return [
            _leg(f"SPY-{s0}-P", OptionType.PUT, s0, Side.SELL, 4.0),
            _leg(f"SPY-{s0}-C", OptionType.CALL, s0, Side.SELL, 4.0),
            _leg(f"SPY-{s0-10}-P", OptionType.PUT, s0-10, Side.BUY, 1.0),
            _leg(f"SPY-{s0+10}-C", OptionType.CALL, s0+10, Side.BUY, 1.0),
        ]
    if name == "Long call butterfly":
        return [
            _leg(f"SPY-{s0-5}-C", OptionType.CALL, s0-5, Side.BUY, 3.0),
            _leg(f"SPY-{s0}-C", OptionType.CALL, s0, Side.SELL, 2.0, qty=2),
            _leg(f"SPY-{s0+5}-C", OptionType.CALL, s0+5, Side.BUY, 1.0),
        ]
    if name == "Long put butterfly":
        return [
            _leg(f"SPY-{s0+5}-P", OptionType.PUT, s0+5, Side.BUY, 3.0),
            _leg(f"SPY-{s0}-P", OptionType.PUT, s0, Side.SELL, 2.0, qty=2),
            _leg(f"SPY-{s0-5}-P", OptionType.PUT, s0-5, Side.BUY, 1.0),
        ]
    if name == "Long straddle":
        return [_leg(f"SPY-{s0}-C", OptionType.CALL, s0, Side.BUY, 4.0), _leg(f"SPY-{s0}-P", OptionType.PUT, s0, Side.BUY, 4.0)]
    if name == "Long strangle":
        return [_leg(f"SPY-{s0+5}-C", OptionType.CALL, s0+5, Side.BUY, 2.5), _leg(f"SPY-{s0-5}-P", OptionType.PUT, s0-5, Side.BUY, 2.5)]
    if name == "Defined-risk call backspread":
        return [
            _leg(f"SPY-{s0}-C", OptionType.CALL, s0, Side.SELL, 4.0),
            _leg(f"SPY-{s0+5}-C", OptionType.CALL, s0+5, Side.BUY, 2.0, qty=2),
            _leg(f"SPY-{s0+15}-C", OptionType.CALL, s0+15, Side.SELL, 0.5),
        ]
    if name == "Defined-risk put backspread":
        return [
            _leg(f"SPY-{s0}-P", OptionType.PUT, s0, Side.SELL, 4.0),
            _leg(f"SPY-{s0-5}-P", OptionType.PUT, s0-5, Side.BUY, 2.0, qty=2),
            _leg(f"SPY-{s0-15}-P", OptionType.PUT, s0-15, Side.SELL, 0.5),
        ]
    raise ValueError(f"Unknown strategy: {name}")
