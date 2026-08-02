from __future__ import annotations

from datetime import UTC, datetime, timedelta

from data.schemas import OptionContractSnapshot, OptionType, SPYQuote


def synthetic_spy_quote(price: float = 550.0) -> SPYQuote:
    return SPYQuote(bid=price - 0.05, ask=price + 0.05, last=price, timestamp=datetime.now(UTC))


def generate_option_chain(spot: float = 550.0, days: int = 30, step: int = 5) -> list[OptionContractSnapshot]:
    expiry = datetime.now(UTC) + timedelta(days=days)
    chain: list[OptionContractSnapshot] = []
    for strike in range(int(spot * 0.8), int(spot * 1.2) + 1, step):
        moneyness = abs(strike - spot) / spot
        iv = 0.12 + 0.2 * moneyness
        spread = 0.08 + 0.02 * moneyness
        base = max(0.1, 8.0 * (1.0 - moneyness))
        for option_type in (OptionType.CALL, OptionType.PUT):
            mid = base if option_type == OptionType.CALL else base * 0.98
            bid = max(0.01, mid - spread / 2)
            ask = bid + spread
            contract_id = f"SPY-{expiry.date()}-{strike}-{option_type.value.upper()}"
            sign = 1.0 if option_type == OptionType.CALL else -1.0
            chain.append(
                OptionContractSnapshot(
                    contract_id=contract_id,
                    expiration=expiry,
                    strike=float(strike),
                    option_type=option_type,
                    bid=bid,
                    ask=ask,
                    last=(bid + ask) / 2,
                    implied_volatility=iv,
                    open_interest=max(1, int(2000 * (1 - moneyness))),
                    volume=max(1, int(500 * (1 - moneyness))),
                    gamma=0.01 * (1 - moneyness),
                    delta=sign * max(0.1, 0.5 - moneyness),
                )
            )
    return chain
