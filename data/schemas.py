from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class SPYQuote(BaseModel):
    symbol: str = "SPY"
    bid: float
    ask: float
    last: float
    timestamp: datetime


class SPYPriceBar(BaseModel):
    symbol: str = "SPY"
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class OptionContractSnapshot(BaseModel):
    symbol: str = "SPY"
    contract_id: str
    expiration: datetime
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    last: float
    implied_volatility: float = Field(gt=0)
    open_interest: int = Field(ge=0)
    volume: int = Field(ge=0)
    gamma: float = 0.0
    delta: float = 0.0


class OptionTrade(BaseModel):
    contract_id: str
    side: Side
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    timestamp: datetime


class MarketSnapshot(BaseModel):
    quote: SPYQuote
    options: list[OptionContractSnapshot]


class ForecastBundle(BaseModel):
    market_state_probabilities: dict[str, float]
    expected_move: float


class OptionLeg(BaseModel):
    contract_id: str
    option_type: OptionType
    strike: float
    expiration: datetime
    side: Side
    quantity: int = Field(gt=0)
    premium: float = Field(ge=0)


class StrategyCandidate(BaseModel):
    name: str
    legs: list[OptionLeg]
    max_profit: float
    max_loss: float
    breakevens: list[float]
    expected_return_on_risk: float
    liquidity: float
    expected_utility: float
    rejection_reasons: list[str] = Field(default_factory=list)


class RiskDecision(BaseModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)


class OrderStatus(str, Enum):
    NEW = "new"
    FILLED = "filled"
    REJECTED = "rejected"


class Order(BaseModel):
    order_id: str
    legs: list[OptionLeg]
    limit_price: float
    status: OrderStatus = OrderStatus.NEW


class Fill(BaseModel):
    order_id: str
    fill_price: float
    filled_at: datetime


class Position(BaseModel):
    position_id: str
    strategy_name: str
    legs: list[OptionLeg]
    entry_price: float
    current_price: float
    realized_pnl: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        return self.current_price - self.entry_price


class RiskModel(BaseModel):
    finite_max_loss: bool
    requires_shares: bool = False


class StrategySpec(BaseModel):
    name: str
    legs: list[OptionLeg]

    @model_validator(mode="after")
    def ensure_spy_only(self) -> StrategySpec:
        if any("SPY" not in leg.contract_id for leg in self.legs):
            raise ValueError("Only SPY options are supported")
        return self
