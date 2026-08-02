from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from uuid import uuid4

from data.schemas import Fill, Order, OrderStatus, Position, StrategyCandidate
from risk.rules import KillSwitch, RiskLimits, approve_trade


class Broker(ABC):
    @abstractmethod
    def place_order(self, order: Order) -> Order:
        raise NotImplementedError


class PaperBroker(Broker):
    def __init__(self, slippage_bps: float = 5.0) -> None:
        self.slippage_bps = slippage_bps
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.kill_switch = KillSwitch()
        self.limits = RiskLimits()

    def place_order(self, order: Order) -> Order:
        self.orders[order.order_id] = order
        return order

    def execute_candidate(self, candidate: StrategyCandidate, open_positions: int) -> tuple[Order, Fill | None, str | None]:
        decision = approve_trade(candidate, open_positions, self.limits, self.kill_switch)
        order = Order(order_id=str(uuid4()), legs=candidate.legs, limit_price=sum(l.premium for l in candidate.legs))
        if not decision.approved:
            order.status = OrderStatus.REJECTED
            self.orders[order.order_id] = order
            return order, None, "; ".join(decision.reasons)
        slip = order.limit_price * (self.slippage_bps / 10_000)
        fill_price = order.limit_price + slip
        fill = Fill(order_id=order.order_id, fill_price=fill_price, filled_at=datetime.now(UTC))
        order.status = OrderStatus.FILLED
        self.orders[order.order_id] = order
        self.fills.append(fill)
        position = Position(
            position_id=order.order_id,
            strategy_name=candidate.name,
            legs=candidate.legs,
            entry_price=fill_price,
            current_price=fill_price,
        )
        self.positions[position.position_id] = position
        return order, fill, None

    def close_position(self, position_id: str, mark_price: float) -> Position | None:
        position = self.positions.get(position_id)
        if position is None:
            return None
        position.current_price = mark_price
        position.realized_pnl = mark_price - position.entry_price
        del self.positions[position_id]
        return position
