"""Paper broker with a realistic fill model (spec sections 31, 40 phase 2).

Fills are executable-price fills, not midpoint fills. An order fills only if
the market is actually there at the limit, and it pays the modelled slippage
when it does. Spec 40 requires paper trading to use executable prices before
anything advances toward live, so this broker is deliberately pessimistic:

* an order that cannot fill at its limit is repriced a bounded number of times
  and then expires, rather than filling at whatever price is available;
* partial fills are supported and leave a *reduced but still complete*
  structure, never a partially built one;
* commissions and fees are charged on both the open and the close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from spy_der.domain.enums import CONTRACT_MULTIPLIER, OrderIntent, OrderStatus
from spy_der.domain.execution import BrokerState, Fill, MultiLegOrder, Position
from spy_der.domain.market import MarketSnapshot, OptionQuote
from spy_der.execution.broker_base import Broker, BrokerError
from spy_der.execution.fills import CostModel, fill_probability, leg_slippage


@dataclass
class PaperBrokerConfig:
    starting_equity: float = 100_000.0
    max_reprice_attempts: int = 3
    reprice_step: float = 0.01
    """Net price concession per reprice attempt, per leg."""

    allow_partial_fills: bool = True
    seed: int | None = 424242


@dataclass
class PaperBroker(Broker):
    """Simulated broker used for paper trading and backtests."""

    config: PaperBrokerConfig = field(default_factory=PaperBrokerConfig)
    costs: CostModel = field(default_factory=CostModel)

    equity: float = field(init=False)
    orders: dict[str, MultiLegOrder] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    _positions: dict[str, Position] = field(default_factory=dict)
    connected: bool = True
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.equity = self.config.starting_equity
        self._rng = np.random.default_rng(self.config.seed)

    # -- Broker interface --------------------------------------------------

    @property
    def supports_live(self) -> bool:
        return False

    def state(self) -> BrokerState:
        return BrokerState(
            connected=self.connected,
            equity=self.equity,
            buying_power=self.equity,
            reconciled=True,
        )

    def submit(self, order: MultiLegOrder, snapshot: MarketSnapshot) -> MultiLegOrder:
        if not self.connected:
            raise BrokerError("paper broker is disconnected")
        if order.order_id in self.orders:
            raise BrokerError(f"duplicate order id {order.order_id}")

        working = order.with_status(OrderStatus.WORKING, broker_order_id=f"B{len(self.orders):06d}")
        self.orders[order.order_id] = working
        return working

    def poll(
        self, order_id: str, snapshot: MarketSnapshot
    ) -> tuple[MultiLegOrder, Fill | None]:
        order = self.orders.get(order_id)
        if order is None:
            raise BrokerError(f"unknown order {order_id}")
        if order.is_terminal:
            return order, None

        quotes = self._quotes_for(order, snapshot)
        if quotes is None:
            updated = order.with_status(
                OrderStatus.REJECTED, reject_reason="one or more legs are not quoted"
            )
            self.orders[order_id] = updated
            return updated, None

        achievable = self._achievable_price(order, quotes)
        acceptable = (
            achievable <= order.limit_price
            if order.limit_price >= 0.0
            else achievable <= order.limit_price
        )

        if not acceptable:
            return self._reprice_or_expire(order, snapshot)

        probability = fill_probability(
            quotes,
            leg_count=order.strategy.leg_count,
            contracts=order.remaining_contracts,
            model=self.costs,
            minutes_to_close=snapshot.context.minutes_to_close,
        )
        if self._rng.random() > probability:
            return self._reprice_or_expire(order, snapshot)

        contracts = self._fill_size(order, quotes, probability)
        if contracts < 1:
            return self._reprice_or_expire(order, snapshot)

        return self._record_fill(order, snapshot, achievable, contracts, quotes)

    def cancel(self, order_id: str, *, now: datetime) -> MultiLegOrder:
        order = self.orders.get(order_id)
        if order is None:
            raise BrokerError(f"unknown order {order_id}")
        if order.is_terminal:
            return order
        cancelled = order.with_status(OrderStatus.CANCELLED)
        self.orders[order_id] = cancelled
        return cancelled

    def positions(self) -> tuple[Position, ...]:
        return tuple(p for p in self._positions.values() if p.is_open)

    # -- position bookkeeping ---------------------------------------------

    def register_position(self, position: Position) -> None:
        self._positions[position.position_id] = position

    def settle_position(self, position: Position, *, pnl: float) -> None:
        self._positions[position.position_id] = position
        self.equity += pnl

    def disconnect(self) -> None:
        """Simulate a broker outage; used by the kill-switch tests."""
        self.connected = False

    def reconnect(self) -> None:
        self.connected = True

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _quotes_for(
        order: MultiLegOrder, snapshot: MarketSnapshot
    ) -> tuple[OptionQuote, ...] | None:
        quotes: list[OptionQuote] = []
        for leg in order.strategy.legs:
            quote = snapshot.quote_for(leg.expiry, leg.strike, leg.right)
            if quote is None or quote.mid <= 0.0:
                return None
            quotes.append(quote)
        return tuple(quotes)

    def _achievable_price(
        self, order: MultiLegOrder, quotes: tuple[OptionQuote, ...]
    ) -> float:
        """Net price actually available now, per share, with slippage applied.

        Signed so that lower is always better for the account: a debit is
        positive and is minimized, a credit is negative and is made more
        negative. Comparing achievable to limit is then one inequality in both
        directions.
        """
        total = 0.0
        for leg, quote in zip(order.strategy.legs, quotes):
            slip = leg_slippage(quote, self.costs)
            per_share = quote.mid + (slip if leg.quantity > 0 else -slip)
            total += leg.quantity * per_share
        return total

    def _fill_size(
        self, order: MultiLegOrder, quotes: tuple[OptionQuote, ...], probability: float
    ) -> int:
        depth = min(min(q.bid_size, q.ask_size) for q in quotes)
        capacity = max(0, int(depth // max(1, max(abs(leg.quantity) for leg in order.strategy.legs))))
        wanted = order.remaining_contracts
        if capacity >= wanted:
            return wanted
        if not self.config.allow_partial_fills:
            return 0
        return max(0, min(wanted, capacity))

    def _reprice_or_expire(
        self, order: MultiLegOrder, snapshot: MarketSnapshot
    ) -> tuple[MultiLegOrder, None]:
        """Concede one step, or give up once the attempt budget is spent."""
        if order.reprice_attempts >= self.config.max_reprice_attempts:
            expired = order.with_status(
                OrderStatus.EXPIRED,
                reject_reason=f"unfilled after {order.reprice_attempts} reprice attempts",
            )
            self.orders[order.order_id] = expired
            return expired, None

        concession = self.config.reprice_step * order.strategy.leg_count
        new_limit = (
            order.limit_price + concession
            if order.limit_price >= 0.0
            else order.limit_price - concession
        )
        repriced = order.with_status(
            OrderStatus.WORKING,
            limit_price=new_limit,
            reprice_attempts=order.reprice_attempts + 1,
        )
        self.orders[order.order_id] = repriced
        return repriced, None

    def _record_fill(
        self,
        order: MultiLegOrder,
        snapshot: MarketSnapshot,
        price: float,
        contracts: int,
        quotes: tuple[OptionQuote, ...],
    ) -> tuple[MultiLegOrder, Fill]:
        per_contract_fees = self.costs.commission_per_contract + self.costs.exchange_fee_per_contract
        commission = (
            sum(abs(leg.quantity) for leg in order.strategy.legs) * per_contract_fees * contracts
        )
        mid_price = sum(
            leg.quantity * quote.mid for leg, quote in zip(order.strategy.legs, quotes)
        )

        fill = Fill(
            order_id=order.order_id,
            filled_at=snapshot.timestamp,
            contracts=contracts,
            fill_price=price,
            commission=commission,
            slippage_vs_mid=abs(price - mid_price) * CONTRACT_MULTIPLIER * contracts,
        )
        self.fills.append(fill)

        filled_total = order.filled_contracts + contracts
        status = (
            OrderStatus.FILLED
            if filled_total >= order.contracts
            else OrderStatus.PARTIALLY_FILLED
        )
        updated = order.with_status(status, filled_contracts=filled_total)
        self.orders[order.order_id] = updated

        if order.intent is OrderIntent.OPEN:
            self.equity -= commission
        return updated, fill
