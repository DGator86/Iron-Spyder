"""Broker interface (spec section 31).

A live adapter is intentionally **not** shipped. Spec 40 gates live execution
behind six months of paper trading, 250 completed trades, shadow execution, and
an explicit limited-live phase; publishing a live adapter now would invite
skipping all of that. Implement this interface against a real broker only after
those gates are met, and keep :meth:`supports_live` returning ``False`` until
the adapter has passed reconciliation testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from spy_der.domain.execution import BrokerState, Fill, MultiLegOrder, Position
from spy_der.domain.market import MarketSnapshot


class Broker(ABC):
    """Submits atomic multi-leg orders and reports state."""

    @property
    def supports_live(self) -> bool:
        """Whether this adapter may place real orders."""
        return False

    @abstractmethod
    def state(self) -> BrokerState:
        """Current connectivity, equity, and reconciliation status."""

    @abstractmethod
    def submit(self, order: MultiLegOrder, snapshot: MarketSnapshot) -> MultiLegOrder:
        """Submit an order and return it with an updated status."""

    @abstractmethod
    def poll(self, order_id: str, snapshot: MarketSnapshot) -> tuple[MultiLegOrder, Fill | None]:
        """Advance an order's lifecycle against the current market."""

    @abstractmethod
    def cancel(self, order_id: str, *, now: datetime) -> MultiLegOrder:
        """Cancel a working order."""

    @abstractmethod
    def positions(self) -> tuple[Position, ...]:
        """Positions as the broker sees them, for reconciliation."""


class BrokerError(RuntimeError):
    """Raised when the broker rejects a request outright."""
