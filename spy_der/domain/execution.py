"""Order, fill, and position types (spec sections 31 and 34.4)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from spy_der.domain.enums import (
    CONTRACT_MULTIPLIER,
    ExitReason,
    OrderIntent,
    OrderStatus,
    StrategyFamily,
)
from spy_der.domain.strategy import ExitPlan, OptionLeg, Strategy


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Risk-engine verdict (spec 34.4). The risk engine has absolute veto."""

    approved: bool
    veto_reasons: tuple[str, ...] = ()
    max_contracts: int = 0
    approved_risk: float = 0.0
    required_price_limit: float = 0.0

    @classmethod
    def veto(cls, *reasons: str) -> RiskDecision:
        return cls(approved=False, veto_reasons=tuple(reasons))

    def merged(self, other: RiskDecision) -> RiskDecision:
        """Combine two verdicts; any veto wins and limits take the minimum."""
        return RiskDecision(
            approved=self.approved and other.approved,
            veto_reasons=self.veto_reasons + other.veto_reasons,
            max_contracts=min(self.max_contracts, other.max_contracts),
            approved_risk=min(self.approved_risk, other.approved_risk),
            required_price_limit=(
                min(self.required_price_limit, other.required_price_limit)
                if self.required_price_limit and other.required_price_limit
                else (self.required_price_limit or other.required_price_limit)
            ),
        )


@dataclass(frozen=True, slots=True)
class MultiLegOrder:
    """An atomic multi-leg order.

    The system submits complex structures as one order. Legging in is not
    supported (spec section 31) because a partially built defined-risk
    structure is, transiently, an undefined-risk position.
    """

    order_id: str
    created_at: datetime
    strategy: Strategy
    contracts: int
    limit_price: float
    intent: OrderIntent = OrderIntent.OPEN
    status: OrderStatus = OrderStatus.PENDING_NEW
    filled_contracts: int = 0
    reprice_attempts: int = 0
    broker_order_id: str | None = None
    reject_reason: str | None = None

    @property
    def is_debit(self) -> bool:
        return self.limit_price > 0.0

    @property
    def remaining_contracts(self) -> int:
        return max(0, self.contracts - self.filled_contracts)

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

    def with_status(self, status: OrderStatus, **changes: object) -> MultiLegOrder:
        return replace(self, status=status, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Fill:
    """A realized execution against a :class:`MultiLegOrder`."""

    order_id: str
    filled_at: datetime
    contracts: int
    fill_price: float
    commission: float
    slippage_vs_mid: float

    @property
    def cash_flow(self) -> float:
        """Signed cash impact: negative for a debit paid, positive for a credit."""
        return -(self.fill_price * CONTRACT_MULTIPLIER * self.contracts) - self.commission


@dataclass(frozen=True, slots=True)
class Position:
    """An open defined-risk position with its immutable risk envelope.

    ``max_loss_per_contract`` is fixed at entry and is never rewritten. Spec
    section 28.4 forbids altering the maximum permitted loss after entry, so the
    field is part of the frozen record rather than something recomputed on
    every mark.
    """

    position_id: str
    opened_at: datetime
    family: StrategyFamily
    strategy: Strategy
    contracts: int
    entry_price: float
    max_loss_per_contract: float
    max_profit_per_contract: float
    exit_plan: ExitPlan
    entry_commission: float = 0.0
    current_price: float = 0.0
    marked_at: datetime | None = None
    realized_pnl: float = 0.0
    exit_reason: ExitReason | None = None
    closed_at: datetime | None = None
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def legs(self) -> tuple[OptionLeg, ...]:
        return self.strategy.legs

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def total_max_loss(self) -> float:
        """Worst-case dollar loss for the whole position, including entry costs."""
        return self.max_loss_per_contract * self.contracts + self.entry_commission

    @property
    def unrealized_pnl(self) -> float:
        """Mark-to-market P&L.

        ``entry_price`` and ``current_price`` are both net structure prices in
        per-share terms with debits positive, so the position gains when the
        structure marks above the entry debit (or below the entry credit).
        """
        return (self.current_price - self.entry_price) * CONTRACT_MULTIPLIER * self.contracts

    @property
    def pnl(self) -> float:
        return self.realized_pnl if not self.is_open else self.unrealized_pnl

    @property
    def loss_fraction(self) -> float:
        """Current loss as a fraction of the position's maximum loss."""
        denominator = self.max_loss_per_contract * self.contracts
        if denominator <= 0.0:
            return 0.0
        return max(0.0, -self.unrealized_pnl) / denominator

    @property
    def capture_ratio(self) -> float:
        """``(current - entry) / (max - entry)`` (spec 30.1)."""
        span = self.max_profit_per_contract
        if span <= 0.0:
            return 0.0
        return (self.current_price - self.entry_price) * CONTRACT_MULTIPLIER * self.contracts / (
            span * CONTRACT_MULTIPLIER * self.contracts
        )

    def marked(self, price: float, at: datetime) -> Position:
        return replace(self, current_price=price, marked_at=at)

    def closed(self, *, at: datetime, pnl: float, reason: ExitReason) -> Position:
        return replace(self, closed_at=at, realized_pnl=pnl, exit_reason=reason)


@dataclass(frozen=True, slots=True)
class BrokerState:
    """Broker connectivity and account state used by the risk engine."""

    connected: bool = True
    equity: float = 100_000.0
    buying_power: float = 100_000.0
    last_heartbeat: datetime | None = None
    reconciled: bool = True
    discrepancies: tuple[str, ...] = ()

    @property
    def is_healthy(self) -> bool:
        return self.connected and self.reconciled and not self.discrepancies
