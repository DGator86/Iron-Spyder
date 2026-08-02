"""Multi-leg order construction (spec section 31)."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from spy_der.domain.enums import CONTRACT_MULTIPLIER, OrderIntent, OrderStatus
from spy_der.domain.execution import MultiLegOrder, Position, RiskDecision
from spy_der.domain.strategy import Strategy, StrategyCandidate


def build_open_order(
    candidate: StrategyCandidate,
    decision: RiskDecision,
    *,
    now: datetime,
    order_id: str | None = None,
) -> MultiLegOrder:
    """Construct the atomic opening order for an approved candidate.

    The contract count comes from the risk decision, never from the candidate.
    The candidate is evaluated per contract and has no authority over size;
    routing sizing through the decision is what makes the risk limits
    unbypassable at the execution boundary.
    """
    if not decision.approved:
        raise ValueError("cannot build an order from an unapproved risk decision")
    if decision.max_contracts < 1:
        raise ValueError("risk decision approved zero contracts")

    return MultiLegOrder(
        order_id=order_id or f"O-{uuid4().hex[:12]}",
        created_at=now,
        strategy=candidate.strategy,
        contracts=decision.max_contracts,
        limit_price=decision.required_price_limit or candidate.entry_price,
        intent=OrderIntent.OPEN,
        status=OrderStatus.PENDING_NEW,
    )


def build_close_order(
    position: Position, *, limit_price: float, now: datetime, order_id: str | None = None
) -> MultiLegOrder:
    """Construct the closing order for an open position.

    The strategy is inverted leg by leg so the close is the exact offset of the
    open. Inverting preserves strikes and expirations, which is what lets the
    broker net the position to flat rather than leaving a residual leg.
    """
    inverted = Strategy(
        family=position.strategy.family,
        legs=tuple(leg.scaled(-1) for leg in position.strategy.legs),
    )
    return MultiLegOrder(
        order_id=order_id or f"C-{uuid4().hex[:12]}",
        created_at=now,
        strategy=inverted,
        contracts=position.contracts,
        limit_price=limit_price,
        intent=OrderIntent.CLOSE,
        status=OrderStatus.PENDING_NEW,
    )


def order_notional(order: MultiLegOrder) -> float:
    """Absolute cash value of the order at its limit price."""
    return abs(order.limit_price) * CONTRACT_MULTIPLIER * order.contracts


def describe(order: MultiLegOrder) -> str:
    """Human-readable order summary for logs and the audit trail."""
    legs = ", ".join(
        f"{leg.quantity:+d} {leg.expiry:%Y-%m-%d} {leg.strike:g}{leg.right.value[0].upper()}"
        for leg in order.strategy.legs
    )
    side = "DEBIT" if order.limit_price > 0 else "CREDIT"
    return (
        f"{order.intent.value.upper()} {order.contracts}x {order.strategy.family.value} "
        f"[{legs}] @ {side} {abs(order.limit_price):.2f}"
    )
