"""Position lifecycle and exit management (spec section 30).

Every open position is re-marked on each snapshot and tested against its full
exit contract. Exits are evaluated in a fixed precedence, with the
non-negotiable ones first:

1. kill switch;
2. assignment risk / pre-expiry closure;
3. end of session;
4. risk stop;
5. profit target;
6. state invalidation;
7. time stop.

Assignment control outranks both the stop and the target because spec 2.3 and
30.5 forbid relying on assignment at all: a short in-the-money leg must be
closed before settlement regardless of whether the trade is winning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from spy_der.domain.enums import CONTRACT_MULTIPLIER, ExitReason, MarketState
from spy_der.domain.execution import MultiLegOrder, Position
from spy_der.domain.forecast import ForecastBundle
from spy_der.domain.market import MarketSnapshot
from spy_der.execution.fills import CostModel, leg_slippage
from spy_der.execution.orders import build_close_order


@dataclass(frozen=True)
class ExitSignal:
    position_id: str
    reason: ExitReason
    detail: str
    mark_price: float
    unrealized_pnl: float


@dataclass
class PositionManager:
    """Marks open positions and decides when to close them."""

    costs: CostModel = field(default_factory=CostModel)
    assignment_moneyness_buffer: float = 0.002
    """Fraction of spot within which a short strike counts as assignment risk."""

    def mark(self, position: Position, snapshot: MarketSnapshot) -> Position:
        """Re-mark a position at currently executable prices.

        Marked at the price the position could actually be *closed* at, not at
        the midpoint. A midpoint mark systematically overstates open P&L by the
        half-spread on every leg, which flatters both the exit logic and the
        reported drawdown.
        """
        price = self.exit_price(position, snapshot)
        if price is None:
            return position
        return position.marked(price, snapshot.timestamp)

    def exit_price(self, position: Position, snapshot: MarketSnapshot) -> float | None:
        """Net per-share price to close the structure now, or ``None`` if unquoted."""
        total = 0.0
        for leg in position.legs:
            quote = snapshot.quote_for(leg.expiry, leg.strike, leg.right)
            if quote is None or quote.mid <= 0.0:
                return None
            slip = leg_slippage(quote, self.costs)
            # Closing a long means selling into the bid side, and vice versa.
            per_share = quote.mid - slip if leg.quantity > 0 else quote.mid + slip
            total += leg.quantity * per_share
        return total

    def check_exit(
        self,
        position: Position,
        snapshot: MarketSnapshot,
        forecast: ForecastBundle | None = None,
        *,
        kill_switch_active: bool = False,
    ) -> ExitSignal | None:
        """Return an exit signal when any exit condition is met."""
        marked = self.mark(position, snapshot)
        price = marked.current_price
        pnl = marked.unrealized_pnl
        plan = position.exit_plan

        def signal(reason: ExitReason, detail: str) -> ExitSignal:
            return ExitSignal(
                position_id=position.position_id,
                reason=reason,
                detail=detail,
                mark_price=price,
                unrealized_pnl=pnl,
            )

        if kill_switch_active:
            return signal(ExitReason.KILL_SWITCH, "kill switch active")

        assignment = self.assignment_exposure(position, snapshot)
        if assignment:
            return signal(ExitReason.ASSIGNMENT_RISK, assignment)

        minutes_left = snapshot.context.minutes_to_close
        if minutes_left <= plan.close_before_expiry_minutes:
            return signal(
                ExitReason.END_OF_SESSION, f"{minutes_left:.0f} minutes to close"
            )

        max_loss = position.max_loss_per_contract * position.contracts
        if max_loss > 0.0 and pnl <= -plan.stop_loss_fraction * max_loss:
            return signal(
                ExitReason.RISK_STOP,
                f"loss {pnl:.2f} beyond {plan.stop_loss_fraction:.0%} of {max_loss:.0f}",
            )

        max_profit = position.max_profit_per_contract * position.contracts
        if max_profit > 0.0 and pnl >= plan.profit_target_fraction * max_profit:
            return signal(
                ExitReason.PROFIT_TARGET,
                f"gain {pnl:.2f} reached {plan.profit_target_fraction:.0%} of {max_profit:.0f}",
            )

        if forecast is not None:
            invalidation = self.state_invalidated(position, forecast)
            if invalidation:
                return signal(ExitReason.STATE_INVALIDATION, invalidation)

        held = (snapshot.timestamp - position.opened_at).total_seconds() / 60.0
        if held >= plan.max_holding_minutes:
            return signal(
                ExitReason.TIME_STOP, f"held {held:.0f} of {plan.max_holding_minutes:.0f} minutes"
            )
        return None

    def assignment_exposure(self, position: Position, snapshot: MarketSnapshot) -> str | None:
        """Detect a short leg at risk of assignment (spec 30.5)."""
        spot = snapshot.spot
        buffer = spot * self.assignment_moneyness_buffer
        cutoff = timedelta(minutes=position.exit_plan.close_before_expiry_minutes)

        for leg in position.legs:
            if leg.quantity >= 0:
                continue
            time_left = leg.expiry - snapshot.timestamp
            in_the_money = (
                spot > leg.strike - buffer
                if leg.right.value == "call"
                else spot < leg.strike + buffer
            )
            if in_the_money and time_left <= cutoff:
                return (
                    f"short {leg.right.value} {leg.strike:g} is at or near the money "
                    f"with {time_left.total_seconds() / 60.0:.0f} minutes to expiry"
                )
        return None

    @staticmethod
    def state_invalidated(position: Position, forecast: ForecastBundle) -> str | None:
        """Detect that the thesis behind the position no longer holds."""
        plan = position.exit_plan
        for name in plan.invalidation_states:
            try:
                state = MarketState(name)
            except ValueError:  # pragma: no cover - defensive
                continue
            probability = forecast.state_probability(state)
            if probability >= 0.40:
                return f"{state.value} probability rose to {probability:.2f}"

        if forecast.confidence < plan.min_forecast_probability:
            return (
                f"forecast confidence {forecast.confidence:.2f} fell below "
                f"{plan.min_forecast_probability:.2f}"
            )
        return None

    def build_exit_order(
        self, position: Position, snapshot: MarketSnapshot, *, now: datetime | None = None
    ) -> MultiLegOrder | None:
        """Construct the closing order for a position being exited."""
        price = self.exit_price(position, snapshot)
        if price is None:
            return None
        # Closing inverts every leg, so the limit inverts too.
        return build_close_order(
            position, limit_price=-price, now=now or snapshot.timestamp
        )

    @staticmethod
    def realized_pnl(position: Position, close_price: float) -> float:
        """Dollar P&L from entry to ``close_price``, net of entry commission."""
        gross = (close_price - position.entry_price) * CONTRACT_MULTIPLIER * position.contracts
        return gross - position.entry_commission
