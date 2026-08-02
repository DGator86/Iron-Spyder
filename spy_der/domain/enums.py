"""Enumerations shared across the SPY-DER system.

The taxonomies here are governed by the specification and must not be widened
casually: :class:`MarketState` is the operational state space of section 11 and
:class:`StrategyFamily` is the allowed strategy universe of section 21.
"""

from __future__ import annotations

from enum import Enum

CONTRACT_MULTIPLIER = 100
"""Shares per SPY option contract."""


class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    AMERICAN = "american"
    EUROPEAN = "european"


class MarketState(str, Enum):
    """Operational state space (spec section 11)."""

    STRONG_PIN = "StrongPin"
    BROAD_RANGE = "BroadRange"
    BULL_GRIND = "BullGrind"
    BEAR_GRIND = "BearGrind"
    BULL_BREAKOUT = "BullBreakout"
    BEAR_BREAKDOWN = "BearBreakdown"
    VOL_EXPANSION = "VolExpansion"
    VOL_CONTRACTION = "VolContraction"
    DIRECTIONAL_RANGE = "DirectionalRange"
    TRANSITION = "Transition"
    UNSTABLE = "Unstable"


#: States in which the system is never permitted to open a position (spec 11.10, 11.11).
NO_TRADE_STATES: frozenset[MarketState] = frozenset(
    {MarketState.TRANSITION, MarketState.UNSTABLE}
)


class StrategyFamily(str, Enum):
    """Allowed defined-risk families (spec section 21).

    Every member must be expressible with a finite, pre-computable maximum loss
    and must never require SPY share ownership.
    """

    # 21.1 bullish
    BULL_CALL_DEBIT_SPREAD = "BullCallDebitSpread"
    BULL_PUT_CREDIT_SPREAD = "BullPutCreditSpread"
    CALL_BUTTERFLY = "CallButterfly"
    BROKEN_WING_CALL_BUTTERFLY = "BrokenWingCallButterfly"
    CALL_BACKSPREAD = "CallBackspread"
    BULLISH_CALENDAR = "BullishCalendar"
    BULLISH_DIAGONAL = "BullishDiagonal"

    # 21.2 bearish
    BEAR_PUT_DEBIT_SPREAD = "BearPutDebitSpread"
    BEAR_CALL_CREDIT_SPREAD = "BearCallCreditSpread"
    PUT_BUTTERFLY = "PutButterfly"
    BROKEN_WING_PUT_BUTTERFLY = "BrokenWingPutButterfly"
    PUT_BACKSPREAD = "PutBackspread"
    BEARISH_CALENDAR = "BearishCalendar"
    BEARISH_DIAGONAL = "BearishDiagonal"

    # 21.3 neutral / range
    IRON_CONDOR = "IronCondor"
    IRON_BUTTERFLY = "IronButterfly"
    BROKEN_WING_CONDOR = "BrokenWingCondor"
    ASYMMETRIC_CONDOR = "AsymmetricCondor"
    DOUBLE_CALENDAR = "DoubleCalendar"

    # 21.4 volatility expansion
    LONG_STRADDLE = "LongStraddle"
    LONG_STRANGLE = "LongStrangle"

    @property
    def is_calendarized(self) -> bool:
        """True when the family intentionally spans more than one expiration."""
        return self in _CALENDARIZED


_CALENDARIZED: frozenset[StrategyFamily] = frozenset(
    {
        StrategyFamily.BULLISH_CALENDAR,
        StrategyFamily.BEARISH_CALENDAR,
        StrategyFamily.BULLISH_DIAGONAL,
        StrategyFamily.BEARISH_DIAGONAL,
        StrategyFamily.DOUBLE_CALENDAR,
    }
)


class DealerSignConvention(str, Enum):
    """Dealer positioning assumptions (spec 8.1).

    Structural inference from open interest is imperfect, so every exposure is
    computed under each convention and the disagreement is carried forward as a
    confidence penalty rather than hidden behind a single number.
    """

    CUSTOMER_LONG = "customer_long"
    CUSTOMER_SHORT = "customer_short"
    CALLS_POSITIVE_PUTS_NEGATIVE = "calls_positive_puts_negative"
    UNSIGNED = "unsigned"
    FLOW_ADJUSTED = "flow_adjusted"


class OrderStatus(str, Enum):
    PENDING_NEW = "pending_new"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderIntent(str, Enum):
    OPEN = "open"
    CLOSE = "close"


class ExitReason(str, Enum):
    PROFIT_TARGET = "profit_target"
    RISK_STOP = "risk_stop"
    TIME_STOP = "time_stop"
    STATE_INVALIDATION = "state_invalidation"
    END_OF_SESSION = "end_of_session"
    ASSIGNMENT_RISK = "assignment_risk"
    KILL_SWITCH = "kill_switch"
    EXPIRATION = "expiration"


class KillSwitchReason(str, Enum):
    """Kill switches required by spec section 42."""

    MANUAL = "manual"
    DATA_QUALITY = "data_quality"
    BROKER_STATE = "broker_state"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    SLIPPAGE = "slippage"
    CALIBRATION = "calibration"
    MODEL_DISAGREEMENT = "model_disagreement"
    ASSIGNMENT_RISK = "assignment_risk"
    EVENT = "event"
    MARKET_HALT = "market_halt"
    CONSECUTIVE_LOSSES = "consecutive_losses"


class TradeDirection(int, Enum):
    """Aggressor side inferred from trade price versus the prevailing quote."""

    BUY = 1
    UNKNOWN = 0
    SELL = -1
