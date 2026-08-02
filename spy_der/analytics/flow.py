"""Options trade-flow analytics (spec section 10).

Spec 10 is explicit that raw call volume is not bullish and raw put volume is
not bearish. Nothing here maps volume to direction. Flow is aggregated by
*aggressor-signed premium* and converted into Greek-space demand, so a large
put purchase and an equally large put sale cancel rather than both registering
as "put activity".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from spy_der.analytics import pricing
from spy_der.domain.enums import OptionRight, TradeDirection
from spy_der.domain.market import OptionTrade


@dataclass(frozen=True)
class FlowMetrics:
    """Aggregate signed-flow features (spec section 10)."""

    call_signed_premium: float
    put_signed_premium: float
    net_delta_flow: float
    net_gamma_flow: float
    net_vega_flow: float
    trade_count: int
    unknown_direction_fraction: float
    sweep_premium: float
    block_premium: float
    strike_concentration: float
    expiry_concentration: float
    by_strike: dict[float, float] = field(default_factory=dict)
    by_expiry: dict[datetime, float] = field(default_factory=dict)

    @property
    def net_signed_premium(self) -> float:
        return self.call_signed_premium + self.put_signed_premium

    @property
    def directional_tilt(self) -> float:
        """Call minus put signed premium, scaled to ``[-1, 1]``.

        Note this is a tilt in *premium paid*, not a directional forecast: buying
        calls and buying puts both spend premium, and only the imbalance is
        informative.
        """
        total = abs(self.call_signed_premium) + abs(self.put_signed_premium)
        if total <= 0.0:
            return 0.0
        return float(
            np.clip((self.call_signed_premium - self.put_signed_premium) / total, -1.0, 1.0)
        )

    @property
    def is_informative(self) -> bool:
        """False when too much flow could not be signed to be worth reading."""
        return self.trade_count >= 5 and self.unknown_direction_fraction < 0.6


def compute_flow(
    trades: Sequence[OptionTrade],
    *,
    spot: float,
    rate: float = 0.04,
    div_yield: float = 0.013,
    now: datetime | None = None,
    window: timedelta | None = None,
    block_threshold: int = 100,
) -> FlowMetrics:
    """Aggregate signed flow and its Greek-space demand."""
    if window is not None and now is not None:
        cutoff = now - window
        trades = [t for t in trades if t.timestamp >= cutoff]

    if not trades:
        return FlowMetrics(
            call_signed_premium=0.0,
            put_signed_premium=0.0,
            net_delta_flow=0.0,
            net_gamma_flow=0.0,
            net_vega_flow=0.0,
            trade_count=0,
            unknown_direction_fraction=0.0,
            sweep_premium=0.0,
            block_premium=0.0,
            strike_concentration=0.0,
            expiry_concentration=0.0,
        )

    call_premium = 0.0
    put_premium = 0.0
    delta_flow = 0.0
    gamma_flow = 0.0
    vega_flow = 0.0
    sweep_premium = 0.0
    block_premium = 0.0
    unknown = 0
    by_strike: dict[float, float] = {}
    by_expiry: dict[datetime, float] = {}

    for trade in trades:
        signed = trade.signed_premium
        if trade.direction is TradeDirection.UNKNOWN:
            unknown += 1

        if trade.right is OptionRight.CALL:
            call_premium += signed
        else:
            put_premium += signed

        if trade.is_sweep:
            sweep_premium += signed
        if trade.size >= block_threshold:
            block_premium += signed

        by_strike[trade.strike] = by_strike.get(trade.strike, 0.0) + abs(signed)
        by_expiry[trade.expiration] = by_expiry.get(trade.expiration, 0.0) + abs(signed)

        tau = max(0.0, (trade.expiration - trade.timestamp).total_seconds()) / (
            365.0 * 24.0 * 3600.0
        )
        iv = _implied_vol_or_default(trade, spot, tau, rate, div_yield)
        contracts = float(trade.direction.value) * trade.size * 100.0
        delta_flow += contracts * float(
            pricing.delta(spot, trade.strike, tau, rate, div_yield, iv, trade.right)
        )
        gamma_flow += contracts * float(
            pricing.gamma(spot, trade.strike, tau, rate, div_yield, iv)
        )
        vega_flow += contracts * float(
            pricing.vega(spot, trade.strike, tau, rate, div_yield, iv)
        )

    return FlowMetrics(
        call_signed_premium=call_premium,
        put_signed_premium=put_premium,
        net_delta_flow=delta_flow,
        net_gamma_flow=gamma_flow,
        net_vega_flow=vega_flow,
        trade_count=len(trades),
        unknown_direction_fraction=unknown / len(trades),
        sweep_premium=sweep_premium,
        block_premium=block_premium,
        strike_concentration=_herfindahl(by_strike.values()),
        expiry_concentration=_herfindahl(by_expiry.values()),
        by_strike=by_strike,
        by_expiry=by_expiry,
    )


def _implied_vol_or_default(
    trade: OptionTrade, spot: float, tau: float, rate: float, div_yield: float
) -> float:
    """Back out IV from the traded price, defaulting when it cannot be inverted."""
    iv = pricing.implied_volatility(
        trade.price, spot, trade.strike, tau, rate, div_yield, trade.right
    )
    return iv if iv is not None else 0.15


def _herfindahl(values: object) -> float:
    """Herfindahl concentration in ``[0, 1]``; ``1`` means a single bucket."""
    array = np.asarray(list(values), dtype=np.float64)  # type: ignore[call-overload]
    total = float(np.sum(array))
    if total <= 0.0 or array.size == 0:
        return 0.0
    shares = array / total
    return float(np.sum(shares * shares))


@dataclass
class FlowTracker:
    """Rolling flow state for acceleration, persistence, and reversal.

    These are inherently temporal features, so like :class:`WallTracker` this is
    owned by the pipeline and updated once per snapshot.
    """

    history: list[FlowMetrics] = field(default_factory=list)
    max_history: int = 64

    def update(self, metrics: FlowMetrics) -> None:
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    @property
    def acceleration(self) -> float:
        """Change in net signed premium between the last two observations."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].net_signed_premium - self.history[-2].net_signed_premium

    @property
    def persistence(self) -> float:
        """Fraction of recent observations sharing the current flow sign."""
        if len(self.history) < 3:
            return 0.0
        recent = self.history[-8:]
        current = np.sign(recent[-1].net_signed_premium)
        if current == 0.0:
            return 0.0
        matching = sum(1 for m in recent if np.sign(m.net_signed_premium) == current)
        return matching / len(recent)

    @property
    def reversal(self) -> bool:
        """True when signed flow has just flipped sign."""
        if len(self.history) < 2:
            return False
        prev = np.sign(self.history[-2].net_signed_premium)
        cur = np.sign(self.history[-1].net_signed_premium)
        return bool(prev != 0.0 and cur != 0.0 and prev != cur)

    def divergence(self, price_change: float) -> float:
        """Signed flow against price: positive means flow and price disagree."""
        if not self.history:
            return 0.0
        flow_sign = np.sign(self.history[-1].net_signed_premium)
        price_sign = np.sign(price_change)
        if flow_sign == 0.0 or price_sign == 0.0:
            return 0.0
        return 1.0 if flow_sign != price_sign else 0.0
