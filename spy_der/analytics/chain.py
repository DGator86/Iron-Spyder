"""Vectorized chain views and model Greeks.

Provider-supplied Greeks are used only when explicitly requested. By default
every Greek is recomputed from the contract's implied volatility with a single
consistent model, because mixing vendor Greeks (each with its own rate,
dividend and time-to-expiry convention) into an aggregate exposure produces a
number that means nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from spy_der.analytics import pricing
from spy_der.domain.market import MarketSnapshot, OptionQuote

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ChainArrays:
    """Columnar view of an option chain plus its model Greeks."""

    quotes: tuple[OptionQuote, ...]
    strike: FloatArray
    tau: FloatArray
    iv: FloatArray
    open_interest: FloatArray
    volume: FloatArray
    bid: FloatArray
    ask: FloatArray
    mid: FloatArray
    is_call: FloatArray
    """``+1.0`` for calls, ``-1.0`` for puts — broadcastable into pricing."""

    spot: float
    rate: float
    div_yield: float

    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    vanna: FloatArray
    charm: FloatArray

    def __len__(self) -> int:
        return len(self.quotes)

    @property
    def call_mask(self) -> NDArray[np.bool_]:
        return self.is_call > 0.0

    @property
    def put_mask(self) -> NDArray[np.bool_]:
        return self.is_call < 0.0

    def gamma_at(self, spot: float | FloatArray) -> FloatArray:
        """Recompute gamma at a hypothetical spot, holding each strike's IV fixed.

        This is the sticky-strike assumption. It is the conventional choice for
        a gamma profile and is stated here explicitly because the alternative
        (sticky-delta) produces a visibly different flip point.
        """
        return pricing.gamma(spot, self.strike, self.tau, self.rate, self.div_yield, self.iv)

    def delta_at(self, spot: float | FloatArray) -> FloatArray:
        return pricing.delta(
            spot, self.strike, self.tau, self.rate, self.div_yield, self.iv, self.is_call
        )


def build_chain_arrays(
    quotes: Sequence[OptionQuote],
    *,
    spot: float,
    rate: float,
    div_yield: float,
    use_provider_greeks: bool = False,
) -> ChainArrays:
    """Build a :class:`ChainArrays` from raw quotes."""
    if not quotes:
        raise ValueError("cannot build chain arrays from an empty chain")

    quotes = tuple(quotes)
    strike = np.array([q.strike for q in quotes], dtype=np.float64)
    tau = np.array([q.tau for q in quotes], dtype=np.float64)
    iv = np.array([q.implied_volatility for q in quotes], dtype=np.float64)
    open_interest = np.array([float(q.open_interest) for q in quotes], dtype=np.float64)
    volume = np.array([float(q.volume) for q in quotes], dtype=np.float64)
    bid = np.array([q.bid for q in quotes], dtype=np.float64)
    ask = np.array([q.ask for q in quotes], dtype=np.float64)
    mid = 0.5 * (bid + ask)
    is_call = np.array([1.0 if q.right.value == "call" else -1.0 for q in quotes], dtype=np.float64)

    if use_provider_greeks:  # pragma: no cover - opt-in path for vendor parity checks
        raise NotImplementedError("provider Greeks are not carried on OptionQuote")

    return ChainArrays(
        quotes=quotes,
        strike=strike,
        tau=tau,
        iv=iv,
        open_interest=open_interest,
        volume=volume,
        bid=bid,
        ask=ask,
        mid=mid,
        is_call=is_call,
        spot=spot,
        rate=rate,
        div_yield=div_yield,
        delta=pricing.delta(spot, strike, tau, rate, div_yield, iv, is_call),
        gamma=pricing.gamma(spot, strike, tau, rate, div_yield, iv),
        vega=pricing.vega(spot, strike, tau, rate, div_yield, iv),
        theta=pricing.theta(spot, strike, tau, rate, div_yield, iv, is_call),
        vanna=pricing.vanna(spot, strike, tau, rate, div_yield, iv),
        charm=pricing.charm(spot, strike, tau, rate, div_yield, iv, is_call),
    )


def chain_from_snapshot(
    snapshot: MarketSnapshot, *, expiration: object | None = None
) -> ChainArrays:
    """Convenience constructor honouring the snapshot's rate and dividend yield."""
    quotes = (
        snapshot.option_chain
        if expiration is None
        else tuple(q for q in snapshot.option_chain if q.expiration == expiration)
    )
    return build_chain_arrays(
        quotes,
        spot=snapshot.spot,
        rate=snapshot.risk_free_rate,
        div_yield=snapshot.dividend_yield,
    )


def liquidity_score(arrays: ChainArrays) -> float:
    """Chain-level liquidity in ``[0, 1]``.

    Combines relative spread, quoted depth and open interest. Spread dominates
    because it is the cost actually paid on entry and exit.
    """
    mid = np.maximum(arrays.mid, 1e-6)
    spread_ratio = np.clip((arrays.ask - arrays.bid) / mid, 0.0, 2.0)
    spread_term = np.clip(1.0 - spread_ratio / 0.5, 0.0, 1.0)
    oi_term = np.clip(np.log1p(arrays.open_interest) / np.log1p(5_000.0), 0.0, 1.0)
    vol_term = np.clip(np.log1p(arrays.volume) / np.log1p(2_000.0), 0.0, 1.0)
    score = 0.6 * spread_term + 0.25 * oi_term + 0.15 * vol_term
    return float(np.clip(np.mean(score), 0.0, 1.0))
