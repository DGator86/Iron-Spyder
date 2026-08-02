"""Dealer exposure estimates: GEX, DEX, VEX, CEX (spec section 8).

Spec 2.6 is the governing constraint here. Open interest does not reveal who is
long and who is short, so a single "the dealers are short 4 billion gamma"
number is a fabrication dressed as a measurement. Every exposure is therefore
computed under *all* sign conventions, and the spread between them is reported
as :attr:`ExposureBundle.agreement`, which flows into the confidence term and
shrinks position size when the conventions disagree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from spy_der.analytics.chain import ChainArrays
from spy_der.domain.enums import DealerSignConvention
from spy_der.domain.market import FloatVector, OptionTrade

FloatArray: TypeAlias = NDArray[np.float64]

#: Conventions evaluated on every pass. UNSIGNED measures concentration rather
#: than direction, so it is excluded from the agreement statistic.
DIRECTIONAL_CONVENTIONS: tuple[DealerSignConvention, ...] = (
    DealerSignConvention.CUSTOMER_LONG,
    DealerSignConvention.CUSTOMER_SHORT,
    DealerSignConvention.CALLS_POSITIVE_PUTS_NEGATIVE,
    DealerSignConvention.FLOW_ADJUSTED,
)

#: Conventions that carry information about *this* chain, used for the
#: agreement statistic.
#:
#: ``CUSTOMER_SHORT`` is deliberately excluded. It is the exact mirror of
#: ``CUSTOMER_LONG`` — every sign flipped — so including both guarantees a
#: dispersion of roughly ``|GEX|`` and an agreement near zero on every snapshot,
#: regardless of the data. That measures the choice of priors, not the market.
#: The remaining three differ in ways the chain can actually resolve: a
#: uniform prior, a call/put split, and one inferred from observed flow.
INFORMATIVE_CONVENTIONS: tuple[DealerSignConvention, ...] = (
    DealerSignConvention.CUSTOMER_LONG,
    DealerSignConvention.CALLS_POSITIVE_PUTS_NEGATIVE,
    DealerSignConvention.FLOW_ADJUSTED,
)


def dealer_signs(
    arrays: ChainArrays,
    convention: DealerSignConvention,
    *,
    flow_signs: FloatArray | None = None,
) -> FloatArray:
    """Per-contract dealer sign multiplier under ``convention``."""
    if convention is DealerSignConvention.CUSTOMER_LONG:
        # Customers own the options, dealers are short them.
        return np.full(len(arrays), -1.0)
    if convention is DealerSignConvention.CUSTOMER_SHORT:
        return np.full(len(arrays), 1.0)
    if convention is DealerSignConvention.CALLS_POSITIVE_PUTS_NEGATIVE:
        return arrays.is_call.copy()
    if convention is DealerSignConvention.UNSIGNED:
        return np.full(len(arrays), 1.0)
    if convention is DealerSignConvention.FLOW_ADJUSTED:
        if flow_signs is None:
            return np.full(len(arrays), -1.0)
        return flow_signs
    raise ValueError(f"unhandled convention {convention!r}")  # pragma: no cover


def flow_implied_signs(arrays: ChainArrays, trades: Sequence[OptionTrade]) -> FloatArray:
    """Infer a dealer sign per contract from observed aggressor flow.

    Net customer buying implies the dealer is short that contract (``-1``); net
    customer selling implies the dealer is long it (``+1``). Contracts with no
    observed flow fall back to the customer-long prior, which is the
    conventional default for index options.
    """
    signed: dict[str, float] = {}
    for trade in trades:
        signed[trade.contract_symbol] = (
            signed.get(trade.contract_symbol, 0.0) + trade.signed_premium
        )

    out = np.full(len(arrays), -1.0)
    for index, quote in enumerate(arrays.quotes):
        net = signed.get(quote.contract_symbol)
        if net is None or net == 0.0:
            continue
        out[index] = -1.0 if net > 0.0 else 1.0
    return out


@dataclass(frozen=True)
class ExposureSet:
    """Aggregate exposures under one dealer-sign convention."""

    convention: DealerSignConvention
    gex: float
    dex: float
    vex: float
    cex: float
    gex_by_strike: dict[float, float]
    dex_by_strike: dict[float, float]


@dataclass(frozen=True)
class ExposureBundle:
    """Exposures under every convention, plus their agreement."""

    per_convention: dict[DealerSignConvention, ExposureSet]
    agreement: float
    """``1 - normalized dispersion`` of GEX across directional conventions."""

    unsigned_gex_by_strike: dict[float, float]
    spot: float

    @property
    def primary(self) -> ExposureSet:
        """The customer-long convention, used as the reporting default."""
        return self.per_convention[DealerSignConvention.CUSTOMER_LONG]

    def gex(self, convention: DealerSignConvention) -> float:
        return self.per_convention[convention].gex

    @property
    def gex_spread(self) -> float:
        """Absolute spread of GEX across directional conventions."""
        values = [self.per_convention[c].gex for c in DIRECTIONAL_CONVENTIONS]
        return float(max(values) - min(values))

    @property
    def sign_consensus(self) -> float:
        """Fraction of informative conventions agreeing on the sign of GEX.

        Returns a value in ``[0.5, 1.0]``; ``0.5`` means the conventions are
        evenly split and the sign of dealer gamma is genuinely unknown.
        """
        signs = [np.sign(self.per_convention[c].gex) for c in INFORMATIVE_CONVENTIONS]
        positives = sum(1 for s in signs if s > 0)
        negatives = sum(1 for s in signs if s < 0)
        total = max(1, positives + negatives)
        return max(positives, negatives) / total


def _aggregate_by_strike(strikes: FloatArray, values: FloatArray) -> dict[float, float]:
    out: dict[float, float] = {}
    for strike, value in zip(strikes.tolist(), values.tolist()):
        out[strike] = out.get(strike, 0.0) + value
    return out


def gamma_exposure(arrays: ChainArrays, signs: FloatArray, *, spot: float | None = None) -> FloatArray:
    """``GEX_i = Gamma_i * OI_i * 100 * S^2 * 0.01 * Sign_i`` (spec 8.1).

    The ``S^2 * 0.01`` factor converts unit gamma into dollars of dealer hedging
    demand per 1% move in SPY.
    """
    s = arrays.spot if spot is None else spot
    return arrays.gamma_at(s) * arrays.open_interest * 100.0 * s * s * 0.01 * signs


def delta_exposure(arrays: ChainArrays, signs: FloatArray) -> FloatArray:
    """``DEX_i = Delta_i * OI_i * 100 * S * Sign_i`` (spec 8.2)."""
    return arrays.delta * arrays.open_interest * 100.0 * arrays.spot * signs


def vanna_exposure(arrays: ChainArrays, signs: FloatArray) -> FloatArray:
    """``VEX_i = Vanna_i * OI_i * 100 * S * Sign_i`` (spec 8.3)."""
    return arrays.vanna * arrays.open_interest * 100.0 * arrays.spot * signs


def charm_exposure(arrays: ChainArrays, signs: FloatArray) -> FloatArray:
    """``CEX_i = Charm_i * OI_i * 100 * S * Sign_i`` (spec 8.4)."""
    return arrays.charm * arrays.open_interest * 100.0 * arrays.spot * signs


def compute_exposures(
    arrays: ChainArrays, *, trades: Sequence[OptionTrade] = ()
) -> ExposureBundle:
    """Compute exposures under every dealer-sign convention."""
    flow_signs = flow_implied_signs(arrays, trades) if trades else None

    per_convention: dict[DealerSignConvention, ExposureSet] = {}
    for convention in (*DIRECTIONAL_CONVENTIONS, DealerSignConvention.UNSIGNED):
        signs = dealer_signs(arrays, convention, flow_signs=flow_signs)
        if convention is DealerSignConvention.UNSIGNED:
            gex_values = np.abs(gamma_exposure(arrays, np.ones(len(arrays))))
            dex_values = np.abs(delta_exposure(arrays, np.ones(len(arrays))))
            vex_values = np.abs(vanna_exposure(arrays, np.ones(len(arrays))))
            cex_values = np.abs(charm_exposure(arrays, np.ones(len(arrays))))
        else:
            gex_values = gamma_exposure(arrays, signs)
            dex_values = delta_exposure(arrays, signs)
            vex_values = vanna_exposure(arrays, signs)
            cex_values = charm_exposure(arrays, signs)

        per_convention[convention] = ExposureSet(
            convention=convention,
            gex=float(np.sum(gex_values)),
            dex=float(np.sum(dex_values)),
            vex=float(np.sum(vex_values)),
            cex=float(np.sum(cex_values)),
            gex_by_strike=_aggregate_by_strike(arrays.strike, gex_values),
            dex_by_strike=_aggregate_by_strike(arrays.strike, dex_values),
        )

    # Distinct from the per-contract `signs` array used inside the loop above:
    # this is one sign per convention, for the agreement statistic.
    gex_signs = [float(np.sign(per_convention[c].gex)) for c in INFORMATIVE_CONVENTIONS]
    positives = sum(1 for s in gex_signs if s > 0)
    negatives = sum(1 for s in gex_signs if s < 0)
    consensus = max(positives, negatives) / max(1, positives + negatives)

    return ExposureBundle(
        per_convention=per_convention,
        agreement=dealer_agreement(
            [per_convention[c].gex for c in INFORMATIVE_CONVENTIONS],
            sign_consensus=consensus,
        ),
        unsigned_gex_by_strike=per_convention[DealerSignConvention.UNSIGNED].gex_by_strike,
        spot=arrays.spot,
    )


def dealer_agreement(values: FloatVector, *, sign_consensus: float | None = None) -> float:
    """``1 - NormalizedDispersion(...)`` across conventions (spec 8.1).

    Combines two things a decision actually depends on: whether the conventions
    agree on the *sign* of dealer gamma, and how far apart their magnitudes are.
    Magnitude dispersion alone is a poor summary, because two conventions can
    agree that dealers are short gamma while differing by a factor of two — a
    disagreement about size, not about direction.

    Returns ``1.0`` for unanimity and values near zero when the structural read
    is unusable. This feeds the confidence term, so it directly shrinks position
    size and pushes the decision toward no-trade.
    """
    if len(values) < 2:
        return 1.0
    array = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(array)))
    magnitude_term = (
        1.0 if scale <= 0.0 else float(np.clip(1.0 - float(np.std(array)) / scale, 0.0, 1.0))
    )
    if sign_consensus is None:
        return magnitude_term

    # Rescale consensus from its natural [0.5, 1] range onto [0, 1].
    direction_term = float(np.clip(2.0 * (sign_consensus - 0.5), 0.0, 1.0))
    return float(np.clip(0.6 * direction_term + 0.4 * magnitude_term, 0.0, 1.0))
