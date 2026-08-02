"""Expected utility and the no-trade competitor (spec sections 3 and 27).

::

    U(S) = EV(S)
         - l1 * CVaR(S)
         - l2 * ExecutionRisk(S)
         - l3 * ModelUncertainty(S)
         - l4 * LiquidityPenalty(S)
         - l5 * ComplexityPenalty(S)
         - l6 * CapitalPenalty(S)

Every term is expressed in **dollars per contract** so the subtraction is
meaningful. Mixing a dollar EV with a dimensionless liquidity score would make
the lambdas depend on account size and strategy scale, and they would stop
being comparable across candidates — which is the one thing a ranking function
must get right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spy_der.domain.strategy import NO_TRADE_ID, LiquidityProfile


@dataclass(frozen=True)
class UtilityWeights:
    """The six lambdas of spec section 3.

    Defaults are risk-averse on purpose. Spec 45 states the system's most
    important competency is rejecting trades whose apparent edge does not
    survive costs and uncertainty, so the penalties are set to bite.

    ``cvar`` needs care. For a *defined-risk* structure the 95% conditional
    loss is already close to the maximum loss, so the penalty is roughly
    ``cvar * 0.8 * max_loss`` while expected value runs a few percent of
    max loss. A weight near ``0.35`` would therefore demand a return on risk
    above 30% before any trade cleared, which rejects everything rather than
    discriminating. At ``0.05`` the penalty sits on the same scale as expected
    value and does what it is meant to: separate two candidates with equal EV
    but different tail shapes.
    """

    cvar: float = 0.05
    execution_risk: float = 1.0
    model_uncertainty: float = 1.0
    liquidity: float = 1.0
    complexity: float = 1.0
    capital: float = 1.0

    def scaled(self, factor: float) -> UtilityWeights:
        return UtilityWeights(
            cvar=self.cvar * factor,
            execution_risk=self.execution_risk * factor,
            model_uncertainty=self.model_uncertainty * factor,
            liquidity=self.liquidity * factor,
            complexity=self.complexity * factor,
            capital=self.capital * factor,
        )


@dataclass(frozen=True)
class PenaltyScales:
    """Constants converting dimensionless quality scores into dollar penalties."""

    liquidity_fraction_of_risk: float = 0.06
    """Illiquidity penalty at zero liquidity, as a fraction of maximum loss."""

    complexity_fraction_of_risk: float = 0.015
    """Penalty per leg beyond two, as a fraction of maximum loss."""

    capital_cost_annual: float = 0.05
    """Opportunity cost of committed capital, annualized."""

    unfilled_penalty_fraction: float = 0.25
    """Share of entry slippage charged again for fill uncertainty."""


@dataclass(frozen=True)
class UtilityBreakdown:
    """Decomposed utility so a rejection can be attributed to a cause."""

    expected_value: float
    cvar_penalty: float
    execution_penalty: float
    uncertainty_penalty: float
    liquidity_penalty: float
    complexity_penalty: float
    capital_penalty: float

    @property
    def total_penalty(self) -> float:
        return (
            self.cvar_penalty
            + self.execution_penalty
            + self.uncertainty_penalty
            + self.liquidity_penalty
            + self.complexity_penalty
            + self.capital_penalty
        )

    @property
    def utility(self) -> float:
        return self.expected_value - self.total_penalty

    def as_dict(self) -> dict[str, float]:
        return {
            "expected_value": self.expected_value,
            "penalty_cvar": self.cvar_penalty,
            "penalty_execution": self.execution_penalty,
            "penalty_uncertainty": self.uncertainty_penalty,
            "penalty_liquidity": self.liquidity_penalty,
            "penalty_complexity": self.complexity_penalty,
            "penalty_capital": self.capital_penalty,
            "utility": self.utility,
        }


def compute_utility(
    *,
    expected_value: float,
    cvar: float,
    max_loss: float,
    expected_slippage: float,
    fill_probability: float,
    confidence: float,
    liquidity: LiquidityProfile,
    leg_count: int,
    capital_usage: float,
    holding_minutes: float,
    weights: UtilityWeights | None = None,
    scales: PenaltyScales | None = None,
) -> UtilityBreakdown:
    """Evaluate ``U(S)`` for one candidate, per contract."""
    w = weights or UtilityWeights()
    s = scales or PenaltyScales()

    # Execution risk: the slippage actually expected, plus a surcharge for the
    # chance the order does not fill at the requested limit and has to chase.
    execution = expected_slippage * (
        1.0 + s.unfilled_penalty_fraction * (1.0 - float(np.clip(fill_probability, 0.0, 1.0)))
    )

    # Model uncertainty scales the *edge*, not the risk: a confident forecast
    # earns its EV, an unconfident one has its EV discounted toward zero.
    uncertainty = (1.0 - float(np.clip(confidence, 0.0, 1.0))) * abs(expected_value)

    illiquidity = (1.0 - float(np.clip(liquidity.score, 0.0, 1.0)))
    liquidity_penalty = illiquidity * s.liquidity_fraction_of_risk * max_loss

    complexity_penalty = (
        max(0, leg_count - 2) * s.complexity_fraction_of_risk * max_loss
    )

    years_held = max(holding_minutes, 1.0) / (252.0 * 390.0)
    capital_penalty = capital_usage * s.capital_cost_annual * years_held

    return UtilityBreakdown(
        expected_value=expected_value,
        cvar_penalty=w.cvar * max(0.0, cvar),
        execution_penalty=w.execution_risk * execution,
        uncertainty_penalty=w.model_uncertainty * uncertainty,
        liquidity_penalty=w.liquidity * liquidity_penalty,
        complexity_penalty=w.complexity * complexity_penalty,
        capital_penalty=w.capital * capital_penalty,
    )


def normalized_score(
    *,
    expected_value: float,
    max_loss: float,
    confidence: float,
    liquidity: float,
    regime_fit: float,
    fill_probability: float,
    risk_penalty: float = 0.0,
) -> float:
    """The alternative normalized score of spec section 3.

    Reported alongside utility as a scale-free cross-check. Ranking is done on
    utility; this exists so a candidate that looks good only because it is
    large is visible as such.
    """
    if max_loss <= 0.0:
        return 0.0
    return (
        (expected_value / max_loss)
        * float(np.clip(confidence, 0.0, 1.0))
        * float(np.clip(liquidity, 0.0, 1.0))
        * float(np.clip(regime_fit, 0.0, 1.0))
        * float(np.clip(fill_probability, 0.0, 1.0))
        - risk_penalty
    )


@dataclass(frozen=True)
class NoTradeOption:
    """The explicit do-nothing competitor (spec section 27).

    Not a fallback for an empty candidate list — a first-class option that must
    be beaten by a margin. ``epsilon`` is a token cost for inactivity, kept
    tiny so it only breaks ties when a clearly positive trade exists.
    """

    epsilon: float = 0.0
    margin: float = 10.0
    """Dollars of utility a trade must beat no-trade by, per contract.

    Prevents acting on trivial modelled advantages that are inside the noise of
    the forecast itself. Ten dollars per contract is roughly twice the
    round-trip cost of a typical two-leg SPY structure, so a candidate must
    clear its own execution costs with room to spare.
    """

    strategy_id: str = NO_TRADE_ID

    @property
    def utility(self) -> float:
        return -self.epsilon

    def beaten_by(self, utility: float) -> bool:
        return utility > self.utility + self.margin

    def shortfall(self, utility: float) -> float:
        """How far a candidate's utility falls short of clearing the bar."""
        return (self.utility + self.margin) - utility
