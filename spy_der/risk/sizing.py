"""Position sizing (spec section 28.4).

::

    R_base  = Equity * f
    R_trade = R_base * Confidence * Liquidity * Stability * CalibrationQuality
    N       = floor(R_trade / MaxLossPerContract)

Hard caps override every formula. The system must never average down,
martingale, increase size after a loss, add to a losing position, or alter the
maximum permitted loss after entry — so nothing here reads realized P&L, and
sizing is computed once, at entry, from forward-looking quality terms only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spy_der.domain.strategy import StrategyCandidate
from spy_der.risk.limits import RiskConfig


@dataclass(frozen=True)
class SizingResult:
    contracts: int
    base_risk: float
    adjusted_risk: float
    risk_per_contract: float
    binding_constraint: str

    @property
    def total_risk(self) -> float:
        return self.contracts * self.risk_per_contract

    @property
    def is_tradeable(self) -> bool:
        return self.contracts >= 1


def size_position(
    candidate: StrategyCandidate,
    *,
    equity: float,
    confidence: float,
    stability: float,
    calibration_quality: float,
    config: RiskConfig,
    available_risk: float | None = None,
) -> SizingResult:
    """Compute the contract count for ``candidate``."""
    limits = config.trade
    risk_per_contract = candidate.max_loss
    if risk_per_contract <= 0.0 or not math.isfinite(risk_per_contract):
        return SizingResult(
            contracts=0,
            base_risk=0.0,
            adjusted_risk=0.0,
            risk_per_contract=risk_per_contract,
            binding_constraint="undefined risk per contract",
        )

    base_risk = max(0.0, equity) * config.base_risk_fraction

    quality = (
        float(np.clip(confidence, 0.0, 1.0))
        * float(np.clip(candidate.liquidity.score, 0.0, 1.0))
        * float(np.clip(stability, 0.0, 1.0))
        * float(np.clip(calibration_quality, 0.0, 1.0))
    )
    adjusted_risk = base_risk * quality

    # Hard caps, applied after the formula and never softened by it.
    caps: list[tuple[str, float]] = [
        ("base risk budget", adjusted_risk),
        ("account fraction cap", equity * limits.max_account_fraction_at_risk),
        ("absolute dollar cap", limits.max_dollar_risk),
    ]
    if available_risk is not None:
        caps.append(("remaining portfolio risk", max(0.0, available_risk)))

    binding_name, allowed = min(caps, key=lambda kv: kv[1])
    contracts = int(allowed // risk_per_contract)

    if contracts > limits.max_contracts:
        contracts = limits.max_contracts
        binding_name = "max contracts cap"

    return SizingResult(
        contracts=max(0, contracts),
        base_risk=base_risk,
        adjusted_risk=adjusted_risk,
        risk_per_contract=risk_per_contract,
        binding_constraint=binding_name if contracts >= 1 else f"{binding_name} (below one contract)",
    )
