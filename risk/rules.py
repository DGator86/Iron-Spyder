from __future__ import annotations

from dataclasses import dataclass

from data.schemas import RiskDecision, StrategyCandidate


@dataclass
class RiskLimits:
    max_per_trade_loss: float = 500.0
    max_open_positions: int = 10


class KillSwitch:
    def __init__(self) -> None:
        self.enabled = False

    def set(self, enabled: bool) -> None:
        self.enabled = enabled


def approve_trade(candidate: StrategyCandidate, open_positions: int, limits: RiskLimits, kill_switch: KillSwitch) -> RiskDecision:
    reasons: list[str] = []
    if kill_switch.enabled:
        reasons.append("Kill switch enabled")
    if candidate.max_loss > limits.max_per_trade_loss:
        reasons.append("Per-trade max loss exceeded")
    if open_positions >= limits.max_open_positions:
        reasons.append("Max open positions reached")
    if candidate.rejection_reasons:
        reasons.extend(candidate.rejection_reasons)
    return RiskDecision(approved=not reasons, reasons=reasons)
