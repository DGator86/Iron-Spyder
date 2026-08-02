from data.schemas import StrategyCandidate
from optimizer.optimizer import OptimizerConfig, rank_candidates
from risk.rules import KillSwitch, RiskLimits, approve_trade


def test_risk_limits() -> None:
    cand = StrategyCandidate(name="x", legs=[], max_profit=100, max_loss=1000, breakevens=[], expected_return_on_risk=0.1, liquidity=1.0, expected_utility=0.1)
    decision = approve_trade(cand, open_positions=0, limits=RiskLimits(max_per_trade_loss=500), kill_switch=KillSwitch())
    assert not decision.approved


def test_no_trade_behavior() -> None:
    best, ranked = rank_candidates(spot=550, market_state="Strong pin", liquidity=0.01, config=OptimizerConfig(min_liquidity=0.5, min_expected_return_on_risk=1.0))
    assert best == "no_trade"
    assert ranked == []
