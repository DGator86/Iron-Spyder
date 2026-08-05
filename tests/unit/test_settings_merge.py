"""Nested YAML overrides for mode settings."""

from __future__ import annotations

from spy_der.config.settings import load_settings
from spy_der.risk.limits import TradeLimits


def test_backtest_mode_tunes_replay_gates():
    settings = load_settings("backtest")
    assert isinstance(settings.risk.trade, TradeLimits)
    assert settings.risk.trade.min_confidence == 0.05
    assert settings.optimizer.min_fill_probability == 0.15
    assert settings.optimizer.no_trade.margin == 1.0
    assert settings.costs.fill_probability_floor == 0.15
    # Paper is open-ready / aggressive on a $1k book; live stays the strict baseline.
    paper = load_settings("paper")
    assert paper.starting_equity == 1000.0
    assert paper.risk.trade.min_confidence == 0.05
    assert paper.optimizer.no_trade.margin == 0.5
    live = load_settings("live")
    assert live.risk.trade.min_confidence == 0.20
    assert live.optimizer.no_trade.margin == 10.0
