"""Nested YAML overrides for mode settings."""

from __future__ import annotations

from spy_der.config.settings import load_settings
from spy_der.risk.limits import TradeLimits


def test_backtest_mode_lowers_min_confidence():
    settings = load_settings("backtest")
    assert isinstance(settings.risk.trade, TradeLimits)
    assert settings.risk.trade.min_confidence == 0.08
    # Live defaults remain stricter.
    assert load_settings("paper").risk.trade.min_confidence == 0.20
