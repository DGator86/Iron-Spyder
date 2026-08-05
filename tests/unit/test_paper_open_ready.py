"""Paper mode must size a $1k account and not freeze at default 0.20 confidence."""

from __future__ import annotations

from spy_der.config.settings import load_settings


def test_paper_open_ready_equity_and_gates():
    settings = load_settings("paper")
    assert settings.starting_equity == 1000.0
    # Default TradeLimits.min_confidence is 0.20 — that vetoed every open print.
    assert settings.risk.trade.min_confidence <= 0.05
    # $5 of risk (0.5% of $1k) cannot size a 1-wide SPY vertical (~$100).
    assert settings.starting_equity * settings.risk.trade.max_account_fraction_at_risk >= 100.0
    assert settings.risk.trade.max_dollar_risk >= 100.0
    assert settings.optimizer.no_trade.margin <= 1.0
    assert settings.optimizer.min_return_on_risk <= 0.02
