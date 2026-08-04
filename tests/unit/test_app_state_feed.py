"""API AppState must share the supervisor's live-vs-synthetic rule."""

from __future__ import annotations

from spy_der.app.state import resolve_market_provider
from spy_der.data.providers.base import SyntheticProvider
from spy_der.data.providers.tradier import TradierProvider


def test_resolve_market_provider_is_synthetic_without_token(monkeypatch):
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("IRON_SPYDER_FORCE_SYNTHETIC", raising=False)
    provider, feed = resolve_market_provider("broad_range")
    assert feed == "synthetic"
    assert isinstance(provider, SyntheticProvider)


def test_resolve_market_provider_respects_force_synthetic(monkeypatch):
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "not-used")
    monkeypatch.setenv("IRON_SPYDER_FORCE_SYNTHETIC", "1")
    provider, feed = resolve_market_provider("broad_range")
    assert feed == "synthetic"
    assert isinstance(provider, SyntheticProvider)


def test_resolve_market_provider_prefers_tradier(monkeypatch):
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "tok_test")
    monkeypatch.delenv("IRON_SPYDER_FORCE_SYNTHETIC", raising=False)

    class _Ok:
        def check_connectivity(self) -> str:
            return "Tradier production reachable; SPY last=757.67"

    monkeypatch.setattr(
        TradierProvider,
        "from_env",
        classmethod(lambda cls, env=None, **kw: _Ok()),
    )
    provider, feed = resolve_market_provider("broad_range")
    assert feed == "tradier"
    assert isinstance(provider, _Ok)
