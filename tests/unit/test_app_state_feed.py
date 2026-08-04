"""API AppState must share the supervisor's live-vs-synthetic rule."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from spy_der.app.state import AppState, resolve_market_provider
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


def test_current_refreshes_when_last_result_is_stale(monkeypatch):
    monkeypatch.setenv("IRON_SPYDER_API_MAX_AGE_SECONDS", "30")
    monkeypatch.setenv("IRON_SPYDER_FORCE_SYNTHETIC", "1")
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN", raising=False)

    state = AppState(scenario_name="broad_range")
    calls = {"n": 0}
    original = state.run_once

    def counted(at=None):
        calls["n"] += 1
        return original(at=at)

    state.run_once = counted  # type: ignore[method-assign]
    first = state.current()
    assert calls["n"] == 1
    # Fresh enough — reuse.
    again = state.current()
    assert again is first
    assert calls["n"] == 1
    # Age the cached result past the max age.
    state.last_result = SimpleNamespace(
        timestamp=datetime.now(UTC) - timedelta(seconds=45)
    )
    state.current()
    assert calls["n"] == 2
