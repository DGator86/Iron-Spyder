"""Live Tradier provider: payload quirks, IV honesty, and loud failure."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, date, datetime, timedelta

import pytest

from spy_der.data.providers.base import SyntheticProvider
from spy_der.data.providers.tradier import (
    IV_SENTINEL,
    PRODUCTION_BASE_URL,
    SANDBOX_BASE_URL,
    ChainStats,
    TradierAuthError,
    TradierConfig,
    TradierDataError,
    TradierProvider,
    TradierRateLimitError,
    TradierUnavailableError,
    _as_list,
    _expiration_datetime,
    quote_from_option,
)
from spy_der.data.validators.quality import DataQualityConfig
from spy_der.vps.service import VpsServiceConfig, select_provider

SPOT = 640.0
EXPIRY = "2026-08-07"
NOW = datetime.now(UTC)


# --------------------------------------------------------------------------
# fake transport


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FakeOpener:
    """Serves canned JSON by URL substring; records what was asked for."""

    def __init__(self, routes: dict[str, object], error: Exception | None = None):
        self.routes = routes
        self.error = error
        self.calls: list[str] = []

    def open(self, request, timeout=None):  # noqa: ARG002 - signature parity
        url = request.full_url if hasattr(request, "full_url") else str(request)
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        for fragment, payload in self.routes.items():
            if fragment in url:
                return _FakeResponse(json.dumps(payload).encode())
        raise AssertionError(f"unrouted request: {url}")


def option_record(**overrides):
    record = {
        "symbol": "SPY260807C00640000",
        "option_type": "call",
        "strike": 640.0,
        "expiration_date": EXPIRY,
        "bid": 3.10,
        "ask": 3.20,
        "last": 3.15,
        "volume": 1200,
        "open_interest": 5000,
        "bidsize": 25,
        "asksize": 30,
        "greeks": {"mid_iv": 0.15},
    }
    record.update(overrides)
    return record


def chain_records(count: int = 40):
    """A two-sided chain wide enough to clear ``min_contracts``."""
    records = []
    for i in range(count // 2):
        strike = 620.0 + i * 2.0
        for right, price in (("call", max(0.05, SPOT - strike + 2)), ("put", max(0.05, strike - SPOT + 2))):
            records.append(
                option_record(
                    symbol=f"SPY260807{right[0].upper()}{int(strike * 1000):08d}",
                    option_type=right,
                    strike=strike,
                    bid=price - 0.05,
                    ask=price + 0.05,
                    last=price,
                )
            )
    return records


def full_routes(chain=None):
    return {
        "/markets/quotes": {"quotes": {"quote": {"symbol": "SPY", "bid": 639.98, "ask": 640.02, "last": 640.0}}},
        "/markets/options/expirations": {"expirations": {"date": [EXPIRY]}},
        "/markets/options/chains": {"options": {"option": chain if chain is not None else chain_records()}},
        "/markets/timesales": {
            "series": {
                "data": [
                    {"time": "2026-08-03T13:30:00", "open": 639.0, "high": 640.5, "low": 638.9, "close": 640.0, "volume": 10000}
                ]
            }
        },
    }


def provider(routes=None, **config_kwargs) -> TradierProvider:
    config = TradierConfig(access_token="test-token", max_retries=1, backoff_base=0.0, **config_kwargs)
    prov = TradierProvider(config=config)
    prov.client.opener = FakeOpener(routes if routes is not None else full_routes())
    return prov


# --------------------------------------------------------------------------
# payload normalization


def test_as_list_handles_tradier_singular_form():
    """One matching contract arrives as an object, not a one-element list.

    Treating the object as a list yields an empty chain rather than an error,
    which reads downstream as "this expiration has no options".
    """
    assert _as_list({"strike": 640.0}) == [{"strike": 640.0}]
    assert len(_as_list([{"a": 1}, {"b": 2}])) == 2
    assert _as_list(None) == []
    assert _as_list("garbage") == []


def test_single_contract_expiration_is_not_silently_empty():
    routes = full_routes(chain=option_record())  # bare object, not a list
    prov = provider(routes, min_contracts=1)
    snapshot = prov.snapshot(NOW)
    assert len(snapshot.option_chain) == 1


# --------------------------------------------------------------------------
# implied volatility


def test_vendor_iv_is_used_when_present():
    stats = ChainStats()
    quote = quote_from_option(
        option_record(), timestamp=NOW, spot=SPOT, config=TradierConfig(access_token="x"), stats=stats
    )
    assert quote.implied_volatility == pytest.approx(0.15)
    assert stats.iv_from_vendor == 1
    assert stats.iv_derived == 0


def test_iv_sentinel_is_not_believed():
    """Tradier emits 10.0 for "no IV". Taken literally that is 1000% vol."""
    stats = ChainStats()
    quote = quote_from_option(
        option_record(greeks={"mid_iv": IV_SENTINEL}),
        timestamp=NOW,
        spot=SPOT,
        config=TradierConfig(access_token="x"),
        stats=stats,
    )
    assert stats.iv_sentinel == 1
    assert quote is not None, "a priceable contract should be solved, not discarded"
    assert quote.implied_volatility < 1.0
    assert stats.iv_derived == 1


def test_missing_greeks_are_solved_not_zeroed():
    """Sandbox omits greeks. IV=0 would price the contract at intrinsic."""
    stats = ChainStats()
    quote = quote_from_option(
        option_record(greeks=None), timestamp=NOW, spot=SPOT, config=TradierConfig(access_token="x"), stats=stats
    )
    assert quote is not None
    assert quote.implied_volatility > 0.0
    assert stats.iv_derived == 1


def test_unpriceable_contract_is_dropped_never_zero_iv():
    stats = ChainStats()
    quote = quote_from_option(
        option_record(bid=0.0, ask=0.0, last=0.0, greeks=None),
        timestamp=NOW,
        spot=SPOT,
        config=TradierConfig(access_token="x"),
        stats=stats,
    )
    assert quote is None
    assert stats.dropped_unpriceable == 1


def test_no_snapshot_contract_carries_zero_iv():
    snapshot = provider().snapshot(NOW)
    bounds = DataQualityConfig()
    assert snapshot.option_chain
    for quote in snapshot.option_chain:
        assert bounds.min_iv <= quote.implied_volatility <= bounds.max_iv


# --------------------------------------------------------------------------
# expiration timing


def test_expiration_uses_session_close_not_midnight():
    assert _expiration_datetime(date(2026, 8, 7)).hour == 20  # 16:00 ET in UTC


def test_early_close_expiration_is_three_hours_earlier():
    """Half-days close at 13:00 ET; a 0DTE tau must not claim the extra hours."""
    normal = _expiration_datetime(date(2026, 11, 27))  # day after Thanksgiving
    assert normal.hour == 18  # 13:00 ET


# --------------------------------------------------------------------------
# point-in-time honesty


def test_snapshot_refuses_historical_timestamps():
    """A REST quote endpoint cannot speak for the past; it must not pretend."""
    with pytest.raises(TradierDataError, match="live data only"):
        provider().snapshot(NOW - timedelta(days=3))


def test_snapshot_accepts_now_within_tolerance():
    assert provider().snapshot(NOW - timedelta(seconds=30)) is not None


def test_thin_chain_raises_rather_than_deciding_on_scraps():
    prov = provider(full_routes(chain=[option_record()]), min_contracts=20)
    with pytest.raises(TradierDataError, match="too thin"):
        prov.snapshot(NOW)


# --------------------------------------------------------------------------
# transport failures


def test_auth_failure_is_terminal_and_not_retried():
    prov = provider()
    error = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
    prov.client.opener = FakeOpener({}, error=error)
    prov.client.config = TradierConfig(access_token="bad", max_retries=3, backoff_base=0.0)
    with pytest.raises(TradierAuthError, match="rejected the access token"):
        prov.fetch_spy_quote(NOW)
    assert len(prov.client.opener.calls) == 1  # no retry storm on a bad token


def test_rate_limit_raises_its_own_type():
    prov = provider()
    prov.client.opener = FakeOpener({}, error=urllib.error.HTTPError("url", 429, "Too Many", {}, None))
    with pytest.raises(TradierRateLimitError):
        prov.fetch_spy_quote(NOW)


def test_server_error_retries_then_raises():
    prov = provider()
    prov.client.config = TradierConfig(access_token="x", max_retries=3, backoff_base=0.0)
    prov.client.opener = FakeOpener({}, error=urllib.error.URLError("connection reset"))
    with pytest.raises(TradierUnavailableError):
        prov.fetch_spy_quote(NOW)
    assert len(prov.client.opener.calls) == 3


def test_missing_bars_degrade_rather_than_fail_the_cycle():
    routes = full_routes()
    routes["/markets/timesales"] = {"series": None}
    snapshot = provider(routes).snapshot(NOW)
    assert snapshot.spy_bars == ()
    assert snapshot.option_chain  # the decision inputs survived


# --------------------------------------------------------------------------
# configuration


def test_from_env_returns_none_without_a_token():
    """Absence is a legitimate state and must not raise."""
    assert TradierProvider.from_env({}) is None
    assert TradierConfig.from_env({"TRADIER_ACCESS_TOKEN": "   "}) is None


def test_endpoint_selection():
    assert TradierConfig.from_env({"TRADIER_ACCESS_TOKEN": "t"}).base_url == PRODUCTION_BASE_URL
    sandbox = TradierConfig.from_env({"TRADIER_ACCESS_TOKEN": "t", "TRADIER_ENDPOINT": "sandbox"})
    assert sandbox.base_url == SANDBOX_BASE_URL
    assert sandbox.is_sandbox
    explicit = TradierConfig.from_env({"TRADIER_ACCESS_TOKEN": "t", "TRADIER_BASE_URL": "https://x.test/"})
    assert explicit.base_url == "https://x.test"


# --------------------------------------------------------------------------
# provider selection — the bug this whole module exists to remove


def test_no_token_selects_synthetic_with_a_warning(monkeypatch, caplog):
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN", raising=False)
    with caplog.at_level("WARNING"):
        chosen = select_provider(VpsServiceConfig())
    assert isinstance(chosen, SyntheticProvider)
    assert "SYNTHETIC" in caplog.text
    assert "invented" in caplog.text


def test_broken_token_raises_instead_of_falling_back(monkeypatch):
    """The failure this replaces: a real token, a healthy daemon, fake prices.

    Falling back here is worse than no token at all — someone deliberately
    configured a live feed, so a silent downgrade is one nobody looks for.
    """
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "expired-token")

    def explode(self):
        raise TradierAuthError("Tradier rejected the access token (401)")

    monkeypatch.setattr(TradierProvider, "check_connectivity", explode)
    with pytest.raises(RuntimeError, match="Refusing to start on synthetic data"):
        select_provider(VpsServiceConfig())


def test_working_token_selects_live(monkeypatch):
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "good-token")
    monkeypatch.setattr(TradierProvider, "check_connectivity", lambda self: "ok")
    assert isinstance(select_provider(VpsServiceConfig()), TradierProvider)


def test_use_tradier_false_skips_the_token_entirely(monkeypatch):
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "would-explode")
    chosen = select_provider(VpsServiceConfig(use_tradier=False))
    assert isinstance(chosen, SyntheticProvider)
