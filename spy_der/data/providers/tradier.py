"""Live SPY market data from the Tradier REST API.

Until now ``TRADIER_ACCESS_TOKEN`` appeared in ``.env.example`` under a header
reading "Market-data providers (optional; supervisor falls back to synthetic)"
and *no code read it*. An operator who set a real token got a healthy-looking
supervisor running entirely on the synthetic generator, with nothing anywhere
saying so. This module is the missing half; :mod:`spy_der.vps.service` is where
the selection became loud.

Three properties are deliberate, because a live feed is the one place where a
plausible-looking wrong number is most expensive:

**It refuses to answer for a time it cannot see.** Tradier serves *now*. The
:class:`~spy_der.data.providers.base.MarketDataProvider` contract is "the
snapshot as of ``at``", and a REST quote endpoint cannot honour that for a past
``at``. Rather than return current data wearing a historical timestamp — which
would silently corrupt any backtest that reached for this provider — a request
outside ``staleness_tolerance`` raises. Replay is what answers for the past.

**It never invents implied volatility.** Tradier reports ``10.0`` as a sentinel
for "no IV available", not as 1000% vol; taken literally it prices a contract
absurdly and poisons every aggregate it enters. Sentinel and missing values are
solved from the mid, and dropped if unsolvable — never emitted as zero, which
would assert vol *is* zero when the truth is that it is unknown.

**It fails loudly.** An expired token, a revoked scope, or a rate limit raises a
typed error. Nothing here degrades to synthetic on its own; that decision
belongs to the caller, which has to make it visible.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

from spy_der.analytics import pricing
from spy_der.data.providers.base import MarketDataProvider, score_snapshot
from spy_der.data.providers.spyder_recordings import _infer_quote_size
from spy_der.data.validators.quality import DataQualityConfig
from spy_der.domain.enums import OptionRight
from spy_der.domain.market import ContextSnapshot, MarketSnapshot, OptionQuote, PriceBar, SpyQuote
from spy_der.runtime import calendar as market_calendar

logger = logging.getLogger(__name__)

PRODUCTION_BASE_URL = "https://api.tradier.com"
SANDBOX_BASE_URL = "https://sandbox.tradier.com"

#: Tradier emits this as "no IV", not as 1000% volatility. Treated as missing.
IV_SENTINEL = 10.0


class TradierError(RuntimeError):
    """Base class for Tradier transport and payload failures."""


class TradierAuthError(TradierError):
    """Token missing, expired, or lacking the market-data scope."""


class TradierRateLimitError(TradierError):
    """Rate limit hit; the caller should back off rather than hammer."""


class TradierUnavailableError(TradierError):
    """Tradier reachable but not serving — 5xx, timeout, or malformed body."""


class TradierDataError(TradierError):
    """A response parsed cleanly but carried nothing usable."""


@dataclass(frozen=True)
class TradierConfig:
    """Connection and parsing policy for the live feed."""

    access_token: str
    base_url: str = PRODUCTION_BASE_URL
    symbol: str = "SPY"
    timeout: float = 10.0
    max_retries: int = 3
    backoff_base: float = 0.5
    #: How many near expirations to pull. 0DTE strategies need today; the
    #: term-structure reads in the forecast engine need at least one more.
    expirations: int = 3
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.013
    derive_missing_iv: bool = True
    drop_unpriceable: bool = True
    min_contracts: int = 20
    bars_lookback: timedelta = timedelta(minutes=90)
    #: How far ``snapshot(at)`` may sit from now before the request is refused.
    staleness_tolerance: timedelta = timedelta(minutes=5)
    quality: DataQualityConfig | None = None

    @property
    def is_sandbox(self) -> bool:
        return "sandbox" in self.base_url

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> TradierConfig | None:
        """Build from the environment, or ``None`` when no token is configured.

        ``None`` means "not configured", which is a legitimate state. It is
        distinct from a token that is present but broken — that raises when
        used, and must not be confused with absence.
        """
        source = os.environ if env is None else env
        token = (source.get("TRADIER_ACCESS_TOKEN") or "").strip()
        if not token:
            return None

        base = (source.get("TRADIER_BASE_URL") or "").strip()
        if not base:
            endpoint = (source.get("TRADIER_ENDPOINT") or "production").strip().lower()
            base = SANDBOX_BASE_URL if endpoint in {"sandbox", "sbx", "paper"} else PRODUCTION_BASE_URL

        settings: dict[str, Any] = {"access_token": token, "base_url": base.rstrip("/")}
        if (raw := source.get("TRADIER_TIMEOUT")):
            settings["timeout"] = float(raw)
        if (raw := source.get("TRADIER_EXPIRATIONS")):
            settings["expirations"] = max(1, int(raw))
        settings.update(overrides)
        return cls(**settings)


@dataclass
class TradierClient:
    """Minimal authenticated GET client over the stdlib.

    Deliberately not ``httpx``/``requests``: both are dev-only or undeclared in
    this project, and a live trading daemon should not acquire a new runtime
    dependency for four GET endpoints.
    """

    config: TradierConfig
    opener: Any = None

    def __post_init__(self) -> None:
        if self.opener is None:
            self.opener = urllib.request.build_opener()

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"

        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.config.access_token}",
                "Accept": "application/json",
                "User-Agent": "iron-spyder/1.0",
            },
            method="GET",
        )

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with self.opener.open(request, timeout=self.config.timeout) as response:
                    body = response.read().decode("utf-8")
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise TradierDataError(f"{path} returned {type(payload).__name__}, expected object")
                return payload
            except urllib.error.HTTPError as exc:
                # 401/403 are terminal: retrying a bad token only burns the rate
                # limit and delays the operator seeing the real problem.
                if exc.code in (401, 403):
                    raise TradierAuthError(
                        f"Tradier rejected the access token ({exc.code}) for {path}. "
                        "Check TRADIER_ACCESS_TOKEN and that the account has market-data access."
                    ) from exc
                if exc.code == 429:
                    raise TradierRateLimitError(f"Tradier rate limit hit on {path}") from exc
                last_error = exc
                if exc.code < 500:
                    raise TradierUnavailableError(f"Tradier {path} returned HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                last_error = exc

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.backoff_base * (2**attempt))

        raise TradierUnavailableError(
            f"Tradier {path} failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error


def _as_list(value: Any) -> list[dict[str, Any]]:
    """Normalize Tradier's singular-or-list payloads.

    Tradier returns a bare object when exactly one record matches and a list
    otherwise. Indexing the object as a list silently yields nothing, so a
    one-contract expiration would read as an empty chain rather than an error.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _expiration_datetime(day: date) -> datetime:
    """The moment an option on ``day`` actually expires, in UTC.

    Uses the market calendar so half-days land at 13:00 ET rather than 16:00.
    Getting this wrong shifts tau on every 0DTE contract by three hours, which
    is most of its remaining life.
    """
    session = market_calendar.session_for(day)
    if session is not None:
        return session.close_at
    return datetime.combine(day, market_calendar.REGULAR_CLOSE, tzinfo=market_calendar.EASTERN).astimezone(UTC)


@dataclass
class ChainStats:
    """What the parse kept, solved, and threw away."""

    contracts_seen: int = 0
    contracts_kept: int = 0
    iv_from_vendor: int = 0
    iv_derived: int = 0
    iv_sentinel: int = 0
    dropped_unpriceable: int = 0

    def summary(self) -> str:
        return (
            f"{self.contracts_kept}/{self.contracts_seen} contracts kept "
            f"(iv: {self.iv_from_vendor} vendor, {self.iv_derived} solved, "
            f"{self.iv_sentinel} sentinel; {self.dropped_unpriceable} dropped)"
        )


def quote_from_option(
    record: Mapping[str, Any],
    *,
    timestamp: datetime,
    spot: float,
    config: TradierConfig,
    stats: ChainStats,
) -> OptionQuote | None:
    """Convert one Tradier option record into an ``OptionQuote``, or drop it."""
    stats.contracts_seen += 1

    option_type = str(record.get("option_type") or "").lower()
    if option_type not in {"call", "put"}:
        return None
    right = OptionRight.CALL if option_type == "call" else OptionRight.PUT

    strike = _to_float(record.get("strike"))
    if strike <= 0.0:
        return None

    raw_expiration = record.get("expiration_date")
    if not raw_expiration:
        return None
    try:
        expiration = _expiration_datetime(date.fromisoformat(str(raw_expiration)))
    except ValueError:
        return None

    bid = _to_float(record.get("bid"))
    ask = _to_float(record.get("ask"))
    last = _to_float(record.get("last"), 0.5 * (bid + ask))
    volume = _to_int(record.get("volume"))
    open_interest = _to_int(record.get("open_interest"))

    bid_size = _to_int(record.get("bidsize"))
    ask_size = _to_int(record.get("asksize"))
    if bid_size <= 0 or ask_size <= 0:
        inferred = _infer_quote_size(open_interest=open_interest, volume=volume)
        bid_size = bid_size if bid_size > 0 else inferred
        ask_size = ask_size if ask_size > 0 else inferred

    greeks = record.get("greeks")
    iv = _to_float(greeks.get("mid_iv")) if isinstance(greeks, Mapping) else 0.0
    # Sandbox omits greeks entirely, and production emits 10.0 when it has no
    # surface for a contract. Both mean "unknown", and both must be solved
    # rather than believed.
    if iv >= IV_SENTINEL:
        stats.iv_sentinel += 1
        iv = 0.0
    elif iv > 0.0:
        stats.iv_from_vendor += 1

    quote = OptionQuote(
        contract_symbol=str(record.get("symbol") or "").strip()
        or f"SPY{expiration:%y%m%d}{'C' if right is OptionRight.CALL else 'P'}{int(round(strike * 1000)):08d}",
        timestamp=timestamp,
        expiration=expiration,
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        last=last,
        bid_size=bid_size,
        ask_size=ask_size,
        volume=volume,
        open_interest=open_interest,
        implied_volatility=iv,
        underlying_price=spot,
        source="tradier",
    )

    bounds = config.quality or DataQualityConfig()
    in_bounds = bounds.min_iv <= quote.implied_volatility <= bounds.max_iv
    if not in_bounds and config.derive_missing_iv and quote.mid > 0.0 and quote.tau > 0.0:
        solved = pricing.implied_volatility(
            quote.mid,
            spot,
            quote.strike,
            quote.tau,
            config.risk_free_rate,
            config.dividend_yield,
            quote.right,
        )
        if solved is not None and bounds.min_iv <= solved <= bounds.max_iv:
            quote = replace(quote, implied_volatility=solved)
            stats.iv_derived += 1
            in_bounds = True

    if config.drop_unpriceable and not in_bounds:
        stats.dropped_unpriceable += 1
        return None

    stats.contracts_kept += 1
    return quote


@dataclass
class TradierProvider(MarketDataProvider):
    """Live SPY chain, quote, and bars from Tradier.

    Construct via :meth:`from_env` in production so a missing token is a clean
    ``None`` rather than an exception at import time.
    """

    config: TradierConfig
    client: TradierClient | None = None
    last_stats: ChainStats = field(default_factory=ChainStats)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = TradierClient(self.config)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, **overrides: Any
    ) -> TradierProvider | None:
        """Provider from the environment, or ``None`` when unconfigured."""
        config = TradierConfig.from_env(env, **overrides)
        return None if config is None else cls(config=config)

    # -- provider contract -------------------------------------------------

    def snapshot(self, at: datetime) -> MarketSnapshot:
        """Live snapshot, refusing any ``at`` the REST API cannot speak for."""
        moment = at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
        now = datetime.now(UTC)
        drift = abs((now - moment).total_seconds())
        if drift > self.config.staleness_tolerance.total_seconds():
            raise TradierDataError(
                f"TradierProvider serves live data only; asked for {moment.isoformat()} "
                f"which is {drift / 60:.1f} min from now. Use ReplayProvider or "
                "HistoricalProvider for point-in-time history."
            )

        spot_quote = self.fetch_spy_quote(moment)
        spot = spot_quote.mid
        if spot <= 0.0:
            raise TradierDataError("Tradier returned no usable SPY price")

        stats = ChainStats()
        quotes: list[OptionQuote] = []
        for expiry in self.fetch_expirations()[: self.config.expirations]:
            quotes.extend(self.fetch_chain(expiry, timestamp=moment, spot=spot, stats=stats))
        self.last_stats = stats

        if len(quotes) < self.config.min_contracts:
            raise TradierDataError(
                f"Tradier chain too thin to decide on: {stats.summary()}. "
                f"Need at least {self.config.min_contracts} priceable contracts."
            )
        logger.debug("tradier chain: %s", stats.summary())

        session = market_calendar.current_session(moment)
        snapshot = MarketSnapshot(
            timestamp=moment,
            spy_quote=spot_quote,
            spy_bars=tuple(self.fetch_bars(moment)),
            option_chain=tuple(quotes),
            context=ContextSnapshot(
                minutes_to_close=market_calendar.minutes_to_close(moment),
                is_early_close=session.is_early_close if session is not None else False,
            ),
            risk_free_rate=self.config.risk_free_rate,
            dividend_yield=self.config.dividend_yield,
        )
        return score_snapshot(snapshot, self.config.quality)

    def session_timestamps(self, day: datetime) -> Sequence[datetime]:
        """Only the present is available; history is a replay concern."""
        now = datetime.now(UTC)
        session = market_calendar.session_for(day.date())
        if session is None or not session.contains(now):
            return ()
        return (now,)

    # -- endpoints ---------------------------------------------------------

    def fetch_spy_quote(self, timestamp: datetime) -> SpyQuote:
        assert self.client is not None
        payload = self.client.get("/v1/markets/quotes", {"symbols": self.config.symbol})
        records = _as_list((payload.get("quotes") or {}).get("quote"))
        if not records:
            raise TradierDataError(f"no quote returned for {self.config.symbol}")

        record = records[0]
        last = _to_float(record.get("last"))
        bid = _to_float(record.get("bid"))
        ask = _to_float(record.get("ask"))
        # Outside regular hours Tradier can return a stale two-sided quote or
        # none at all; the last trade is the honest fallback, and a crossed or
        # empty book must not become a mid of zero.
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            bid = ask = last
        return SpyQuote(timestamp=timestamp, bid=bid, ask=ask, last=last or bid)

    def fetch_expirations(self) -> list[date]:
        assert self.client is not None
        payload = self.client.get(
            "/v1/markets/options/expirations",
            {"symbol": self.config.symbol, "includeAllRoots": "true", "strikes": "false"},
        )
        raw = (payload.get("expirations") or {}).get("date")
        values = raw if isinstance(raw, list) else ([raw] if raw else [])

        today = datetime.now(UTC).date()
        days: list[date] = []
        for value in values:
            try:
                parsed = date.fromisoformat(str(value))
            except (TypeError, ValueError):
                continue
            if parsed >= today:
                days.append(parsed)
        if not days:
            raise TradierDataError(f"no future expirations for {self.config.symbol}")
        return sorted(days)

    def fetch_chain(
        self, expiration: date, *, timestamp: datetime, spot: float, stats: ChainStats
    ) -> list[OptionQuote]:
        assert self.client is not None
        payload = self.client.get(
            "/v1/markets/options/chains",
            {
                "symbol": self.config.symbol,
                "expiration": expiration.isoformat(),
                "greeks": "true",
            },
        )
        records = _as_list((payload.get("options") or {}).get("option"))
        return [
            quote
            for record in records
            if (
                quote := quote_from_option(
                    record, timestamp=timestamp, spot=spot, config=self.config, stats=stats
                )
            )
            is not None
        ]

    def fetch_bars(self, moment: datetime) -> list[PriceBar]:
        """Recent 1-minute bars; an empty list rather than a raise.

        Bars feed realized-volatility context, not the decision itself. A
        snapshot without them is degraded but usable, and the data-quality
        engine scores that; failing the whole cycle would be the worse trade.
        """
        assert self.client is not None
        start = moment - self.config.bars_lookback
        try:
            payload = self.client.get(
                "/v1/markets/timesales",
                {
                    "symbol": self.config.symbol,
                    "interval": "1min",
                    "start": start.strftime("%Y-%m-%d %H:%M"),
                    "end": moment.strftime("%Y-%m-%d %H:%M"),
                    "session_filter": "open",
                },
            )
        except TradierError as exc:
            logger.warning("tradier timesales unavailable, continuing without bars: %s", exc)
            return []

        bars: list[PriceBar] = []
        for record in _as_list((payload.get("series") or {}).get("data")):
            close = _to_float(record.get("close"))
            if close <= 0.0:
                continue
            try:
                stamp = datetime.fromisoformat(str(record.get("time"))).replace(tzinfo=UTC)
            except (TypeError, ValueError):
                continue
            bars.append(
                PriceBar(
                    timestamp=stamp,
                    open=_to_float(record.get("open"), close),
                    high=_to_float(record.get("high"), close),
                    low=_to_float(record.get("low"), close),
                    close=close,
                    volume=_to_float(record.get("volume")),
                    vwap=_to_float(record.get("vwap"), close),
                )
            )
        bars.sort(key=lambda bar: bar.timestamp)
        return bars

    def check_connectivity(self) -> str:
        """Prove the token works, for startup validation. Raises on failure."""
        quote = self.fetch_spy_quote(datetime.now(UTC))
        mode = "sandbox" if self.config.is_sandbox else "production"
        return f"Tradier {mode} reachable; {self.config.symbol} last={quote.last:.2f}"
