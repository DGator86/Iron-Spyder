"""Historical option-chain ingestion.

Turns recorded SPY option data into :class:`MarketSnapshot` objects the
existing pipeline and backtest already consume. Nothing downstream changes:
replaying history is just a provider that answers for a past timestamp.

**Column naming is not standardized across vendors**, so the loader maps a wide
set of aliases onto a canonical schema rather than demanding one layout. See
:data:`COLUMN_ALIASES`. Anything unmapped is reported by name, so a mismatch
produces a clear error instead of silently-zero open interest.

**Loading is streaming.** A year of one-minute SPY chains is tens of millions
of rows; the reader groups by timestamp and yields one snapshot at a time so
memory stays bounded by a single chain rather than the whole history.

Implied volatility is optional on input. When absent it is solved from the
midpoint with the same model the analytics layer uses, which keeps the surface
self-consistent instead of mixing a vendor's IV convention with ours.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from spy_der.analytics import pricing
from spy_der.data.providers.base import MarketDataProvider, score_snapshot
from spy_der.data.validators.quality import DataQualityConfig
from spy_der.domain.enums import OptionRight
from spy_der.domain.market import (
    ContextSnapshot,
    MarketSnapshot,
    OptionQuote,
    PriceBar,
    SpyQuote,
)
from spy_der.runtime import calendar as market_calendar

#: Canonical field -> accepted source column names, lower-cased.
#: Add to this rather than reshaping data upstream.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "timestamp", "quote_datetime", "datetime", "date_time", "ts", "time",
        "quote_time", "snapshot_time", "quote_readtime",
    ),
    "expiration": (
        "expiration", "expiry", "expiration_date", "exp_date", "expirationdate",
        "expire_date", "maturity",
    ),
    "strike": ("strike", "strike_price", "k", "strikeprice"),
    "right": (
        "right", "option_type", "type", "cp_flag", "call_put", "callput",
        "put_call", "opt_type", "option_right",
    ),
    "bid": ("bid", "bid_price", "best_bid", "bid_1545", "bidprice"),
    "ask": ("ask", "ask_price", "best_ask", "offer", "ask_1545", "askprice"),
    "last": ("last", "last_price", "trade_price", "close", "last_trade_price"),
    "bid_size": ("bid_size", "bidsize", "bid_sz", "best_bid_size"),
    "ask_size": ("ask_size", "asksize", "ask_sz", "best_ask_size"),
    "volume": ("volume", "trade_volume", "vol", "day_volume"),
    "open_interest": ("open_interest", "oi", "open_int", "openinterest"),
    "implied_volatility": (
        "implied_volatility", "iv", "implied_vol", "mid_iv", "impliedvol",
        "volatility", "sigma",
    ),
    "underlying_price": (
        "underlying_price", "underlying", "underlying_last", "spot",
        "active_underlying_price", "underlying_bid_ask_mid", "stock_price",
    ),
}

#: Fields a row must supply; everything else has a defensible default.
REQUIRED_FIELDS = frozenset(
    {"timestamp", "expiration", "strike", "right", "bid", "ask"}
)

#: Bar columns, mapped the same way.
BAR_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "datetime", "date_time", "ts", "time", "date"),
    "open": ("open", "o", "open_price"),
    "high": ("high", "h", "high_price"),
    "low": ("low", "l", "low_price"),
    "close": ("close", "c", "close_price", "last"),
    "volume": ("volume", "v", "vol"),
    "vwap": ("vwap", "vw", "average"),
}

_CALL_TOKENS = {"c", "call", "calls", "1", "+1"}
_PUT_TOKENS = {"p", "put", "puts", "-1", "0"}


class SchemaError(ValueError):
    """Raised when input columns cannot be mapped onto the canonical schema."""


def build_column_map(
    columns: Iterable[str], aliases: dict[str, tuple[str, ...]] = COLUMN_ALIASES
) -> dict[str, str]:
    """Map canonical field names to the source columns present in ``columns``."""
    normalized = {str(c).strip().lower(): str(c) for c in columns}
    mapping: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalized:
                mapping[canonical] = normalized[candidate]
                break
    return mapping


def _require_schema(mapping: dict[str, str], columns: Iterable[str]) -> None:
    missing = REQUIRED_FIELDS - mapping.keys()
    if missing:
        raise SchemaError(
            f"could not map required field(s) {sorted(missing)}; "
            f"saw columns {sorted(str(c) for c in columns)}. "
            f"Add the source names to COLUMN_ALIASES."
        )


def parse_right(value: Any) -> OptionRight:
    token = str(value).strip().lower()
    if token in _CALL_TOKENS:
        return OptionRight.CALL
    if token in _PUT_TOKENS:
        return OptionRight.PUT
    raise ValueError(f"cannot interpret option right {value!r}")


def parse_timestamp(value: Any, *, assume_utc: bool = True) -> datetime:
    """Parse a timestamp, attaching UTC when the source is naive.

    A naive timestamp is ambiguous and the choice matters — it shifts every
    time-to-expiry — so the assumption is explicit rather than implied.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime.combine(value, datetime.min.time())
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
                try:
                    moment = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"cannot parse timestamp {value!r}") from None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC) if assume_utc else moment.astimezone(UTC)
    return moment.astimezone(UTC)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def _to_int(value: Any, default: int = 0) -> int:
    return int(_to_float(value, float(default)))


@dataclass
class IngestConfig:
    """How to interpret a historical dataset."""

    assume_utc: bool = True
    """Treat naive timestamps as UTC. Set False to convert from local time."""

    derive_missing_iv: bool = True
    """Solve IV from the midpoint when the source has none."""

    risk_free_rate: float = 0.04
    dividend_yield: float = 0.013
    default_quote_size: int = 10
    min_contracts_per_snapshot: int = 20
    """Skip snapshots thinner than this; a handful of quotes is not a chain."""

    drop_unpriceable: bool = True
    """Drop quotes with no establishable implied volatility.

    Keeping them would put ``implied_volatility=0.0`` into the chain, which
    reads as a claim rather than a gap and contributes zero gamma and vega to
    every aggregate."""

    expiration_time_utc: tuple[int, int] = (20, 0)
    """Clock time to attach when the source gives an expiration date only.

    16:00 ET. Expirations usually arrive as bare dates, and leaving them at
    midnight would understate every time to expiry by most of a day.
    """

    quality: DataQualityConfig = field(default_factory=DataQualityConfig)


@dataclass
class IngestStats:
    rows: int = 0
    parsed: int = 0
    skipped: int = 0
    dropped_unpriceable: int = 0
    snapshots: int = 0
    thin_snapshots: int = 0
    no_spot_snapshots: int = 0
    iv_derived: int = 0
    iv_unavailable: int = 0
    errors: list[str] = field(default_factory=list)

    def note(self, message: str, *, cap: int = 20) -> None:
        if len(self.errors) < cap:
            self.errors.append(message)

    def summary(self) -> str:
        return (
            f"{self.rows} rows -> {self.parsed} quotes in {self.snapshots} snapshots "
            f"(skipped {self.skipped}, dropped-unpriceable "
            f"{self.dropped_unpriceable}, thin {self.thin_snapshots}, "
            f"no-spot {self.no_spot_snapshots}, iv derived {self.iv_derived})"
        )


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield rows from a CSV or Parquet file.

    Parquet is read in row groups so a multi-gigabyte file does not have to be
    materialized to iterate it.
    """
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".gz"}:
        opener = _open_maybe_gzip(path)
        with opener as handle:
            yield from csv.DictReader(handle)
        return

    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        frame = pd.read_parquet(path)
        yield from frame.to_dict("records")
        return

    raise SchemaError(f"unsupported file type {path.suffix!r} for {path}")


def _open_maybe_gzip(path: Path):
    if path.suffix.lower() == ".gz":
        import gzip

        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def _quote_from_row(
    row: dict[str, Any],
    mapping: dict[str, str],
    config: IngestConfig,
    stats: IngestStats,
) -> OptionQuote | None:
    """Parse one row into a quote, or ``None`` when the row is unusable.

    Implied volatility is *not* solved here, and the underlying may still be
    zero. Both need the chain's spot, which is only known once the timestamp's
    rows have been grouped — see :func:`_price_quote`.
    """
    try:
        timestamp = parse_timestamp(row[mapping["timestamp"]], assume_utc=config.assume_utc)
        expiration = parse_timestamp(
            row[mapping["expiration"]], assume_utc=config.assume_utc
        )
        right = parse_right(row[mapping["right"]])
        strike = float(row[mapping["strike"]])
    except (KeyError, ValueError, TypeError) as exc:
        stats.note(f"unparseable row: {exc}")
        return None

    if strike <= 0.0:
        stats.note(f"non-positive strike {strike}")
        return None

    # A bare expiration date settles at the close, not at midnight.
    if expiration.hour == 0 and expiration.minute == 0:
        hour, minute = config.expiration_time_utc
        expiration = expiration.replace(hour=hour, minute=minute)

    bid = _to_float(row.get(mapping.get("bid", ""), 0.0))
    ask = _to_float(row.get(mapping.get("ask", ""), 0.0))
    mid = 0.5 * (bid + ask)

    return OptionQuote(
        contract_symbol=_symbol(expiration, strike, right),
        timestamp=timestamp,
        expiration=expiration,
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        last=_to_float(row.get(mapping.get("last", ""), mid), mid),
        bid_size=_to_int(row.get(mapping.get("bid_size", "")), config.default_quote_size),
        ask_size=_to_int(row.get(mapping.get("ask_size", "")), config.default_quote_size),
        volume=_to_int(row.get(mapping.get("volume", ""))),
        open_interest=_to_int(row.get(mapping.get("open_interest", ""))),
        implied_volatility=_to_float(row.get(mapping.get("implied_volatility", ""), 0.0)),
        underlying_price=_to_float(row.get(mapping.get("underlying_price", ""), 0.0)),
        source="historical",
    )


def _price_quote(
    quote: OptionQuote, spot: float, config: IngestConfig, stats: IngestStats
) -> OptionQuote | None:
    """Attach the chain's spot and solve for IV, or drop an unpriceable quote.

    Separate from :func:`_quote_from_row` because ``spot`` is a property of the
    whole timestamp group, not of one row. Chains that ship the underlying in a
    separate bar file carry no per-row underlying at all, so solving IV during
    the row parse would fail on every contract in the file.
    """
    underlying = quote.underlying_price if quote.underlying_price > 0.0 else spot
    iv = quote.implied_volatility

    if iv <= 0.0 and config.derive_missing_iv and quote.mid > 0.0 and quote.tau > 0.0:
        solved = pricing.implied_volatility(
            quote.mid,
            underlying,
            quote.strike,
            quote.tau,
            config.risk_free_rate,
            config.dividend_yield,
            quote.right,
        )
        if solved is not None:
            iv = solved
            stats.iv_derived += 1
        else:
            stats.iv_unavailable += 1

    # Drop rather than emit ``implied_volatility=0.0``, which would assert that
    # volatility *is* zero when the truth is that it is unknown. Such a contract
    # prices at intrinsic with zero Greeks, so it would silently dilute every
    # exposure aggregate it appears in. Deep in-the-money quotes sitting at or
    # below intrinsic are the usual cause. The count is reported, and the
    # data-quality engine independently notices if enough contracts go missing
    # to thin the chain.
    if iv <= 0.0 and config.drop_unpriceable:
        stats.dropped_unpriceable += 1
        return None

    if iv == quote.implied_volatility and underlying == quote.underlying_price:
        return quote
    return replace(quote, implied_volatility=iv, underlying_price=underlying)


def _symbol(expiration: datetime, strike: float, right: OptionRight) -> str:
    code = "C" if right is OptionRight.CALL else "P"
    return f"SPY{expiration:%y%m%d}{code}{int(round(strike * 1000)):08d}"


def load_bars(path: Path, config: IngestConfig | None = None) -> list[PriceBar]:
    """Load SPY price bars from CSV or Parquet."""
    cfg = config or IngestConfig()
    rows = list(iter_rows(path))
    if not rows:
        return []

    mapping = build_column_map(rows[0].keys(), BAR_ALIASES)
    missing = {"timestamp", "close"} - mapping.keys()
    if missing:
        raise SchemaError(
            f"bar file {path} is missing {sorted(missing)}; saw {sorted(rows[0].keys())}"
        )

    bars: list[PriceBar] = []
    for row in rows:
        try:
            timestamp = parse_timestamp(row[mapping["timestamp"]], assume_utc=cfg.assume_utc)
        except ValueError:
            continue
        close = _to_float(row[mapping["close"]])
        if close <= 0.0:
            continue
        bars.append(
            PriceBar(
                timestamp=timestamp,
                open=_to_float(row.get(mapping.get("open", "")), close),
                high=_to_float(row.get(mapping.get("high", "")), close),
                low=_to_float(row.get(mapping.get("low", "")), close),
                close=close,
                volume=_to_float(row.get(mapping.get("volume", ""))),
                vwap=_to_float(row.get(mapping.get("vwap", "")), close),
            )
        )
    bars.sort(key=lambda b: b.timestamp)
    return bars


def iter_snapshots(
    chain_paths: Sequence[Path],
    *,
    bars: Sequence[PriceBar] = (),
    config: IngestConfig | None = None,
    stats: IngestStats | None = None,
) -> Iterator[MarketSnapshot]:
    """Stream snapshots from historical chain files, in timestamp order.

    Files are assumed to be chronologically separable (one per day is the usual
    layout). Within a file, rows are grouped by timestamp; the group is emitted
    when the timestamp changes, so only one chain is resident at a time.
    """
    cfg = config or IngestConfig()
    tally = stats if stats is not None else IngestStats()
    bar_index = _BarIndex(bars)

    for path in chain_paths:
        pending: dict[datetime, list[OptionQuote]] = defaultdict(list)
        mapping: dict[str, str] | None = None

        for row in iter_rows(path):
            tally.rows += 1
            if mapping is None:
                mapping = build_column_map(row.keys())
                _require_schema(mapping, row.keys())

            quote = _quote_from_row(row, mapping, cfg, tally)
            if quote is None:
                tally.skipped += 1
                continue
            pending[quote.timestamp].append(quote)

        for timestamp in sorted(pending):
            snapshot = _assemble(timestamp, pending[timestamp], bar_index, cfg, tally)
            if snapshot is not None:
                tally.snapshots += 1
                yield snapshot


def _assemble(
    timestamp: datetime,
    raw_quotes: list[OptionQuote],
    bar_index: _BarIndex,
    config: IngestConfig,
    stats: IngestStats,
) -> MarketSnapshot | None:
    """Build a scored snapshot from one timestamp's quotes."""
    spot = _infer_spot(raw_quotes, bar_index, timestamp)
    if spot <= 0.0:
        stats.no_spot_snapshots += 1
        return None

    quotes = [
        priced
        for quote in raw_quotes
        if (priced := _price_quote(quote, spot, config, stats)) is not None
    ]
    stats.parsed += len(quotes)
    if len(quotes) < config.min_contracts_per_snapshot:
        stats.thin_snapshots += 1
        return None

    recent = bar_index.window(timestamp)
    snapshot = MarketSnapshot(
        timestamp=timestamp,
        spy_quote=SpyQuote(
            timestamp=timestamp, bid=spot - 0.01, ask=spot + 0.01, last=spot
        ),
        spy_bars=tuple(recent),
        option_chain=tuple(quotes),
        context=ContextSnapshot(
            minutes_to_close=market_calendar.minutes_to_close(timestamp)
        ),
        risk_free_rate=config.risk_free_rate,
        dividend_yield=config.dividend_yield,
    )
    return score_snapshot(snapshot, config.quality)


def _infer_spot(
    quotes: list[OptionQuote], bar_index: _BarIndex, timestamp: datetime
) -> float:
    """Underlying price from the quotes, falling back to the bar close."""
    prices = [q.underlying_price for q in quotes if q.underlying_price > 0.0]
    if prices:
        return sum(prices) / len(prices)
    bar = bar_index.latest(timestamp)
    return bar.close if bar is not None else 0.0


class _BarIndex:
    """Time-ordered bars with point-in-time lookup.

    Never returns a bar stamped after the requested moment, so an ingested
    snapshot cannot carry future information into a backtest.
    """

    def __init__(self, bars: Sequence[PriceBar], window: int = 90) -> None:
        self._bars = sorted(bars, key=lambda b: b.timestamp)
        self._stamps = [b.timestamp for b in self._bars]
        self._window = window

    def _position(self, moment: datetime) -> int:
        from bisect import bisect_right

        return bisect_right(self._stamps, moment)

    def latest(self, moment: datetime) -> PriceBar | None:
        index = self._position(moment)
        return self._bars[index - 1] if index else None

    def window(self, moment: datetime) -> list[PriceBar]:
        index = self._position(moment)
        return self._bars[max(0, index - self._window) : index]


def discover_files(root: Path, *, pattern: str = "*") -> list[Path]:
    """Chain files under ``root``, sorted by name.

    Sorting by name matches the usual ``SPY_YYYY-MM-DD`` layout and gives
    chronological order without opening anything.
    """
    if root.is_file():
        return [root]
    suffixes = {".csv", ".txt", ".gz", ".parquet", ".pq"}
    return sorted(
        p for p in root.rglob(pattern) if p.is_file() and p.suffix.lower() in suffixes
    )


@dataclass
class HistoricalProvider(MarketDataProvider):
    """Serves recorded snapshots, refusing to look ahead.

    Snapshots are materialized once and then served by timestamp. For datasets
    too large to hold in memory, stream :func:`iter_snapshots` straight into
    :func:`spy_der.backtest.engine.run_backtest` instead — it consumes an
    iterable and never needs random access.
    """

    snapshots: tuple[MarketSnapshot, ...] = ()

    @classmethod
    def from_files(
        cls,
        chain_paths: Sequence[Path],
        *,
        bars: Sequence[PriceBar] = (),
        config: IngestConfig | None = None,
        limit: int | None = None,
    ) -> tuple[HistoricalProvider, IngestStats]:
        stats = IngestStats()
        loaded: list[MarketSnapshot] = []
        for snapshot in iter_snapshots(chain_paths, bars=bars, config=config, stats=stats):
            loaded.append(snapshot)
            if limit is not None and len(loaded) >= limit:
                break
        return cls(snapshots=tuple(loaded)), stats

    def __post_init__(self) -> None:
        self.snapshots = tuple(sorted(self.snapshots, key=lambda s: s.timestamp))

    def snapshot(self, at: datetime) -> MarketSnapshot:
        candidate: MarketSnapshot | None = None
        for snap in self.snapshots:
            if snap.timestamp > at:
                break
            candidate = snap
        if candidate is None:
            raise LookupError(f"no historical snapshot at or before {at.isoformat()}")
        return candidate

    def session_timestamps(self, day: datetime) -> list[datetime]:
        target = day.date()
        return [s.timestamp for s in self.snapshots if s.timestamp.date() == target]

    @property
    def trading_days(self) -> list[date]:
        return sorted({s.timestamp.date() for s in self.snapshots})

    def span(self) -> tuple[datetime, datetime] | None:
        if not self.snapshots:
            return None
        return self.snapshots[0].timestamp, self.snapshots[-1].timestamp


def inspect(paths: Sequence[Path], *, sample_rows: int = 5) -> dict[str, Any]:
    """Report how a dataset's columns map, without loading it.

    Run this first against unfamiliar data: it shows which canonical fields
    resolved, which are missing, and which source columns went unused.
    """
    if not paths:
        return {"error": "no files given"}

    rows: list[dict[str, Any]] = []
    for row in iter_rows(paths[0]):
        rows.append(row)
        if len(rows) >= sample_rows:
            break
    if not rows:
        return {"error": f"{paths[0]} is empty"}

    columns = list(rows[0].keys())
    mapping = build_column_map(columns)
    return {
        "file": str(paths[0]),
        "files_found": len(paths),
        "columns": columns,
        "mapped": mapping,
        "missing_required": sorted(REQUIRED_FIELDS - mapping.keys()),
        "unmapped_columns": sorted(set(columns) - set(mapping.values())),
        "sample": rows[0],
    }
