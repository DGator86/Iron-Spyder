"""Import SPY-DER VPS market recordings into Iron-Spyder snapshots.

The legacy SPY-DER box wrote canonical session tapes at
``/var/lib/spy-der/market/<YYYY-MM-DD>.jsonl``. Each line is::

    {"schema_version", "seq", "snapshot_id", "record_hash", "snapshot": {...}}

Iron-Spyder's decision path only understands :class:`MarketSnapshot`. This
module is the bridge: it streams those JSONL files, maps contracts/bars/
underlying quotes, solves missing IV with the same pricing model the rest of
the system uses, and exposes a :class:`MarketDataProvider` for replay.

The importer is read-only with respect to the source tree. Copied archives
live under ``/var/lib/iron-spyder/imports/spy-der/`` on the dedicated CPU VPS
and are never written into the live runtime directories.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time
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

__all__ = [
    "ImportStats",
    "SpyDerRecordingProvider",
    "discover_session_files",
    "inspect_recordings",
    "iter_records",
    "iter_snapshots",
    "snapshot_from_record",
]

#: Expiration clock used when a recording only carries a calendar date.
_EXPIRY_CLOSE_UTC = time(20, 0)


@dataclass
class ImportStats:
    """Counters for one import / inspect pass."""

    files: int = 0
    records: int = 0
    snapshots: int = 0
    contracts_kept: int = 0
    contracts_dropped: int = 0
    iv_derived: int = 0
    iv_unavailable: int = 0
    skipped_status: int = 0
    parse_errors: int = 0
    notes: list[str] = field(default_factory=list)

    def note(self, message: str, *, cap: int = 20) -> None:
        if len(self.notes) < cap:
            self.notes.append(message)

    def summary(self) -> str:
        return (
            f"files={self.files} records={self.records} snapshots={self.snapshots} "
            f"kept={self.contracts_kept} dropped={self.contracts_dropped} "
            f"iv_derived={self.iv_derived} iv_unavailable={self.iv_unavailable} "
            f"skipped_status={self.skipped_status} errors={self.parse_errors}"
        )


def discover_session_files(root: str | Path, *, pattern: str = "*.jsonl") -> list[Path]:
    """Session tapes under ``root``, newest last. ``.bak`` files are ignored."""
    path = Path(root)
    if path.is_file():
        return [path]
    files = sorted(
        p for p in path.glob(pattern) if p.is_file() and not p.name.endswith(".bak")
    )
    return files


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from one session tape."""
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: record is not an object")
            yield payload


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing timestamp: {value!r}")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


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


def _parse_right(value: Any) -> OptionRight:
    text = str(value or "").strip().upper()
    if text in {"C", "CALL"}:
        return OptionRight.CALL
    if text in {"P", "PUT"}:
        return OptionRight.PUT
    raise ValueError(f"unknown option right {value!r}")


def _expiration(value: Any) -> datetime:
    """SPY-DER stores expirations as dates; settle them at the cash close."""
    if isinstance(value, datetime):
        stamp = value if value.tzinfo else value.replace(tzinfo=UTC)
        if stamp.hour == 0 and stamp.minute == 0 and stamp.second == 0:
            return datetime.combine(stamp.date(), _EXPIRY_CLOSE_UTC, tzinfo=UTC)
        return stamp
    text = str(value)
    if "T" in text:
        return _parse_ts(text)
    day = date.fromisoformat(text[:10])
    return datetime.combine(day, _EXPIRY_CLOSE_UTC, tzinfo=UTC)


def _bars(raw: Sequence[dict[str, Any]] | None) -> tuple[PriceBar, ...]:
    out: list[PriceBar] = []
    for row in raw or ():
        try:
            ts = _parse_ts(row.get("timestamp"))
        except ValueError:
            continue
        close = _to_float(row.get("close"))
        if close <= 0.0:
            continue
        out.append(
            PriceBar(
                timestamp=ts,
                open=_to_float(row.get("open"), close),
                high=_to_float(row.get("high"), close),
                low=_to_float(row.get("low"), close),
                close=close,
                volume=_to_float(row.get("volume")),
            )
        )
    return tuple(out)


def _option_quote(
    row: dict[str, Any],
    *,
    timestamp: datetime,
    spot: float,
    stats: ImportStats,
    derive_missing_iv: bool,
    drop_unpriceable: bool,
) -> OptionQuote | None:
    contract = row.get("contract") or {}
    if not isinstance(contract, dict):
        stats.contracts_dropped += 1
        stats.note("contract missing object")
        return None
    try:
        right = _parse_right(contract.get("option_type") or contract.get("right"))
        strike = float(contract["strike"])
        expiration = _expiration(contract["expiration"])
    except (KeyError, TypeError, ValueError) as exc:
        stats.contracts_dropped += 1
        stats.note(f"bad contract: {exc}")
        return None
    if strike <= 0.0:
        stats.contracts_dropped += 1
        return None

    bid = _to_float(row.get("bid"))
    ask = _to_float(row.get("ask"))
    mark = _to_float(row.get("mark"), 0.5 * (bid + ask) if bid or ask else 0.0)
    last = _to_float(row.get("last"), mark)
    iv = _to_float(row.get("implied_volatility"))
    volume = _to_int(row.get("volume"))
    open_interest = _to_int(row.get("open_interest"))
    symbol = str(contract.get("contract_id") or "")
    if not symbol:
        code = "C" if right is OptionRight.CALL else "P"
        symbol = f"SPY{expiration:%y%m%d}{code}{int(round(strike * 1000)):08d}"

    # Tradier-backed SPY-DER tapes omit bid/ask size. Size=1 structurally caps
    # fill probability below the optimizer's 0.25 gate, so approximate displayed
    # depth from open interest when the tape has no size fields.
    bid_size = _to_int(row.get("bid_size") or row.get("bidSize") or row.get("size"))
    ask_size = _to_int(row.get("ask_size") or row.get("askSize") or row.get("size"))
    if bid_size <= 0 or ask_size <= 0:
        inferred = _infer_quote_size(open_interest=open_interest, volume=volume)
        if bid_size <= 0:
            bid_size = inferred
        if ask_size <= 0:
            ask_size = inferred

    quote = OptionQuote(
        contract_symbol=symbol,
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
        source=str(row.get("source") or "spyder_recording"),
    )

    dq_bounds = DataQualityConfig()
    iv_in_bounds = dq_bounds.min_iv <= quote.implied_volatility <= dq_bounds.max_iv
    if (quote.implied_volatility <= 0.0 or not iv_in_bounds) and derive_missing_iv:
        if quote.mid > 0.0 and quote.tau > 0.0:
            solved = pricing.implied_volatility(
                quote.mid,
                spot,
                quote.strike,
                quote.tau,
                0.04,
                0.013,
                quote.right,
            )
            if solved is not None and dq_bounds.min_iv <= solved <= dq_bounds.max_iv:
                quote = replace(quote, implied_volatility=solved)
                stats.iv_derived += 1
                iv_in_bounds = True
            else:
                stats.iv_unavailable += 1
        else:
            stats.iv_unavailable += 1

    # Drop unpriceable or Tradier-capped (IV=10) contracts so they cannot sink
    # the session DQ score or poison ATM / GEX reads.
    if drop_unpriceable and (
        quote.implied_volatility <= 0.0
        or not (dq_bounds.min_iv <= quote.implied_volatility <= dq_bounds.max_iv)
    ):
        stats.contracts_dropped += 1
        return None

    stats.contracts_kept += 1
    return quote


def _infer_quote_size(*, open_interest: int, volume: int) -> int:
    """Approximate NBBO size when the recording omitted it."""
    if open_interest > 0:
        return max(10, min(250, open_interest // 40 or 10))
    if volume > 0:
        return max(10, min(100, volume))
    return 25


def snapshot_from_record(
    record: dict[str, Any],
    *,
    stats: ImportStats | None = None,
    open_only: bool = True,
    derive_missing_iv: bool = True,
    drop_unpriceable: bool = True,
    quality_config: DataQualityConfig | None = None,
) -> MarketSnapshot | None:
    """Convert one SPY-DER recording envelope into a scored MarketSnapshot."""
    stats = stats or ImportStats()
    snap = record.get("snapshot")
    if not isinstance(snap, dict):
        stats.parse_errors += 1
        stats.note("record missing snapshot object")
        return None

    status = str(snap.get("session_status") or "")
    if open_only and status not in {"OPEN", "RTH", "REGULAR"}:
        # PRE_OPEN / CLOSED tapes are useful for inspection but not for the
        # decision loop, which assumes an open session.
        stats.skipped_status += 1
        return None

    try:
        timestamp = _parse_ts(snap.get("timestamp") or snap.get("observed_at"))
    except ValueError as exc:
        stats.parse_errors += 1
        stats.note(str(exc))
        return None

    spot = _to_float(snap.get("underlying_price"))
    bid = _to_float(snap.get("underlying_bid"), spot)
    ask = _to_float(snap.get("underlying_ask"), spot)
    if spot <= 0.0:
        if bid > 0.0 and ask > 0.0:
            spot = 0.5 * (bid + ask)
        else:
            stats.parse_errors += 1
            stats.note("non-positive underlying")
            return None

    chain_raw = snap.get("option_chain") or []
    if not isinstance(chain_raw, list):
        stats.parse_errors += 1
        return None

    quotes: list[OptionQuote] = []
    for row in chain_raw:
        if not isinstance(row, dict):
            stats.contracts_dropped += 1
            continue
        quote = _option_quote(
            row,
            timestamp=timestamp,
            spot=spot,
            stats=stats,
            derive_missing_iv=derive_missing_iv,
            drop_unpriceable=drop_unpriceable,
        )
        if quote is not None:
            quotes.append(quote)

    if not quotes:
        stats.parse_errors += 1
        stats.note(f"empty chain at {timestamp.isoformat()}")
        return None

    minutes = snap.get("minutes_to_close")
    minutes_to_close = (
        float(minutes)
        if isinstance(minutes, (int, float)) and minutes is not None
        else 390.0
    )
    # SPY-DER sometimes reports residual minutes into the next day during
    # pre-open; clamp for the decision context.
    if minutes_to_close > 390.0:
        minutes_to_close = 390.0

    breadth_raw = snap.get("breadth")
    breadth: dict[str, Any] = breadth_raw if isinstance(breadth_raw, dict) else {}
    vol_term = snap.get("volatility_term_structure")
    vol_term_map: dict[str, Any] = vol_term if isinstance(vol_term, dict) else {}
    catalyst = snap.get("catalyst_state")
    catalyst_map: dict[str, Any] = catalyst if isinstance(catalyst, dict) else {}
    events: list[str] = []
    if catalyst_map.get("lockout_active"):
        reason = catalyst_map.get("reason") or "catalyst_lockout"
        events.append(str(reason))

    context = ContextSnapshot(
        vix=_to_float(vol_term_map.get("vix"), 0.0) or None,
        vvix=_to_float(vol_term_map.get("vvix"), 0.0) or None,
        # SPY-DER breadth uses rsp_spy_div / sector_align rather than classic A/D.
        advance_decline=(
            _to_float(breadth.get("advance_decline"), 0.0)
            or _to_float(breadth.get("rsp_spy_div"), 0.0)
            or None
        ),
        up_down_volume_ratio=(
            _to_float(breadth.get("up_down_volume_ratio"), 0.0)
            or _to_float(breadth.get("sector_align"), 0.0)
            or None
        ),
        minutes_to_close=max(0.0, minutes_to_close),
        scheduled_events=tuple(events),
    )

    snapshot = MarketSnapshot(
        timestamp=timestamp,
        spy_quote=SpyQuote(timestamp=timestamp, bid=bid, ask=ask, last=spot),
        spy_bars=_bars(snap.get("bars_1m") if isinstance(snap.get("bars_1m"), list) else None),
        option_chain=tuple(quotes),
        context=context,
    )
    stats.snapshots += 1
    return score_snapshot(snapshot, quality_config)


def iter_snapshots(
    paths: Sequence[Path],
    *,
    stats: ImportStats | None = None,
    open_only: bool = True,
    quality_config: DataQualityConfig | None = None,
) -> Iterator[MarketSnapshot]:
    """Stream Iron-Spyder snapshots from one or more SPY-DER session tapes."""
    stats = stats or ImportStats()
    for path in paths:
        stats.files += 1
        try:
            records = iter_records(path)
        except OSError as exc:
            stats.parse_errors += 1
            stats.note(f"{path}: {exc}")
            continue
        for record in records:
            stats.records += 1
            snapshot = snapshot_from_record(
                record,
                stats=stats,
                open_only=open_only,
                quality_config=quality_config,
            )
            if snapshot is not None:
                yield snapshot


def inspect_recordings(paths: Sequence[Path], *, sample: int = 3) -> dict[str, Any]:
    """Summarize tapes without loading every contract into memory.

    ``open_chain_contracts`` is the first OPEN tick's chain length (often empty
    at the open bell). ``max_open_chain_contracts`` is the peak OPEN chain size
    in the file — use that to spot sessions whose tapes never carried a chain.
    """
    stats = ImportStats()
    sessions: list[dict[str, Any]] = []
    for path in paths:
        stats.files += 1
        statuses: dict[str, int] = {}
        first_open: str | None = None
        last_ts: str | None = None
        first_chain_len = 0
        max_chain_len = 0
        try:
            for record in iter_records(path):
                stats.records += 1
                snap = record.get("snapshot") or {}
                status = str(snap.get("session_status") or "?")
                statuses[status] = statuses.get(status, 0) + 1
                last_ts = str(snap.get("timestamp") or "")
                if status == "OPEN":
                    chain_len = len(snap.get("option_chain") or [])
                    if chain_len > max_chain_len:
                        max_chain_len = chain_len
                    if first_open is None:
                        first_open = last_ts
                        first_chain_len = chain_len
        except ValueError as exc:
            stats.parse_errors += 1
            stats.note(str(exc))
            continue
        if statuses.get("OPEN", 0) and max_chain_len == 0:
            stats.note(f"{path.name}: OPEN records present but option_chain is empty")
        sessions.append(
            {
                "file": path.name,
                "records": sum(statuses.values()),
                "statuses": statuses,
                "first_open": first_open,
                "last_timestamp": last_ts,
                "open_chain_contracts": first_chain_len,
                "max_open_chain_contracts": max_chain_len,
            }
        )
        if len(sessions) >= sample and sample > 0:
            # Still count remaining files lightly.
            pass
    return {
        "files": stats.files,
        "records_scanned": stats.records,
        "parse_errors": stats.parse_errors,
        "sessions": sessions,
        "notes": stats.notes,
    }


@dataclass
class SpyDerRecordingProvider(MarketDataProvider):
    """Replay imported SPY-DER session tapes as Iron-Spyder snapshots."""

    root: Path
    open_only: bool = True
    quality_config: DataQualityConfig | None = None
    _index: dict[date, list[MarketSnapshot]] = field(default_factory=dict, init=False, repr=False)
    _ordered: list[MarketSnapshot] = field(default_factory=list, init=False, repr=False)
    stats: ImportStats = field(default_factory=ImportStats)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        files = discover_session_files(self.root)
        for snapshot in iter_snapshots(
            files,
            stats=self.stats,
            open_only=self.open_only,
            quality_config=self.quality_config,
        ):
            self._ordered.append(snapshot)
            self._index.setdefault(snapshot.timestamp.date(), []).append(snapshot)
        self._ordered.sort(key=lambda s: s.timestamp)

    def snapshot(self, at: datetime) -> MarketSnapshot:
        candidate: MarketSnapshot | None = None
        for snap in self._ordered:
            if snap.timestamp > at:
                break
            candidate = snap
        if candidate is None:
            raise LookupError(f"no imported snapshot at or before {at.isoformat()}")
        return candidate

    def session_timestamps(self, day: datetime) -> Sequence[datetime]:
        return [s.timestamp for s in self._index.get(day.date(), [])]

    def trading_days(self) -> list[date]:
        return sorted(self._index)

    def iter_all(self) -> Iterator[MarketSnapshot]:
        yield from self._ordered
