"""Historical ingestion tests.

The fixtures here deliberately use vendor spellings rather than the canonical
field names, because that is the whole point of the loader: real files never
arrive pre-normalized, and a test written against canonical columns would pass
while the alias table was empty.
"""

from __future__ import annotations

import csv
import gzip
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from spy_der.analytics import pricing
from spy_der.data.providers import historical as hist
from spy_der.domain.enums import OptionRight
from spy_der.domain.market import PriceBar

RATE = 0.04
DIV = 0.013


# ---------------------------------------------------------------- fixtures


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def chain_rows(
    *,
    stamp: datetime,
    expiry: datetime,
    spot: float = 500.0,
    vol: float = 0.18,
    strikes: range | None = None,
) -> list[dict[str, object]]:
    """A model-priced chain in CBOE-ish column spellings, with no IV column.

    Prices come from the same model the loader inverts, so a derived IV must
    round-trip back to ``vol``. That makes the IV-derivation assertions exact
    rather than approximate.
    """
    tau = (expiry - stamp).total_seconds() / (365.0 * 24.0 * 3600.0)
    out: list[dict[str, object]] = []
    # 13 strikes x 2 rights = 26 quotes, comfortably over the 20-contract floor
    # that ``min_contracts_per_snapshot`` enforces.
    for strike in strikes or range(470, 531, 5):
        for right, token in ((OptionRight.CALL, "C"), (OptionRight.PUT, "P")):
            mid = float(pricing.price(spot, strike, tau, RATE, DIV, vol, right))
            out.append(
                {
                    "quote_datetime": stamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "expiration": expiry.date().isoformat(),
                    "strike": strike,
                    "option_type": token,
                    "bid_price": round(mid - 0.02, 4),
                    "ask_price": round(mid + 0.02, 4),
                    "trade_volume": 100,
                    "open_int": 2500,
                    "active_underlying_price": spot,
                }
            )
    return out


# ------------------------------------------------------------ column mapping


def test_vendor_columns_map_onto_the_canonical_schema():
    mapping = hist.build_column_map(
        [
            "quote_datetime",
            "expiration",
            "strike",
            "option_type",
            "bid_price",
            "ask_price",
            "trade_volume",
            "open_int",
            "active_underlying_price",
        ]
    )
    assert mapping["timestamp"] == "quote_datetime"
    assert mapping["right"] == "option_type"
    assert mapping["bid"] == "bid_price"
    assert mapping["underlying_price"] == "active_underlying_price"
    assert not hist.REQUIRED_FIELDS - mapping.keys()


def test_mapping_is_case_and_whitespace_insensitive():
    mapping = hist.build_column_map([" Quote_DateTime ", "STRIKE", "Option_Type"])
    # The *source* spelling is preserved, since that is the key rows arrive under.
    assert mapping["timestamp"] == " Quote_DateTime "
    assert mapping["strike"] == "STRIKE"
    assert mapping["right"] == "Option_Type"


def test_alias_precedence_follows_declaration_order():
    # Two candidates present: the earlier alias wins, deterministically.
    mapping = hist.build_column_map(["bid", "bid_price", "strike"])
    assert mapping["bid"] == "bid"
    order = hist.COLUMN_ALIASES["bid"]
    assert order.index("bid") < order.index("bid_price")


def test_missing_required_field_names_the_columns_it_saw(tmp_path):
    path = write_csv(
        tmp_path / "bad.csv",
        [{"quote_datetime": "2026-03-02 14:30:00", "strike": 500, "bid": 1.0, "ask": 1.1}],
    )
    with pytest.raises(hist.SchemaError) as excinfo:
        list(hist.iter_snapshots([path]))
    message = str(excinfo.value)
    assert "expiration" in message and "right" in message
    # The operator has to be able to see what to add to COLUMN_ALIASES.
    assert "quote_datetime" in message
    assert "COLUMN_ALIASES" in message


def test_inspect_reports_mapping_without_loading(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    path = write_csv(
        tmp_path / "SPY_2026-03-02.csv",
        chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC)),
    )
    report = hist.inspect([path])
    assert report["missing_required"] == []
    assert report["unmapped_columns"] == []
    assert report["mapped"]["timestamp"] == "quote_datetime"
    assert report["files_found"] == 1


def test_inspect_flags_unmapped_columns(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    rows = chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC))
    for row in rows:
        row["exchange_seq_no"] = 41
    path = write_csv(tmp_path / "extra.csv", rows)
    assert hist.inspect([path])["unmapped_columns"] == ["exchange_seq_no"]


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize("token", ["C", "c", "call", "Calls", "1", "+1"])
def test_call_tokens(token):
    assert hist.parse_right(token) is OptionRight.CALL


@pytest.mark.parametrize("token", ["P", "p", "put", "Puts", "-1", "0"])
def test_put_tokens(token):
    assert hist.parse_right(token) is OptionRight.PUT


def test_unknown_right_raises():
    with pytest.raises(ValueError, match="cannot interpret"):
        hist.parse_right("straddle")


@pytest.mark.parametrize(
    "text",
    [
        "2026-03-02T14:30:00+00:00",
        "2026-03-02T14:30:00Z",
        "2026-03-02 14:30:00",
        "03/02/2026 14:30:00",
    ],
)
def test_timestamp_formats_all_land_on_the_same_instant(text):
    assert hist.parse_timestamp(text) == datetime(2026, 3, 2, 14, 30, tzinfo=UTC)


def test_naive_timestamps_are_tagged_utc_not_shifted():
    # assume_utc means "this clock is already UTC", so the wall time is kept.
    assert hist.parse_timestamp("2026-03-02 14:30:00", assume_utc=True).hour == 14


def test_aware_timestamps_are_converted_regardless_of_assume_utc():
    eastern = hist.parse_timestamp("2026-03-02T09:30:00-05:00", assume_utc=True)
    assert eastern == datetime(2026, 3, 2, 14, 30, tzinfo=UTC)


def test_unparseable_timestamp_raises():
    with pytest.raises(ValueError, match="cannot parse timestamp"):
        hist.parse_timestamp("last tuesday")


def test_bare_expiration_date_settles_at_the_close_not_midnight(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    path = write_csv(
        tmp_path / "chain.csv",
        chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC)),
    )
    snapshot = next(iter(hist.iter_snapshots([path])))
    expiries = {q.expiration for q in snapshot.option_chain}
    assert expiries == {datetime(2026, 3, 20, 20, 0, tzinfo=UTC)}
    # Midnight would understate every tau by two thirds of a day.
    assert all(q.expiration.hour == 20 for q in snapshot.option_chain)


def test_expiration_clock_time_is_configurable(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    path = write_csv(
        tmp_path / "chain.csv",
        chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC)),
    )
    config = hist.IngestConfig(expiration_time_utc=(21, 0))
    snapshot = next(iter(hist.iter_snapshots([path], config=config)))
    assert all(q.expiration.hour == 21 for q in snapshot.option_chain)


# ------------------------------------------------- implied volatility handling


def test_missing_iv_is_solved_back_to_the_generating_volatility(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    path = write_csv(tmp_path / "chain.csv", chain_rows(stamp=stamp, expiry=expiry, vol=0.18))

    stats = hist.IngestStats()
    snapshot = next(iter(hist.iter_snapshots([path], stats=stats)))

    assert stats.iv_derived == len(snapshot.option_chain)
    # The 2c half-spread around a model mid is the only source of error.
    for quote in snapshot.option_chain:
        assert quote.implied_volatility == pytest.approx(0.18, abs=5e-3)


def test_supplied_iv_is_used_verbatim(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    rows = chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC))
    for row in rows:
        row["implied_vol"] = 0.4242
    path = write_csv(tmp_path / "chain.csv", rows)

    stats = hist.IngestStats()
    snapshot = next(iter(hist.iter_snapshots([path], stats=stats)))
    assert stats.iv_derived == 0
    assert all(q.implied_volatility == pytest.approx(0.4242) for q in snapshot.option_chain)


def test_unpriceable_quotes_are_dropped_rather_than_zeroed(tmp_path):
    """A quote below intrinsic has no IV; emitting 0.0 would be a false claim.

    Zero volatility prices at intrinsic with zero gamma and zero vega, so such
    a contract would silently dilute every exposure aggregate it entered.
    """
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    rows = chain_rows(stamp=stamp, expiry=expiry)
    # Mark two deep calls at a nickel: far below intrinsic, so IV has no root.
    broken = [r for r in rows if r["option_type"] == "C" and r["strike"] <= 485][:2]
    for row in broken:
        row["bid_price"], row["ask_price"] = 0.04, 0.06
    path = write_csv(tmp_path / "chain.csv", rows)

    stats = hist.IngestStats()
    snapshot = next(iter(hist.iter_snapshots([path], stats=stats)))

    assert stats.dropped_unpriceable == len(broken)
    assert stats.skipped == 0  # not confused with an unparseable row
    assert len(snapshot.option_chain) == len(rows) - len(broken)
    assert all(q.implied_volatility > 0.0 for q in snapshot.option_chain)


def test_unpriceable_quotes_can_be_kept_when_asked(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    rows = chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC))
    rows[0]["bid_price"], rows[0]["ask_price"] = 0.04, 0.06
    path = write_csv(tmp_path / "chain.csv", rows)

    config = hist.IngestConfig(drop_unpriceable=False)
    snapshot = next(iter(hist.iter_snapshots([path], config=config)))
    assert len(snapshot.option_chain) == len(rows)


def test_iv_derivation_can_be_disabled(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    path = write_csv(
        tmp_path / "chain.csv",
        chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC)),
    )
    config = hist.IngestConfig(derive_missing_iv=False)
    # Every quote is now unpriceable, so the chain empties and nothing survives
    # the thin-snapshot floor.
    assert list(hist.iter_snapshots([path], config=config)) == []


# ------------------------------------------------------------------ snapshots


def test_rows_group_into_one_snapshot_per_timestamp(tmp_path):
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    stamps = [datetime(2026, 3, 2, 14, 30, tzinfo=UTC) + timedelta(hours=h) for h in range(3)]
    rows: list[dict[str, object]] = []
    for stamp in stamps:
        rows.extend(chain_rows(stamp=stamp, expiry=expiry, spot=500.0 + stamps.index(stamp)))
    path = write_csv(tmp_path / "chain.csv", rows)

    snapshots = list(hist.iter_snapshots([path]))
    assert [s.timestamp for s in snapshots] == stamps
    assert all(len(s.option_chain) == 26 for s in snapshots)


def test_snapshots_are_emitted_in_timestamp_order_from_shuffled_rows(tmp_path):
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    stamps = [datetime(2026, 3, 2, 14, 30, tzinfo=UTC) + timedelta(hours=h) for h in range(3)]
    rows: list[dict[str, object]] = []
    for stamp in reversed(stamps):
        rows.extend(chain_rows(stamp=stamp, expiry=expiry))
    path = write_csv(tmp_path / "chain.csv", rows)

    emitted = [s.timestamp for s in hist.iter_snapshots([path])]
    assert emitted == sorted(emitted) == stamps


def test_thin_snapshots_are_skipped_and_counted(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    rows = chain_rows(
        stamp=stamp,
        expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC),
        strikes=range(495, 506, 5),  # 6 quotes, below the default floor of 20
    )
    path = write_csv(tmp_path / "thin.csv", rows)

    stats = hist.IngestStats()
    assert list(hist.iter_snapshots([path], stats=stats)) == []
    assert stats.thin_snapshots == 1

    relaxed = hist.IngestConfig(min_contracts_per_snapshot=4)
    assert len(list(hist.iter_snapshots([path], config=relaxed))) == 1


def test_spot_comes_from_the_quotes_when_present(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    path = write_csv(
        tmp_path / "chain.csv",
        chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC), spot=507.25),
    )
    snapshot = next(iter(hist.iter_snapshots([path])))
    assert snapshot.spy_quote.last == pytest.approx(507.25)
    assert snapshot.spy_quote.bid < snapshot.spy_quote.last < snapshot.spy_quote.ask


def test_spot_falls_back_to_the_bar_close_when_the_chain_has_none(tmp_path):
    """Vendors that ship the underlying separately carry no per-row spot.

    The bar close has to reach IV derivation, not just the SPY quote — otherwise
    every contract in such a file solves against a zero underlying and is
    dropped as unpriceable.
    """
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    rows = chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC), spot=500.0)
    for row in rows:
        del row["active_underlying_price"]
    path = write_csv(tmp_path / "chain.csv", rows)

    bars = [
        PriceBar(
            timestamp=stamp - timedelta(minutes=1),
            open=499.0,
            high=501.0,
            low=498.0,
            close=500.0,
            volume=1e6,
        )
    ]
    stats = hist.IngestStats()
    snapshot = next(iter(hist.iter_snapshots([path], bars=bars, stats=stats)))

    assert snapshot.spy_quote.last == pytest.approx(500.0)
    assert stats.dropped_unpriceable == 0
    assert len(snapshot.option_chain) == len(rows)
    # The fallback spot is stamped onto the quotes, not left at zero.
    assert all(q.underlying_price == pytest.approx(500.0) for q in snapshot.option_chain)
    assert all(q.implied_volatility == pytest.approx(0.18, abs=5e-3) for q in snapshot.option_chain)


def test_snapshot_without_any_spot_source_is_dropped(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    rows = chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC))
    for row in rows:
        del row["active_underlying_price"]
    path = write_csv(tmp_path / "chain.csv", rows)

    # No underlying column and no bars: nothing is emitted, rather than a
    # snapshot priced off a fabricated spot.
    stats = hist.IngestStats()
    assert list(hist.iter_snapshots([path], stats=stats)) == []
    assert stats.no_spot_snapshots == 1


def test_minutes_to_close_is_taken_from_the_market_calendar(tmp_path):
    # 2026-03-02 is a Monday; 14:30 UTC is 09:30 ET, the open.
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    path = write_csv(
        tmp_path / "chain.csv",
        chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC)),
    )
    snapshot = next(iter(hist.iter_snapshots([path])))
    assert snapshot.context.minutes_to_close == pytest.approx(390.0)


def test_multiple_files_are_read_in_the_order_given(tmp_path):
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    paths = []
    for day in (2, 3, 4):
        stamp = datetime(2026, 3, day, 14, 30, tzinfo=UTC)
        paths.append(
            write_csv(tmp_path / f"SPY_2026-03-{day:02d}.csv", chain_rows(stamp=stamp, expiry=expiry))
        )
    days = [s.timestamp.day for s in hist.iter_snapshots(paths)]
    assert days == [2, 3, 4]


def test_gzipped_csv_reads_the_same_as_plain(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    rows = chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC))
    plain = write_csv(tmp_path / "chain.csv", rows)
    packed = tmp_path / "chain.csv.gz"
    with gzip.open(packed, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    a = next(iter(hist.iter_snapshots([plain])))
    b = next(iter(hist.iter_snapshots([packed])))
    assert a.timestamp == b.timestamp
    assert len(a.option_chain) == len(b.option_chain)


def test_unsupported_file_type_is_rejected(tmp_path):
    path = tmp_path / "chain.xlsx"
    path.write_bytes(b"not a chain")
    with pytest.raises(hist.SchemaError, match="unsupported file type"):
        list(hist.iter_rows(path))


# ------------------------------------------------------------- bad-row handling


def test_unparseable_rows_are_counted_not_fatal(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    rows = chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC))
    rows[0]["option_type"] = "???"
    rows[1]["strike"] = "n/a"
    path = write_csv(tmp_path / "chain.csv", rows)

    stats = hist.IngestStats()
    snapshot = next(iter(hist.iter_snapshots([path], stats=stats)))
    assert stats.skipped == 2
    assert stats.errors  # the operator gets told why
    assert len(snapshot.option_chain) == len(rows) - 2


def test_non_positive_strikes_are_rejected(tmp_path):
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    rows = chain_rows(stamp=stamp, expiry=datetime(2026, 3, 20, 20, 0, tzinfo=UTC))
    rows[0]["strike"] = 0
    path = write_csv(tmp_path / "chain.csv", rows)

    stats = hist.IngestStats()
    next(iter(hist.iter_snapshots([path], stats=stats)))
    assert stats.skipped == 1


def test_error_notes_are_capped():
    stats = hist.IngestStats()
    for i in range(500):
        stats.note(f"bad row {i}")
    assert len(stats.errors) == 20


def test_stats_summary_mentions_every_counter():
    stats = hist.IngestStats(rows=10, parsed=8, snapshots=1, skipped=1, dropped_unpriceable=1)
    text = stats.summary()
    assert "10 rows" in text and "8 quotes" in text and "1 snapshots" in text


# ------------------------------------------------------------------- bars


def test_bars_load_from_vendor_columns_and_sort(tmp_path):
    path = write_csv(
        tmp_path / "bars.csv",
        [
            {"t": 0, "o": 0, "h": 0, "l": 0, "c": 0, "v": 0},  # placeholder, replaced below
        ],
    )
    rows = [
        {
            "date_time": f"2026-03-02 {14 + i}:30:00",
            "o": 500 + i,
            "h": 502 + i,
            "l": 499 + i,
            "c": 501 + i,
            "v": 1_000_000,
        }
        for i in (2, 0, 1)  # deliberately out of order
    ]
    path = write_csv(path, rows)

    bars = hist.load_bars(path)
    assert [b.timestamp.hour for b in bars] == [14, 15, 16]
    assert bars[0].close == pytest.approx(501.0)
    assert bars[0].open == pytest.approx(500.0)


def test_bar_columns_default_to_close_when_ohlc_is_absent(tmp_path):
    path = write_csv(
        tmp_path / "bars.csv",
        [{"date": "2026-03-02", "close": 501.5}],
    )
    bar = hist.load_bars(path)[0]
    assert bar.open == bar.high == bar.low == bar.close == pytest.approx(501.5)


def test_bar_file_missing_close_is_rejected(tmp_path):
    path = write_csv(tmp_path / "bars.csv", [{"date": "2026-03-02", "open": 500.0}])
    with pytest.raises(hist.SchemaError, match="close"):
        hist.load_bars(path)


def test_non_positive_bar_closes_are_dropped(tmp_path):
    path = write_csv(
        tmp_path / "bars.csv",
        [
            {"date_time": "2026-03-02 14:30:00", "close": 500.0},
            {"date_time": "2026-03-02 15:30:00", "close": 0.0},
            {"date_time": "2026-03-02 16:30:00", "close": ""},
        ],
    )
    assert len(hist.load_bars(path)) == 1


def test_empty_bar_file_returns_nothing(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text("date_time,close\n")
    assert hist.load_bars(path) == []


# --------------------------------------------------------- point-in-time index


def make_bars(count: int, start: datetime) -> list[PriceBar]:
    return [
        PriceBar(
            timestamp=start + timedelta(minutes=i),
            open=500.0,
            high=501.0,
            low=499.0,
            close=500.0 + i,
            volume=1000.0,
        )
        for i in range(count)
    ]


def test_bar_index_never_returns_a_future_bar():
    start = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    index = hist._BarIndex(make_bars(10, start))
    for offset in range(10):
        moment = start + timedelta(minutes=offset, seconds=30)
        latest = index.latest(moment)
        assert latest is not None
        assert latest.timestamp <= moment
        assert all(b.timestamp <= moment for b in index.window(moment))


def test_bar_index_includes_a_bar_stamped_exactly_now():
    start = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    index = hist._BarIndex(make_bars(5, start))
    # A bar closing at the decision instant is already observable.
    assert index.latest(start + timedelta(minutes=2)).close == pytest.approx(502.0)


def test_bar_index_before_history_returns_none():
    start = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    index = hist._BarIndex(make_bars(5, start))
    assert index.latest(start - timedelta(minutes=1)) is None
    assert index.window(start - timedelta(minutes=1)) == []


def test_bar_index_window_is_bounded():
    start = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    index = hist._BarIndex(make_bars(500, start), window=90)
    assert len(index.window(start + timedelta(minutes=499))) == 90


def test_bar_index_sorts_unordered_input():
    start = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    scrambled = list(reversed(make_bars(10, start)))
    index = hist._BarIndex(scrambled)
    window = index.window(start + timedelta(minutes=9))
    assert [b.timestamp for b in window] == sorted(b.timestamp for b in window)


def test_snapshot_bars_are_point_in_time(tmp_path):
    """The bar history attached to a snapshot must stop at the snapshot."""
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    stamps = [datetime(2026, 3, 2, 14, 30, tzinfo=UTC) + timedelta(hours=h) for h in range(3)]
    rows: list[dict[str, object]] = []
    for stamp in stamps:
        rows.extend(chain_rows(stamp=stamp, expiry=expiry))
    path = write_csv(tmp_path / "chain.csv", rows)
    bars = make_bars(400, datetime(2026, 3, 2, 14, 30, tzinfo=UTC))

    for snapshot in hist.iter_snapshots([path], bars=bars):
        assert snapshot.spy_bars
        assert all(b.timestamp <= snapshot.timestamp for b in snapshot.spy_bars)


# -------------------------------------------------------------- the provider


def build_provider(tmp_path: Path, days: tuple[int, ...] = (2, 3, 4)):
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    paths = []
    for day in days:
        rows: list[dict[str, object]] = []
        for hour in (14, 15):
            rows.extend(
                chain_rows(stamp=datetime(2026, 3, day, hour, 30, tzinfo=UTC), expiry=expiry)
            )
        paths.append(write_csv(tmp_path / f"SPY_2026-03-{day:02d}.csv", rows))
    return hist.HistoricalProvider.from_files(paths)


def test_provider_serves_the_latest_snapshot_at_or_before_a_moment(tmp_path):
    provider, stats = build_provider(tmp_path)
    assert stats.snapshots == 6

    asked = datetime(2026, 3, 3, 15, 45, tzinfo=UTC)
    served = provider.snapshot(asked)
    assert served.timestamp == datetime(2026, 3, 3, 15, 30, tzinfo=UTC)
    assert served.timestamp <= asked


def test_provider_never_looks_ahead(tmp_path):
    provider, _ = build_provider(tmp_path)
    for probe_minutes in range(0, 60 * 48, 37):
        asked = datetime(2026, 3, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=probe_minutes)
        assert provider.snapshot(asked).timestamp <= asked


def test_provider_raises_before_its_history_starts(tmp_path):
    provider, _ = build_provider(tmp_path)
    with pytest.raises(LookupError, match="no historical snapshot"):
        provider.snapshot(datetime(2026, 3, 1, tzinfo=UTC))


def test_provider_sorts_snapshots_regardless_of_file_order(tmp_path):
    provider, _ = build_provider(tmp_path, days=(4, 2, 3))
    stamps = [s.timestamp for s in provider.snapshots]
    assert stamps == sorted(stamps)


def test_provider_reports_span_and_trading_days(tmp_path):
    provider, _ = build_provider(tmp_path)
    span = provider.span()
    assert span is not None
    assert span[0] == datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    assert span[1] == datetime(2026, 3, 4, 15, 30, tzinfo=UTC)
    assert [d.day for d in provider.trading_days] == [2, 3, 4]
    assert len(provider.session_timestamps(datetime(2026, 3, 3, tzinfo=UTC))) == 2


def test_empty_provider_has_no_span():
    assert hist.HistoricalProvider().span() is None


def test_provider_limit_stops_ingestion_early(tmp_path):
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for hour in range(14, 20):
        rows.extend(chain_rows(stamp=datetime(2026, 3, 2, hour, 30, tzinfo=UTC), expiry=expiry))
    path = write_csv(tmp_path / "chain.csv", rows)

    provider, _ = hist.HistoricalProvider.from_files([path], limit=2)
    assert len(provider.snapshots) == 2


# ---------------------------------------------------------------- discovery


def test_discover_finds_chain_files_and_ignores_others(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("a.csv", "b.csv.gz", "sub/c.csv", "notes.md", "index.json"):
        (tmp_path / name).write_text("x")
    found = {p.name for p in hist.discover_files(tmp_path)}
    assert found == {"a.csv", "b.csv.gz", "c.csv"}


def test_discover_sorts_by_name_for_chronological_order(tmp_path):
    for name in ("SPY_2026-03-04.csv", "SPY_2026-03-02.csv", "SPY_2026-03-03.csv"):
        (tmp_path / name).write_text("x")
    names = [p.name for p in hist.discover_files(tmp_path)]
    assert names == sorted(names)


def test_discover_accepts_a_single_file(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("x")
    assert hist.discover_files(path) == [path]


def test_discover_honours_a_pattern(tmp_path):
    (tmp_path / "SPY_2026-03-02.csv").write_text("x")
    (tmp_path / "QQQ_2026-03-02.csv").write_text("x")
    found = hist.discover_files(tmp_path, pattern="SPY_*.csv")
    assert [p.name for p in found] == ["SPY_2026-03-02.csv"]


# ------------------------------------------------------------------ numerics


def test_to_float_treats_blanks_and_nan_as_the_default():
    assert hist._to_float("") == 0.0
    assert hist._to_float(None, 7.0) == 7.0
    assert hist._to_float("junk", 3.0) == 3.0
    assert hist._to_float(float("nan"), 1.5) == 1.5
    assert hist._to_float("2.5") == pytest.approx(2.5)


def test_to_int_truncates_vendor_floats():
    # Open interest often arrives as "2500.0".
    assert hist._to_int("2500.0") == 2500
    assert hist._to_int("", 10) == 10


def test_contract_symbols_follow_the_osi_shape():
    symbol = hist._symbol(datetime(2026, 3, 20, 20, 0, tzinfo=UTC), 500.5, OptionRight.CALL)
    assert symbol == "SPY260320C00500500"
    assert len(symbol) == 18


def test_contract_symbols_are_unique_per_contract():
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    symbols = {
        hist._symbol(expiry, strike, right)
        for strike in (500.0, 500.5)
        for right in OptionRight
    }
    assert len(symbols) == 4


# ------------------------------------------------------- end-to-end sanity


def test_ingested_snapshots_are_internally_consistent(tmp_path):
    """Round-trip: model prices in, snapshot out, Greeks that make sense."""
    stamp = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    expiry = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)
    path = write_csv(
        tmp_path / "chain.csv",
        chain_rows(stamp=stamp, expiry=expiry, spot=500.0, vol=0.20),
    )
    bars = make_bars(60, stamp - timedelta(minutes=60))
    snapshot = next(iter(hist.iter_snapshots([path], bars=bars)))

    assert snapshot.data_quality_score > 0.9
    calls = [q for q in snapshot.option_chain if q.right is OptionRight.CALL]
    puts = [q for q in snapshot.option_chain if q.right is OptionRight.PUT]
    assert len(calls) == len(puts) == 13

    # Monotone call prices in strike, and a flat-by-construction smile.
    by_strike = sorted(calls, key=lambda q: q.strike)
    mids = [0.5 * (q.bid + q.ask) for q in by_strike]
    assert all(a > b for a, b in zip(mids[:-1], mids[1:], strict=True))
    vols = [q.implied_volatility for q in snapshot.option_chain]
    assert max(vols) - min(vols) < 0.01
    assert all(math.isfinite(v) for v in vols)
