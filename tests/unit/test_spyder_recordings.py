"""SPY-DER JSONL → Iron-Spyder MarketSnapshot importer."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from spy_der.data.providers.spyder_recordings import (
    SpyDerRecordingProvider,
    discover_session_files,
    inspect_recordings,
    iter_records,
    iter_snapshots,
    snapshot_from_record,
)
from spy_der.domain.enums import OptionRight

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "spyder_recording_sample.jsonl"


def test_fixture_exists():
    assert FIXTURE.is_file()
    assert FIXTURE.stat().st_size > 0


def test_discover_and_inspect():
    files = discover_session_files(FIXTURE)
    assert files == [FIXTURE]
    report = inspect_recordings(files, sample=0)
    assert report["files"] == 1
    assert report["records_scanned"] == 3
    session = report["sessions"][0]
    assert session["statuses"].get("OPEN", 0) >= 1
    assert session["max_open_chain_contracts"] >= session["open_chain_contracts"]
    assert session["max_open_chain_contracts"] > 0


def test_snapshot_from_open_record():
    record = next(iter_records(FIXTURE))
    snap = snapshot_from_record(record, open_only=True)
    assert snap is not None
    assert snap.spot > 0
    assert len(snap.option_chain) > 0
    assert all(q.implied_volatility > 0 for q in snap.option_chain)
    # Tradier-capped IV=10 rows must be dropped or re-solved into bounds.
    assert all(q.implied_volatility <= 5.0 for q in snap.option_chain)
    assert any(q.right is OptionRight.CALL for q in snap.option_chain)
    # Tapes omit size; importer must infer depth so fill probability can clear
    # the optimizer gate.
    assert all(q.bid_size >= 10 and q.ask_size >= 10 for q in snap.option_chain)
    assert snap.data_quality_score >= 0.0
    assert snap.spy_bars


def test_context_maps_vix_from_volatility_term_structure():
    record = next(iter_records(FIXTURE))
    snap_raw = record["snapshot"]
    snap_raw["volatility_term_structure"] = {"vix": 14.2, "vvix": 88.0}
    snap_raw["catalyst_state"] = {"lockout_active": False, "reason": None}
    snap = snapshot_from_record(record, open_only=True)
    assert snap is not None
    assert snap.context.vix == pytest.approx(14.2)
    assert snap.context.vvix == pytest.approx(88.0)


def test_provider_replay_is_point_in_time():
    provider = SpyDerRecordingProvider(FIXTURE)
    assert provider.trading_days()
    first_snap = next(iter(provider.iter_all()))
    stamps = list(provider.session_timestamps(first_snap.timestamp))
    assert stamps
    first = provider.snapshot(stamps[0])
    assert first.timestamp == stamps[0]
    with pytest.raises(LookupError):
        provider.snapshot(stamps[0] - timedelta(days=1))


def test_iter_snapshots_stream():
    snaps = list(iter_snapshots([FIXTURE], open_only=True))
    assert len(snaps) >= 1
    assert snaps[0].timestamp <= snaps[-1].timestamp
