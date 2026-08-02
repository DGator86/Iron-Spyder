"""Walk-forward windows with purging and embargo (spec sections 37.5, 37.6).

Purging and embargo are mandatory here, not optional hygiene. This system's
labels are *path* labels: whether a barrier was touched, whether a stop fired,
what the maximum adverse excursion was. Those labels span a holding period, so
a training observation timestamped before a test observation can still have a
label that resolved *inside* the test window. Without purging, the model is
fitted on the answer.

Embargo handles the mirror case: an observation immediately after the test
window overlaps it in the other direction, and serial correlation in
volatility makes that leak real rather than theoretical.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Window:
    """One walk-forward fold."""

    index: int
    train_start: datetime
    train_end: datetime
    validate_start: datetime
    validate_end: datetime
    test_start: datetime
    test_end: datetime

    @property
    def label(self) -> str:
        return f"fold-{self.index:02d} test {self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d}"


def rolling_windows(
    start: datetime,
    end: datetime,
    *,
    train: timedelta = timedelta(days=182),
    validate: timedelta = timedelta(days=30),
    test: timedelta = timedelta(days=30),
    step: timedelta = timedelta(days=30),
) -> list[Window]:
    """Build rolling train / validate / test folds (spec 37.5).

    Defaults match the spec's example: six months train, one month validate,
    one month test, rolled forward one month at a time.
    """
    if min(train, validate, test, step) <= timedelta(0):
        raise ValueError("all window durations must be positive")

    windows: list[Window] = []
    cursor = start
    index = 0
    while cursor + train + validate + test <= end:
        train_end = cursor + train
        validate_end = train_end + validate
        test_end = validate_end + test
        windows.append(
            Window(
                index=index,
                train_start=cursor,
                train_end=train_end,
                validate_start=train_end,
                validate_end=validate_end,
                test_start=validate_end,
                test_end=test_end,
            )
        )
        cursor += step
        index += 1
    return windows


def purge_indices(
    timestamps: Sequence[datetime],
    *,
    train_range: tuple[datetime, datetime],
    test_range: tuple[datetime, datetime],
    label_horizon: timedelta,
    embargo: timedelta = timedelta(0),
) -> list[int]:
    """Indices of training observations that survive purging and embargo.

    An observation at ``t`` carries a label resolving at ``t + label_horizon``.
    It is purged when that interval overlaps the test range, and embargoed when
    it falls within ``embargo`` after the test range ends.
    """
    if label_horizon < timedelta(0) or embargo < timedelta(0):
        raise ValueError("label_horizon and embargo must be non-negative")

    train_start, train_end = train_range
    test_start, test_end = test_range
    embargo_end = test_end + embargo

    kept: list[int] = []
    for i, ts in enumerate(timestamps):
        if not (train_start <= ts < train_end):
            continue
        label_end = ts + label_horizon
        overlaps_test = ts < test_end and label_end > test_start
        inside_embargo = test_end <= ts < embargo_end
        if overlaps_test or inside_embargo:
            continue
        kept.append(i)
    return kept


def split_indices(
    timestamps: Sequence[datetime], window: Window, *, label_horizon: timedelta, embargo: timedelta
) -> tuple[list[int], list[int], list[int]]:
    """Return purged ``(train, validate, test)`` index lists for one fold."""
    train = purge_indices(
        timestamps,
        train_range=(window.train_start, window.train_end),
        test_range=(window.test_start, window.test_end),
        label_horizon=label_horizon,
        embargo=embargo,
    )
    # Validation is purged against the test range too: it is used to select
    # models that are then measured on test, so leakage there is just as fatal.
    validate = purge_indices(
        timestamps,
        train_range=(window.validate_start, window.validate_end),
        test_range=(window.test_start, window.test_end),
        label_horizon=label_horizon,
        embargo=embargo,
    )
    test = [
        i for i, ts in enumerate(timestamps) if window.test_start <= ts < window.test_end
    ]
    return train, validate, test
