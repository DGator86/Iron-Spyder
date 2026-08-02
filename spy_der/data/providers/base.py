"""Market-data provider interface and the synthetic implementation.

A provider's only job is to return a :class:`MarketSnapshot` as of a requested
timestamp. Everything downstream consumes snapshots, which is what keeps the
backtest and the live path on identical code: replaying history is just a
provider that answers for a past timestamp.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from spy_der.data.synthetic import ScenarioSpec, build_snapshot, scenario
from spy_der.data.validators.quality import DataQualityConfig, assess
from spy_der.domain.market import MarketSnapshot

#: Fields every normalized option record must carry; absence is schema drift.
REQUIRED_OPTION_FIELDS: frozenset[str] = frozenset(
    {
        "contract_symbol",
        "timestamp",
        "expiration",
        "strike",
        "right",
        "bid",
        "ask",
        "implied_volatility",
        "open_interest",
        "volume",
    }
)


class SchemaDriftError(RuntimeError):
    """Raised when an upstream payload no longer matches the expected schema."""


def validate_schema(record: dict[str, object]) -> None:
    """Fail loudly on missing fields rather than silently defaulting them.

    A vendor quietly dropping ``open_interest`` would otherwise be read as
    "open interest is zero everywhere", which turns every exposure calculation
    into a confident zero.
    """
    missing = REQUIRED_OPTION_FIELDS - record.keys()
    if missing:
        raise SchemaDriftError(f"payload is missing required fields: {sorted(missing)}")


class MarketDataProvider(ABC):
    """Source of point-in-time market snapshots."""

    @abstractmethod
    def snapshot(self, at: datetime) -> MarketSnapshot:
        """Return the snapshot as of ``at``, scored for data quality."""

    @abstractmethod
    def session_timestamps(self, day: datetime) -> Sequence[datetime]:
        """Decision timestamps available for ``day``."""

    def iter_session(self, day: datetime) -> Iterator[MarketSnapshot]:
        for ts in self.session_timestamps(day):
            yield self.snapshot(ts)


def score_snapshot(
    snapshot: MarketSnapshot, config: DataQualityConfig | None = None
) -> MarketSnapshot:
    """Attach ``DQ_t`` to a snapshot."""
    report = assess(snapshot, config)
    return replace(snapshot, data_quality_score=report.score)


@dataclass
class SyntheticProvider(MarketDataProvider):
    """Deterministic provider backed by :mod:`spy_der.data.synthetic`.

    Used for tests, demos, and the reference backtest. Because the underlying
    generator is seeded, two runs over the same timestamps produce byte-identical
    snapshots, which is what makes backtest results reproducible.
    """

    spec: ScenarioSpec
    step: timedelta = timedelta(minutes=5)
    session_start_hour: int = 13
    session_minutes: int = 390
    quality_config: DataQualityConfig | None = None

    @classmethod
    def from_scenario(cls, name: str, **kwargs: object) -> SyntheticProvider:
        return cls(spec=scenario(name), **kwargs)  # type: ignore[arg-type]

    def snapshot(self, at: datetime) -> MarketSnapshot:
        # Vary the seed with the timestamp so successive snapshots differ, while
        # staying reproducible for any given timestamp.
        spec = replace(
            self.spec,
            seed=self.spec.seed + int(at.timestamp()) % 100_000,
            # Advance the session clock. A static minutes_to_close would leave
            # the time stop and end-of-session exits permanently unreachable,
            # so no position would ever close on schedule.
            context=replace(self.spec.context, minutes_to_close=self._minutes_to_close(at)),
        )
        return score_snapshot(build_snapshot(spec, timestamp=at), self.quality_config)

    def _minutes_to_close(self, at: datetime) -> float:
        close = at.replace(hour=20, minute=0, second=0, microsecond=0)
        return max(0.0, (close - at).total_seconds() / 60.0)

    def session_timestamps(self, day: datetime) -> Sequence[datetime]:
        start = day.replace(
            hour=self.session_start_hour, minute=30, second=0, microsecond=0
        )
        count = int(self.session_minutes // (self.step.total_seconds() / 60))
        return [start + i * self.step for i in range(count)]


@dataclass
class ReplayProvider(MarketDataProvider):
    """Serves pre-recorded snapshots, refusing to look ahead.

    The point-in-time guarantee of spec 37.4 is enforced structurally here: a
    request for time ``t`` can only return a snapshot whose timestamp is
    ``<= t``. There is no code path that reaches a later record.
    """

    snapshots: tuple[MarketSnapshot, ...]

    def __post_init__(self) -> None:
        self.snapshots = tuple(sorted(self.snapshots, key=lambda s: s.timestamp))

    def snapshot(self, at: datetime) -> MarketSnapshot:
        candidate: MarketSnapshot | None = None
        for snap in self.snapshots:
            if snap.timestamp > at:
                break
            candidate = snap
        if candidate is None:
            raise LookupError(f"no snapshot available at or before {at.isoformat()}")
        return candidate

    def session_timestamps(self, day: datetime) -> Sequence[datetime]:
        target = day.date()
        return [s.timestamp for s in self.snapshots if s.timestamp.date() == target]
