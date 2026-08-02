"""Position reconciliation (spec sections 31, 40 phase 4, 44).

Compares what the system believes it holds against what the broker reports.
Any mismatch is a hard stop: an unreconciled book means the risk engine is
computing exposure from fiction, and every downstream limit becomes advisory.

Reconciliation compares *leg-level* positions rather than strategy identifiers.
Brokers report legs, not strategies, and a structure that has been partially
closed or assigned shows up as a leg mismatch long before any strategy-level
accounting notices.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from spy_der.domain.enums import OptionRight
from spy_der.domain.execution import BrokerState, Position

LegKey = tuple[datetime, float, OptionRight]


@dataclass(frozen=True)
class ReconciliationReport:
    matched_legs: int
    discrepancies: tuple[str, ...]
    internal_only: tuple[str, ...]
    broker_only: tuple[str, ...]
    quantity_mismatches: tuple[str, ...]

    @property
    def is_reconciled(self) -> bool:
        return not self.discrepancies

    def summary(self) -> str:
        if self.is_reconciled:
            return f"reconciled: {self.matched_legs} legs match"
        return f"UNRECONCILED: {len(self.discrepancies)} discrepancies"


def leg_quantities(positions: tuple[Position, ...]) -> Counter[LegKey]:
    """Net signed contract quantity per (expiry, strike, right).

    Zero entries are dropped, negatives are kept. ``Counter.__pos__`` would
    discard negatives too, which would silently exclude every short leg from
    reconciliation — precisely the legs that carry assignment risk.
    """
    totals: Counter[LegKey] = Counter()
    for position in positions:
        if not position.is_open:
            continue
        for leg in position.legs:
            totals[(leg.expiry, leg.strike, leg.right)] += leg.quantity * position.contracts
    return Counter({key: value for key, value in totals.items() if value != 0})


def reconcile(
    internal: tuple[Position, ...], broker_positions: tuple[Position, ...]
) -> ReconciliationReport:
    """Compare internal and broker books leg by leg."""
    ours = leg_quantities(internal)
    theirs = leg_quantities(broker_positions)

    internal_only: list[str] = []
    broker_only: list[str] = []
    mismatches: list[str] = []

    for key in sorted(set(ours) | set(theirs), key=lambda k: (k[0], k[1], k[2].value)):
        mine, yours = ours.get(key, 0), theirs.get(key, 0)
        if mine == yours:
            continue
        label = f"{key[0]:%Y-%m-%d} {key[1]:g}{key[2].value[0].upper()}"
        if yours == 0:
            internal_only.append(f"{label}: internal {mine:+d}, broker flat")
        elif mine == 0:
            broker_only.append(f"{label}: broker {yours:+d}, internal flat")
        else:
            mismatches.append(f"{label}: internal {mine:+d} vs broker {yours:+d}")

    discrepancies = tuple(internal_only + broker_only + mismatches)
    matched = sum(1 for key in ours if ours[key] == theirs.get(key, 0))

    return ReconciliationReport(
        matched_legs=matched,
        discrepancies=discrepancies,
        internal_only=tuple(internal_only),
        broker_only=tuple(broker_only),
        quantity_mismatches=tuple(mismatches),
    )


def apply_to_broker_state(state: BrokerState, report: ReconciliationReport) -> BrokerState:
    """Fold a reconciliation result into the broker state the risk engine reads."""
    from dataclasses import replace

    return replace(
        state, reconciled=report.is_reconciled, discrepancies=report.discrepancies
    )
