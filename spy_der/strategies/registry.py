"""Strategy registry — the single admission gate (spec section 44).

"No undefined-risk strategy can enter the registry" is an acceptance criterion,
so admission is a chokepoint rather than a convention. Every structure that
reaches the optimizer has passed :func:`assert_defined_risk`, and there is no
method on this class that adds a structure without doing so.

Rejections are retained rather than discarded: spec 32 requires rejected
candidates and their reasons in the audit record, and the dashboard shows them
so an empty candidate list is explainable instead of mysterious.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from spy_der.domain.enums import StrategyFamily
from spy_der.domain.strategy import Strategy
from spy_der.strategies.payoff import PayoffProfile
from spy_der.strategies.validation import UndefinedRiskError, validate_defined_risk


@dataclass(frozen=True)
class AdmittedStrategy:
    """A structure that has passed the defined-risk proof."""

    strategy: Strategy
    profile: PayoffProfile

    @property
    def family(self) -> StrategyFamily:
        return self.strategy.family


@dataclass(frozen=True)
class RejectedStrategy:
    strategy: Strategy
    reasons: tuple[str, ...]

    @property
    def family(self) -> StrategyFamily:
        return self.strategy.family


@dataclass
class StrategyRegistry:
    """Collects admitted structures and the reasons others were refused."""

    admitted: list[AdmittedStrategy] = field(default_factory=list)
    rejected: list[RejectedStrategy] = field(default_factory=list)

    def submit(self, strategy: Strategy) -> AdmittedStrategy | None:
        """Validate and admit ``strategy``, or record why it was refused."""
        result = validate_defined_risk(strategy)
        if not result.is_valid or result.profile is None:
            self.rejected.append(RejectedStrategy(strategy, result.violations))
            return None
        entry = AdmittedStrategy(strategy=strategy, profile=result.profile)
        self.admitted.append(entry)
        return entry

    def submit_all(self, strategies: Iterable[Strategy]) -> list[AdmittedStrategy]:
        return [entry for s in strategies if (entry := self.submit(s)) is not None]

    def require(self, strategy: Strategy) -> AdmittedStrategy:
        """Admit ``strategy`` or raise; used where a refusal is a programming error."""
        result = validate_defined_risk(strategy)
        if not result.is_valid or result.profile is None:
            raise UndefinedRiskError(
                f"{strategy.family.value} rejected: " + "; ".join(result.violations)
            )
        entry = AdmittedStrategy(strategy=strategy, profile=result.profile)
        self.admitted.append(entry)
        return entry

    @property
    def admitted_count(self) -> int:
        return len(self.admitted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def rejection_summary(self) -> dict[str, int]:
        """Count of rejections by reason, for the audit record and dashboard."""
        counts: dict[str, int] = {}
        for entry in self.rejected:
            for reason in entry.reasons:
                key = reason.split(":")[0].split(" of ")[0]
                counts[key] = counts.get(key, 0) + 1
        return counts

    def by_family(self) -> dict[StrategyFamily, int]:
        counts: dict[StrategyFamily, int] = {}
        for entry in self.admitted:
            counts[entry.family] = counts.get(entry.family, 0) + 1
        return counts

    def clear(self) -> None:
        self.admitted.clear()
        self.rejected.clear()
