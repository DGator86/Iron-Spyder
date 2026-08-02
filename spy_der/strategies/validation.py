"""Defined-risk validation (spec sections 2.2, 2.3, 22, 44).

The governing constraint is that ``MaxLoss(S) < inf`` must be provable *before*
entry. This module proves it structurally rather than by probing the payoff at
some large spot and hoping the number that comes back looks finite.

The proof rests on one invariant, the **expiry suffix-coverage rule**:

    For each option right, and for every expiration ``T`` present in the
    structure, the sum of signed quantities over all legs of that right with
    expiry ``>= T`` must be non-negative.

Why that is the right rule:

* *Bounded loss.* As ``S -> inf`` every call contributes ``+1`` to the payoff
  slope per unit of signed quantity — whether it is valued at intrinsic or, for
  an unexpired short, at its upper bound ``S``. The upside slope therefore
  equals the suffix sum at the earliest expiry, so requiring it to be
  non-negative is exactly the condition for the loss to be bounded above.
  On the downside a put can never pay more than its strike, so equity payoffs
  are automatically finite as ``S -> 0``.

* *No uncovered short at any point in time.* Checking suffix sums rather than a
  single net total is what distinguishes a long calendar from a short one. Both
  net to zero overall, but the short calendar leaves a naked far-dated short
  after the near leg expires, and only the suffix at the far expiry reveals it.

* *No ratio write.* A ``-2 x +1`` structure nets to ``-1`` and is rejected,
  while a genuine ``-1 x +2`` backspread nets to ``+1`` and passes.

The rule is checked independently of the payoff arithmetic, and
:func:`validate_defined_risk` then cross-checks it against the computed
:class:`~spy_der.strategies.payoff.PayoffProfile`. Two independent derivations
must agree before a structure is admitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from spy_der.domain.enums import OptionRight, StrategyFamily
from spy_der.domain.strategy import Strategy
from spy_der.strategies.payoff import PayoffProfile, payoff_profile

#: Hard ceiling on legs per structure; more than this is not executable as one
#: atomic multi-leg order at acceptable cost (spec 28.1 execution complexity).
MAX_LEGS = 6

#: A structure claiming a non-positive worst case is a data or modelling error,
#: never a free lunch, so it is rejected rather than ranked first.
MIN_MAX_LOSS = 1e-6


class UndefinedRiskError(ValueError):
    """Raised when a structure fails the defined-risk proof."""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    violations: tuple[str, ...]
    profile: PayoffProfile | None

    @property
    def is_valid(self) -> bool:
        return not self.violations


def coverage_violations(strategy: Strategy) -> tuple[str, ...]:
    """Check the expiry suffix-coverage rule described in the module docstring."""
    problems: list[str] = []
    for right in (OptionRight.CALL, OptionRight.PUT):
        legs = [leg for leg in strategy.legs if leg.right is right]
        if not legs:
            continue
        for expiry in sorted({leg.expiry for leg in legs}):
            suffix = sum(leg.quantity for leg in legs if leg.expiry >= expiry)
            if suffix < 0:
                problems.append(
                    f"uncovered short {right.value} exposure of {suffix} contracts "
                    f"from {expiry.isoformat()} onward"
                )
    return tuple(problems)


def structural_violations(strategy: Strategy) -> tuple[str, ...]:
    """Checks that do not require pricing: shape, size, and family membership."""
    problems: list[str] = []

    if not isinstance(strategy.family, StrategyFamily):
        problems.append(f"family {strategy.family!r} is not in the allowed universe")

    if strategy.leg_count > MAX_LEGS:
        problems.append(f"structure has {strategy.leg_count} legs, limit is {MAX_LEGS}")

    if strategy.is_multi_expiry and not strategy.family.is_calendarized:
        problems.append(
            f"family {strategy.family.value} spans {len(strategy.expiries)} expirations "
            "but is not a calendarized family"
        )

    duplicates = _duplicate_leg_keys(strategy)
    if duplicates:
        problems.append(
            "structure contains duplicate legs that should be netted: " + ", ".join(duplicates)
        )

    return tuple(problems)


def _duplicate_leg_keys(strategy: Strategy) -> tuple[str, ...]:
    seen: dict[tuple[object, float, OptionRight], int] = {}
    for leg in strategy.legs:
        key = (leg.expiry, leg.strike, leg.right)
        seen[key] = seen.get(key, 0) + 1
    return tuple(
        f"{right.value}@{strike}" for (_, strike, right), count in seen.items() if count > 1
    )


def validate_defined_risk(strategy: Strategy) -> ValidationResult:
    """Full pre-entry validation. Empty violations means the structure is admissible."""
    problems = list(structural_violations(strategy))
    problems.extend(coverage_violations(strategy))

    profile: PayoffProfile | None = None
    try:
        profile = payoff_profile(strategy)
    except (ValueError, OverflowError) as exc:
        problems.append(f"payoff profile could not be computed: {exc}")
        return ValidationResult(violations=tuple(problems), profile=None)

    # Independent cross-check: the payoff arithmetic must agree with the
    # structural rule. Disagreement means one of the two is wrong, and an
    # unresolved disagreement is itself a reason to refuse the trade.
    if not math.isfinite(profile.max_loss):
        problems.append("maximum loss is not finite")
    if profile.upside_slope < 0.0:
        problems.append(
            f"unbounded upside loss: terminal slope {profile.upside_slope:.2f} per point"
        )
    if not problems and profile.max_loss < MIN_MAX_LOSS:
        problems.append(
            f"computed maximum loss {profile.max_loss:.6f} is non-positive, "
            "which indicates a pricing or construction error"
        )

    return ValidationResult(violations=tuple(problems), profile=profile)


def assert_defined_risk(strategy: Strategy) -> PayoffProfile:
    """Return the payoff profile, or raise :class:`UndefinedRiskError`.

    This is the chokepoint used by the registry and by the execution path, so
    that an undefined-risk structure cannot reach an order under any code path.
    """
    result = validate_defined_risk(strategy)
    if not result.is_valid or result.profile is None:
        raise UndefinedRiskError(
            f"{strategy.family.value} rejected: " + "; ".join(result.violations)
        )
    return result.profile


def is_defined_risk(strategy: Strategy) -> bool:
    return validate_defined_risk(strategy).is_valid
