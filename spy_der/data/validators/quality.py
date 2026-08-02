"""Data-quality engine and trading lockout (spec section 6).

Produces ``DQ_t`` in ``[0, 1]``. Trading is disabled when ``DQ_t < DQ_min``.

The engine fails closed. A check that cannot be evaluated counts against the
score rather than being skipped, and any FATAL finding drives the score to zero
outright — no amount of otherwise-clean data offsets a crossed market or a
missing underlying price.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from spy_der.domain.market import MarketSnapshot, OptionQuote

#: Default lockout threshold; overridable per deployment mode.
DEFAULT_DQ_MIN = 0.70


class Severity(str, Enum):
    FATAL = "fatal"
    MAJOR = "major"
    MINOR = "minor"


#: Score penalty applied per affected fraction of the chain.
SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.FATAL: 1.0,
    Severity.MAJOR: 0.45,
    Severity.MINOR: 0.12,
}


@dataclass(frozen=True)
class DataQualityIssue:
    code: str
    severity: Severity
    message: str
    affected: int = 0
    total: int = 0

    @property
    def affected_fraction(self) -> float:
        if self.total <= 0:
            return 1.0 if self.affected else 0.0
        return self.affected / self.total


@dataclass(frozen=True)
class DataQualityReport:
    score: float
    issues: tuple[DataQualityIssue, ...]
    usable_contracts: int
    total_contracts: int
    threshold: float = DEFAULT_DQ_MIN

    @property
    def is_tradeable(self) -> bool:
        """``DQ_t >= DQ_min`` and no fatal finding."""
        return self.score >= self.threshold and not self.has_fatal

    @property
    def has_fatal(self) -> bool:
        return any(i.severity is Severity.FATAL for i in self.issues)

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(i.code for i in self.issues)

    def issues_of(self, severity: Severity) -> tuple[DataQualityIssue, ...]:
        return tuple(i for i in self.issues if i.severity is severity)

    def summary(self) -> str:
        if not self.issues:
            return f"DQ={self.score:.3f} clean"
        parts = [f"{i.code}({i.affected})" for i in self.issues]
        return f"DQ={self.score:.3f} " + " ".join(parts)


@dataclass
class DataQualityConfig:
    threshold: float = DEFAULT_DQ_MIN
    max_quote_age: timedelta = timedelta(seconds=90)
    max_underlying_skew: timedelta = timedelta(seconds=30)
    max_spread_ratio: float = 0.35
    min_iv: float = 0.005
    max_iv: float = 5.0
    min_contracts: int = 20
    require_same_day_expiry: bool = False
    max_stale_fraction: float = 0.25


def assess(snapshot: MarketSnapshot, config: DataQualityConfig | None = None) -> DataQualityReport:
    """Run every data-quality check and compute ``DQ_t``."""
    cfg = config or DataQualityConfig()
    chain = snapshot.option_chain
    total = len(chain)
    issues: list[DataQualityIssue] = []

    if total == 0:
        return DataQualityReport(
            score=0.0,
            issues=(
                DataQualityIssue("empty_chain", Severity.FATAL, "option chain is empty"),
            ),
            usable_contracts=0,
            total_contracts=0,
            threshold=cfg.threshold,
        )

    issues.extend(_check_underlying(snapshot, cfg))
    issues.extend(_check_quotes(snapshot, cfg))
    issues.extend(_check_structure(snapshot, cfg))
    issues.extend(_check_timeliness(snapshot, cfg))

    usable = sum(1 for q in chain if _is_usable(q, snapshot, cfg))
    score = _score(issues, usable=usable, total=total)

    return DataQualityReport(
        score=score,
        issues=tuple(issues),
        usable_contracts=usable,
        total_contracts=total,
        threshold=cfg.threshold,
    )


def _check_underlying(
    snapshot: MarketSnapshot, cfg: DataQualityConfig
) -> list[DataQualityIssue]:
    out: list[DataQualityIssue] = []
    quote = snapshot.spy_quote

    if quote.last <= 0.0 and quote.mid <= 0.0:
        out.append(
            DataQualityIssue(
                "missing_underlying", Severity.FATAL, "SPY price is missing", affected=1
            )
        )
    if quote.bid > quote.ask > 0.0:
        out.append(
            DataQualityIssue(
                "underlying_crossed", Severity.FATAL, "SPY quote is crossed", affected=1
            )
        )
    if not snapshot.spy_bars:
        out.append(
            DataQualityIssue(
                "missing_bars", Severity.MAJOR, "no SPY price bars available", affected=1
            )
        )
    return out


def _check_quotes(snapshot: MarketSnapshot, cfg: DataQualityConfig) -> list[DataQualityIssue]:
    chain = snapshot.option_chain
    total = len(chain)
    counts: Counter[str] = Counter()

    for q in chain:
        if q.bid < 0.0:
            counts["negative_bid"] += 1
        if q.ask < q.bid:
            counts["crossed_market"] += 1
        elif q.ask == q.bid and q.bid > 0.0:
            counts["locked_market"] += 1
        if q.mid <= 0.0:
            counts["zero_mid"] += 1
        if not (cfg.min_iv <= q.implied_volatility <= cfg.max_iv):
            counts["invalid_iv"] += 1
        if q.mid > 0.0 and q.spread_ratio > cfg.max_spread_ratio:
            counts["wide_spread"] += 1
        if q.open_interest < 0 or q.volume < 0:
            counts["negative_size"] += 1
        if q.expiration < q.timestamp:
            counts["expired_contract"] += 1
        if not _within_no_arbitrage(q, snapshot):
            counts["implausible_price"] += 1

    severities = {
        "negative_bid": Severity.FATAL,
        "crossed_market": Severity.FATAL,
        "zero_mid": Severity.MAJOR,
        "invalid_iv": Severity.MAJOR,
        "expired_contract": Severity.FATAL,
        "implausible_price": Severity.MAJOR,
        "negative_size": Severity.MAJOR,
        "locked_market": Severity.MINOR,
        "wide_spread": Severity.MINOR,
    }
    messages = {
        "negative_bid": "contracts quoted with a negative bid",
        "crossed_market": "contracts with ask below bid",
        "zero_mid": "contracts with a non-positive midpoint",
        "invalid_iv": "contracts with implied volatility outside plausible bounds",
        "expired_contract": "expired contracts present in a live chain",
        "implausible_price": "contracts priced outside no-arbitrage bounds",
        "negative_size": "contracts with negative volume or open interest",
        "locked_market": "locked markets requiring special handling",
        "wide_spread": "contracts with an excessive bid/ask spread ratio",
    }

    return [
        DataQualityIssue(code, severities[code], messages[code], affected=count, total=total)
        for code, count in counts.items()
        if count
    ]


def _check_structure(snapshot: MarketSnapshot, cfg: DataQualityConfig) -> list[DataQualityIssue]:
    out: list[DataQualityIssue] = []
    chain = snapshot.option_chain
    total = len(chain)

    keys = Counter((q.expiration, q.strike, q.right) for q in chain)
    duplicates = sum(count - 1 for count in keys.values() if count > 1)
    if duplicates:
        out.append(
            DataQualityIssue(
                "duplicate_contracts",
                Severity.MAJOR,
                "duplicate contracts in the chain",
                affected=duplicates,
                total=total,
            )
        )

    if total < cfg.min_contracts:
        out.append(
            DataQualityIssue(
                "sparse_chain",
                Severity.MAJOR,
                f"only {total} contracts, need at least {cfg.min_contracts}",
                affected=total,
                total=total,
            )
        )

    # Every expiration must quote both rights, or the surface is unusable.
    for expiration in snapshot.expirations:
        section = snapshot.chain_for(expiration)
        rights = {q.right for q in section}
        if len(rights) < 2:
            out.append(
                DataQualityIssue(
                    "missing_chain_section",
                    Severity.MAJOR,
                    f"expiration {expiration:%Y-%m-%d} is missing calls or puts",
                    affected=len(section),
                    total=total,
                )
            )

    if cfg.require_same_day_expiry:
        today = snapshot.timestamp.date()
        if today not in snapshot.expiration_dates:
            out.append(
                DataQualityIssue(
                    "missing_same_day_expiry",
                    Severity.MAJOR,
                    "same-day expiration is absent from the chain",
                    affected=1,
                )
            )

    return out


def _check_timeliness(snapshot: MarketSnapshot, cfg: DataQualityConfig) -> list[DataQualityIssue]:
    out: list[DataQualityIssue] = []
    chain = snapshot.option_chain
    total = len(chain)

    stale = sum(1 for q in chain if snapshot.timestamp - q.timestamp > cfg.max_quote_age)
    if stale:
        fraction = stale / total
        severity = Severity.FATAL if fraction > cfg.max_stale_fraction else Severity.MAJOR
        out.append(
            DataQualityIssue(
                "stale_quotes",
                severity,
                f"{stale} contracts older than {cfg.max_quote_age.total_seconds():.0f}s",
                affected=stale,
                total=total,
            )
        )

    skew = abs((snapshot.spy_quote.timestamp - snapshot.timestamp).total_seconds())
    if skew > cfg.max_underlying_skew.total_seconds():
        out.append(
            DataQualityIssue(
                "timestamp_mismatch",
                Severity.MAJOR,
                f"underlying quote is {skew:.0f}s away from the snapshot timestamp",
                affected=1,
            )
        )
    return out


def _within_no_arbitrage(quote: OptionQuote, snapshot: MarketSnapshot) -> bool:
    """European no-arbitrage bounds, discounted, with a quote-granularity tolerance.

    The bounds must be discounted. An undiscounted ``K - S`` floor for puts
    flags every deep in-the-money European put as implausible, since such a put
    is legitimately worth less than its intrinsic value before expiry.
    """
    if quote.mid <= 0.0:
        return True  # already reported by the zero_mid check

    import math

    tau = quote.tau
    forward_spot = snapshot.spot * math.exp(-snapshot.dividend_yield * tau)
    discounted_strike = quote.strike * math.exp(-snapshot.risk_free_rate * tau)
    tolerance = max(0.05, quote.spread)

    if quote.right.value == "call":
        lower = max(0.0, forward_spot - discounted_strike)
        upper = forward_spot
    else:
        lower = max(0.0, discounted_strike - forward_spot)
        upper = discounted_strike
    return (lower - tolerance) <= quote.mid <= (upper + tolerance)


def _is_usable(quote: OptionQuote, snapshot: MarketSnapshot, cfg: DataQualityConfig) -> bool:
    """Whether a single contract is fit to be traded or modelled."""
    return (
        quote.bid >= 0.0
        and quote.ask >= quote.bid
        and quote.mid > 0.0
        and cfg.min_iv <= quote.implied_volatility <= cfg.max_iv
        and quote.expiration >= quote.timestamp
        and quote.spread_ratio <= cfg.max_spread_ratio
        and snapshot.timestamp - quote.timestamp <= cfg.max_quote_age
    )


def _score(issues: list[DataQualityIssue], *, usable: int, total: int) -> float:
    """Combine issue penalties with the usable-contract ratio."""
    if any(i.severity is Severity.FATAL for i in issues):
        return 0.0

    penalty = 0.0
    for issue in issues:
        penalty += SEVERITY_WEIGHT[issue.severity] * max(issue.affected_fraction, 0.05)

    usable_ratio = usable / total if total else 0.0
    return float(max(0.0, min(1.0, (1.0 - penalty) * (0.4 + 0.6 * usable_ratio))))


def filter_usable(
    snapshot: MarketSnapshot, config: DataQualityConfig | None = None
) -> tuple[OptionQuote, ...]:
    """Return only the contracts fit for analytics and strategy construction."""
    cfg = config or DataQualityConfig()
    return tuple(q for q in snapshot.option_chain if _is_usable(q, snapshot, cfg))
