"""Configuration and deployment modes (spec sections 33, 40).

Modes are ordered by how much authority they grant, and each is strictly more
permissive than the one before. ``LIVE`` is defined but cannot execute: no live
broker adapter ships, and :func:`assert_live_allowed` refuses unless the spec 40
gates have been explicitly recorded as met.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

# types-PyYAML is a dev dependency, but on Debian-style layouts PyYAML
# resolves from /usr/lib while its stubs land in /usr/local/lib and mypy
# will not pair them. warn_unused_ignores is off, so this stays harmless
# in a clean install where the stubs are found.
import yaml  # type: ignore[import-untyped]

from spy_der.data.validators.quality import DataQualityConfig
from spy_der.execution.fills import CostModel
from spy_der.models.ensemble.combiner import EnsembleConfig
from spy_der.models.forecast_engine import ForecastEngineConfig
from spy_der.optimizer.engine import OptimizerConfig
from spy_der.risk.limits import RiskConfig

CONFIG_DIR = Path(__file__).resolve().parent


class Mode(str, Enum):
    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"

    @property
    def may_submit_orders(self) -> bool:
        """Shadow mode generates orders but never submits them (spec 40 phase 3)."""
        return self in {Mode.PAPER, Mode.LIVE}

    @property
    def is_live(self) -> bool:
        return self is Mode.LIVE


class LiveTradingDisabled(RuntimeError):
    """Raised when live execution is requested without the spec 40 gates met."""


@dataclass(frozen=True)
class LiveGates:
    """Spec 40 phase-2 through phase-4 preconditions for live trading."""

    paper_months: float = 0.0
    completed_paper_trades: int = 0
    shadow_completed: bool = False
    reconciliation_tested: bool = False
    unresolved_risk_defects: int = 1
    operator_authorized: bool = False

    #: Spec 40 phase 2 minimums.
    REQUIRED_MONTHS: float = 6.0
    REQUIRED_TRADES: int = 250

    def unmet(self) -> tuple[str, ...]:
        problems: list[str] = []
        if self.paper_months < self.REQUIRED_MONTHS:
            problems.append(
                f"paper trading {self.paper_months:.1f} months, need {self.REQUIRED_MONTHS:.0f}"
            )
        if self.completed_paper_trades < self.REQUIRED_TRADES:
            problems.append(
                f"{self.completed_paper_trades} paper trades, need {self.REQUIRED_TRADES}"
            )
        if not self.shadow_completed:
            problems.append("shadow execution has not been completed")
        if not self.reconciliation_tested:
            problems.append("broker reconciliation has not been tested")
        if self.unresolved_risk_defects:
            problems.append(f"{self.unresolved_risk_defects} unresolved risk defects")
        if not self.operator_authorized:
            problems.append("no explicit operator authorization")
        return tuple(problems)

    @property
    def satisfied(self) -> bool:
        return not self.unmet()


@dataclass
class Settings:
    """Complete runtime configuration."""

    mode: Mode = Mode.PAPER
    starting_equity: float = 100_000.0
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    costs: CostModel = field(default_factory=CostModel)
    risk: RiskConfig = field(default_factory=RiskConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    forecast: ForecastEngineConfig = field(default_factory=ForecastEngineConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    live_gates: LiveGates = field(default_factory=LiveGates)
    audit_path: str = "spyder_audit.db"

    @property
    def may_submit_orders(self) -> bool:
        return self.mode.may_submit_orders

    def assert_live_allowed(self) -> None:
        """Refuse live execution unless every spec 40 gate is met."""
        if not self.mode.is_live:
            return
        unmet = self.live_gates.unmet()
        if unmet:
            raise LiveTradingDisabled(
                "live trading is not permitted: " + "; ".join(unmet)
            )
        raise LiveTradingDisabled(
            "no live broker adapter is available; implement Broker against a real "
            "venue and verify reconciliation before enabling live mode"
        )


def load_settings(mode: Mode | str = Mode.PAPER, *, path: Path | None = None) -> Settings:
    """Load settings for a deployment mode from YAML, falling back to defaults."""
    mode = Mode(mode)
    config_path = path or CONFIG_DIR / f"{mode.value}.yaml"
    overrides: dict[str, Any] = {}
    if config_path.exists():
        overrides = yaml.safe_load(config_path.read_text()) or {}

    settings = Settings(mode=mode)
    return _apply(settings, overrides)


def _apply(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """Apply a YAML override map onto nested dataclasses.

    Nested sections (``risk.trade.min_confidence``, ``optimizer.no_trade.margin``)
    are merged recursively so mode files can tune leaf fields without replacing
    whole sub-objects with plain dicts.
    """
    scalar = {
        key: value
        for key, value in overrides.items()
        if key in {"starting_equity", "audit_path"}
    }
    settings = replace(settings, **scalar) if scalar else settings

    for section, target in (
        ("risk", "risk"),
        ("optimizer", "optimizer"),
        ("costs", "costs"),
        ("forecast", "forecast"),
        ("data_quality", "data_quality"),
    ):
        values = overrides.get(section)
        if not isinstance(values, dict):
            continue
        current = getattr(settings, target)
        setattr(settings, target, _merge_dataclass(current, values, path=section))

    return settings


def _merge_dataclass(current: Any, values: dict[str, Any], *, path: str) -> Any:
    """Recursively replace dataclass fields from a YAML mapping."""
    known: dict[str, Any] = {}
    unknown: set[str] = set()
    for key, value in values.items():
        if not hasattr(current, key):
            unknown.add(key)
            continue
        existing = getattr(current, key)
        if isinstance(value, dict) and hasattr(existing, "__dataclass_fields__"):
            known[key] = _merge_dataclass(existing, value, path=f"{path}.{key}")
        else:
            known[key] = value
    if unknown:
        raise ValueError(f"unknown {path} settings: {sorted(unknown)}")
    return replace(current, **known) if known else current
