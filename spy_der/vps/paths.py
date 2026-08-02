"""VPS state-root layout.

Everything an operator or dashboard needs is under one tree. Services write;
the status API only reads. Paths are overridable via ``IRON_SPYDER_STATE_ROOT``
so a test or a parallel install never has to patch the module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_STATE_ROOT",
    "STATE_DIRS",
    "StatePaths",
    "ensure_state_tree",
    "state_paths",
]

DEFAULT_STATE_ROOT = os.environ.get("IRON_SPYDER_STATE_ROOT", "/var/lib/iron-spyder")

#: Subdirectories the VPS runtime creates and owns under the state root.
STATE_DIRS: tuple[str, ...] = (
    "health",
    "decisions",
    "positions",
    "audit",
    "reports",
    "configs",
)


@dataclass(frozen=True, slots=True)
class StatePaths:
    root: Path

    @property
    def health(self) -> Path:
        return self.root / "health"

    @property
    def live_state(self) -> Path:
        return self.root / "live_state.json"

    @property
    def deploy_manifest(self) -> Path:
        return self.root / "deploy.json"

    @property
    def runtime_db(self) -> Path:
        return self.root / "runtime.db"

    @property
    def audit_db(self) -> Path:
        return self.root / "audit" / "audit.db"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions"

    @property
    def positions(self) -> Path:
        return self.root / "positions"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def configs(self) -> Path:
        return self.root / "configs"


def state_paths(root: str | Path | None = None) -> StatePaths:
    return StatePaths(Path(root if root is not None else DEFAULT_STATE_ROOT))


def ensure_state_tree(root: str | Path | None = None) -> StatePaths:
    """Create every declared subdirectory. Idempotent."""
    paths = state_paths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    for name in STATE_DIRS:
        (paths.root / name).mkdir(parents=True, exist_ok=True)
    return paths
