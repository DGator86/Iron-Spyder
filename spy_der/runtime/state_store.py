"""Durable runtime state across restarts.

Without this the pipeline's entire memory — open positions, latched kill
switches, session P&L, the hidden-state belief — lives in the process. For an
attended tool that is merely inconvenient. For an unattended one it is a safety
hole, in three distinct ways:

1. **A restart would clear a latched kill switch.** Switches latch precisely so
   that a condition which trips once stays tripped until a human looks at it.
   If the daemon crashes and comes back clean, "restart it" becomes the way to
   bypass a lockout — the opposite of failing closed.
2. **A restart would forget the open book** while the broker still holds it.
   The risk engine would compute exposure from an empty portfolio and happily
   open a second position against limits that say one.
3. **A restart would reset daily loss and consecutive-loss counters**, which
   are exactly the accumulators the session kill switches are built on.

State is written as explicit JSON rather than pickled. Pickle would couple the
on-disk format to the class layout, so a refactor could silently fail to
restore an open position — and a corrupt or hostile file would execute code on
load.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spy_der.domain.enums import (
    ExitReason,
    KillSwitchReason,
    MarketState,
    OptionRight,
    StrategyFamily,
)
from spy_der.domain.execution import Position
from spy_der.domain.strategy import ExitPlan, OptionLeg, Strategy

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from spy_der.pipeline import DecisionPipeline

STATE_SCHEMA_VERSION = "1.0.0"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runtime_state (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    saved_at       TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    payload        TEXT NOT NULL
);
"""


class StateSchemaMismatch(RuntimeError):
    """Raised when a stored payload was written by an incompatible version."""


@dataclass(frozen=True)
class RestoreReport:
    """What a restore recovered, for the startup log and the audit trail."""

    restored: bool
    saved_at: datetime | None = None
    open_positions: int = 0
    kill_switches: tuple[str, ...] = ()
    session_pnl: float = 0.0
    consecutive_losses: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def has_latched_kill_switch(self) -> bool:
        return bool(self.kill_switches)

    def summary(self) -> str:
        if not self.restored:
            return "no prior state; starting fresh"
        parts = [
            f"restored from {self.saved_at:%Y-%m-%d %H:%M:%S}" if self.saved_at else "restored",
            f"{self.open_positions} open position(s)",
            f"session P&L {self.session_pnl:.2f}",
        ]
        if self.kill_switches:
            parts.append(f"LATCHED: {', '.join(self.kill_switches)}")
        return "; ".join(parts)


@dataclass
class RuntimeStateStore:
    """Single-row SQLite store for the pipeline's live state."""

    path: str | Path = "spyder_state.db"

    def __post_init__(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    # -- save -------------------------------------------------------------

    def save(self, pipeline: DecisionPipeline, *, now: datetime | None = None) -> None:
        """Write the pipeline's recoverable state.

        The write replaces a single row inside one transaction, so a crash
        mid-save leaves the previous good state intact rather than a truncated
        one.
        """
        payload = self.capture(pipeline)
        stamp = (now or datetime.now(tz=UTC)).isoformat()
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO runtime_state (id, saved_at, schema_version, payload) "
                "VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET saved_at=excluded.saved_at, "
                "schema_version=excluded.schema_version, payload=excluded.payload",
                (stamp, STATE_SCHEMA_VERSION, json.dumps(payload)),
            )
            conn.commit()

    @staticmethod
    def capture(pipeline: DecisionPipeline) -> dict[str, Any]:
        """Serializable snapshot of everything a restart must not lose."""
        risk = pipeline.risk_engine
        session = risk.session
        context = pipeline.features

        return {
            "positions": [_position_to_dict(p) for p in pipeline.portfolio.open_positions],
            "kill_switches": {r.value: detail for r, detail in risk.kill_switch.active.items()},
            "session": {
                "trading_day": session.trading_day.isoformat() if session.trading_day else None,
                "realized_pnl": session.realized_pnl,
                "peak_equity": session.peak_equity,
                "trades_today": session.trades_today,
                "consecutive_losses": session.consecutive_losses,
                "slippage_samples": [list(pair) for pair in session.slippage_samples],
            },
            "hidden_belief": [float(v) for v in pipeline.forecast_engine.hidden.belief],
            "previous_states": (
                {s.value: v for s, v in pipeline.forecast_engine._previous_states.items()}
                if pipeline.forecast_engine._previous_states
                else None
            ),
            "features": {
                "iv_history": list(context.iv_history),
                "rv_history": list(context.rv_history),
                "spot_history": list(context.spot_history),
            },
            "equity": pipeline.broker.equity,
        }

    # -- restore ----------------------------------------------------------

    def load(self) -> tuple[dict[str, Any], datetime] | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM runtime_state WHERE id = 1").fetchone()
        if row is None:
            return None
        if row["schema_version"] != STATE_SCHEMA_VERSION:
            raise StateSchemaMismatch(
                f"stored state is schema {row['schema_version']}, "
                f"this build expects {STATE_SCHEMA_VERSION}"
            )
        return json.loads(row["payload"]), datetime.fromisoformat(row["saved_at"])

    def restore(self, pipeline: DecisionPipeline) -> RestoreReport:
        """Rehydrate ``pipeline`` from stored state.

        Returns a report rather than raising on partial recovery: a position
        that cannot be rebuilt is a warning the supervisor must surface and act
        on, not a reason to refuse to start and leave the book unmanaged.
        """
        loaded = self.load()
        if loaded is None:
            return RestoreReport(restored=False)

        payload, saved_at = loaded
        warnings: list[str] = []

        # Kill switches first: if one is latched, everything that follows runs
        # under a lockout rather than briefly appearing clear.
        for value, detail in payload.get("kill_switches", {}).items():
            try:
                pipeline.risk_engine.kill_switch.trip(KillSwitchReason(value), detail)
            except ValueError:  # pragma: no cover - unknown reason from an older build
                warnings.append(f"unrecognized kill switch {value!r} retained as manual")
                pipeline.risk_engine.kill_switch.trip(KillSwitchReason.MANUAL, str(detail))

        session_data = payload.get("session", {})
        session = pipeline.risk_engine.session
        raw_day = session_data.get("trading_day")
        session.trading_day = date.fromisoformat(raw_day) if raw_day else None
        session.realized_pnl = float(session_data.get("realized_pnl", 0.0))
        session.peak_equity = float(session_data.get("peak_equity", 0.0))
        session.trades_today = int(session_data.get("trades_today", 0))
        session.consecutive_losses = int(session_data.get("consecutive_losses", 0))
        session.slippage_samples = [
            (float(a), float(b)) for a, b in session_data.get("slippage_samples", [])
        ]

        restored_positions = 0
        for record in payload.get("positions", []):
            try:
                position = _position_from_dict(record)
            except (KeyError, ValueError, TypeError) as exc:
                warnings.append(
                    f"could not rebuild position {record.get('position_id', '?')}: {exc}"
                )
                continue
            pipeline.portfolio.add(position)
            pipeline.broker.register_position(position)
            restored_positions += 1

        belief = payload.get("hidden_belief")
        if belief:
            import numpy as np

            vector = np.array(belief, dtype=np.float64)
            total = float(vector.sum())
            if total > 0.0:
                pipeline.forecast_engine.hidden.belief = vector / total
            else:  # pragma: no cover - defensive
                warnings.append("stored hidden-state belief was degenerate; reset to uniform")

        previous = payload.get("previous_states")
        if previous:
            pipeline.forecast_engine._previous_states = {
                MarketState(k): float(v) for k, v in previous.items()
            }

        features = payload.get("features", {})
        pipeline.features.iv_history = [float(v) for v in features.get("iv_history", [])]
        pipeline.features.rv_history = [float(v) for v in features.get("rv_history", [])]
        pipeline.features.spot_history = [float(v) for v in features.get("spot_history", [])]

        equity = payload.get("equity")
        if equity is not None:
            pipeline.broker.equity = float(equity)

        if warnings and restored_positions != len(payload.get("positions", [])):
            # A position we know existed but cannot rebuild means the in-memory
            # book no longer matches reality. Refuse to trade until a human
            # resolves it rather than operating on a partial view.
            pipeline.risk_engine.kill_switch.trip(
                KillSwitchReason.BROKER_STATE,
                "state restore could not rebuild every open position",
            )

        return RestoreReport(
            restored=True,
            saved_at=saved_at,
            open_positions=restored_positions,
            kill_switches=tuple(
                r.value for r in sorted(pipeline.risk_engine.kill_switch.active)
            ),
            session_pnl=session.realized_pnl,
            consecutive_losses=session.consecutive_losses,
            warnings=tuple(warnings),
        )

    def clear(self) -> None:
        """Drop stored state. Only for tests and deliberate operator resets."""
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM runtime_state WHERE id = 1")
            conn.commit()


# --------------------------------------------------------------------------
# Explicit serialization
# --------------------------------------------------------------------------


def _leg_to_dict(leg: OptionLeg) -> dict[str, Any]:
    return {
        "expiry": leg.expiry.isoformat(),
        "strike": leg.strike,
        "right": leg.right.value,
        "quantity": leg.quantity,
        "entry_price": leg.entry_price,
        "entry_iv": leg.entry_iv,
    }


def _leg_from_dict(data: dict[str, Any]) -> OptionLeg:
    return OptionLeg(
        expiry=datetime.fromisoformat(data["expiry"]),
        strike=float(data["strike"]),
        right=OptionRight(data["right"]),
        quantity=int(data["quantity"]),
        entry_price=float(data.get("entry_price", 0.0)),
        entry_iv=float(data.get("entry_iv", 0.0)),
    )


def _exit_plan_to_dict(plan: ExitPlan) -> dict[str, Any]:
    return {
        "profit_target_fraction": plan.profit_target_fraction,
        "stop_loss_fraction": plan.stop_loss_fraction,
        "max_holding_minutes": plan.max_holding_minutes,
        "invalidation_states": list(plan.invalidation_states),
        "close_before_expiry_minutes": plan.close_before_expiry_minutes,
        "min_forecast_probability": plan.min_forecast_probability,
    }


def _exit_plan_from_dict(data: dict[str, Any]) -> ExitPlan:
    return ExitPlan(
        profit_target_fraction=float(data["profit_target_fraction"]),
        stop_loss_fraction=float(data["stop_loss_fraction"]),
        max_holding_minutes=float(data["max_holding_minutes"]),
        invalidation_states=tuple(data.get("invalidation_states", ())),
        close_before_expiry_minutes=float(data.get("close_before_expiry_minutes", 30.0)),
        min_forecast_probability=float(data.get("min_forecast_probability", 0.0)),
    )


def _position_to_dict(position: Position) -> dict[str, Any]:
    return {
        "position_id": position.position_id,
        "opened_at": position.opened_at.isoformat(),
        "family": position.family.value,
        "legs": [_leg_to_dict(leg) for leg in position.legs],
        "contracts": position.contracts,
        "entry_price": position.entry_price,
        "max_loss_per_contract": position.max_loss_per_contract,
        "max_profit_per_contract": position.max_profit_per_contract,
        "exit_plan": _exit_plan_to_dict(position.exit_plan),
        "entry_commission": position.entry_commission,
        "current_price": position.current_price,
        "marked_at": position.marked_at.isoformat() if position.marked_at else None,
        "realized_pnl": position.realized_pnl,
        "exit_reason": position.exit_reason.value if position.exit_reason else None,
        "closed_at": position.closed_at.isoformat() if position.closed_at else None,
        "metadata": dict(position.metadata),
    }


def _position_from_dict(data: dict[str, Any]) -> Position:
    family = StrategyFamily(data["family"])
    legs = tuple(_leg_from_dict(leg) for leg in data["legs"])
    marked_at = data.get("marked_at")
    closed_at = data.get("closed_at")
    exit_reason = data.get("exit_reason")

    return Position(
        position_id=data["position_id"],
        opened_at=datetime.fromisoformat(data["opened_at"]),
        family=family,
        strategy=Strategy(family=family, legs=legs),
        contracts=int(data["contracts"]),
        entry_price=float(data["entry_price"]),
        max_loss_per_contract=float(data["max_loss_per_contract"]),
        max_profit_per_contract=float(data["max_profit_per_contract"]),
        exit_plan=_exit_plan_from_dict(data["exit_plan"]),
        entry_commission=float(data.get("entry_commission", 0.0)),
        current_price=float(data.get("current_price", 0.0)),
        marked_at=datetime.fromisoformat(marked_at) if marked_at else None,
        realized_pnl=float(data.get("realized_pnl", 0.0)),
        exit_reason=ExitReason(exit_reason) if exit_reason else None,
        closed_at=datetime.fromisoformat(closed_at) if closed_at else None,
        metadata={k: float(v) for k, v in data.get("metadata", {}).items()},
    )
