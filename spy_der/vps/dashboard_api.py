"""VPS status + desk API.

Binds loopback by default. The Vercel BFF reaches it through Caddy. Reads the
state root for live status and the trade journal; accepts authenticated
optimize enqueue/schedule writes (job files only — never mutates the live book).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spy_der.vps.heartbeat import write_heartbeat
from spy_der.vps.journal import build_journal
from spy_der.vps.optimize import (
    build_optimize_status,
    enqueue_run,
    load_schedule,
    save_schedule,
)
from spy_der.vps.paths import state_paths
from spy_der.vps.system_status import build_system_status

__all__ = ["create_app", "build_arg_parser", "main"]

log = logging.getLogger("spy_der.vps.dashboard_api")

SERVICE_NAME = "dashboard-api"


class OptimizeRunRequest(BaseModel):
    session_count: int | None = Field(default=None, ge=1, le=30)
    snapshot_limit: int | None = Field(default=None, ge=20, le=2000)


class OptimizeScheduleRequest(BaseModel):
    enabled: bool | None = None
    cadence: str | None = Field(default=None, pattern="^(daily|weekly)$")
    hour_utc: int | None = Field(default=None, ge=0, le=23)
    session_count: int | None = Field(default=None, ge=1, le=30)
    snapshot_limit: int | None = Field(default=None, ge=20, le=2000)


def create_app(state_root: str | Path) -> FastAPI:
    root = Path(state_root)
    app = FastAPI(
        title="Iron-Spyder VPS Desk API",
        version="0.3.0",
        description="Status, trade journal, and backtest-optimize controls",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        write_heartbeat(root, SERVICE_NAME, interval_seconds=60.0, detail="ok")
        return {"status": "ok", "state_root": str(root)}

    @app.get("/v1/system")
    def system() -> dict[str, Any]:
        write_heartbeat(root, SERVICE_NAME, interval_seconds=60.0, detail="system")
        return build_system_status(root)

    @app.get("/v1/live-state")
    def live_state() -> dict[str, Any]:
        path = state_paths(root).live_state
        if not path.is_file():
            raise HTTPException(404, "live_state.json not published yet")
        import json

        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(503, f"live_state unreadable: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(503, "live_state is not an object")
        write_heartbeat(root, SERVICE_NAME, interval_seconds=60.0, detail="live-state")
        return data

    @app.get("/journal")
    def journal(limit: int = 100) -> dict[str, Any]:
        write_heartbeat(root, SERVICE_NAME, interval_seconds=60.0, detail="journal")
        return build_journal(root, limit=min(max(limit, 1), 500))

    @app.get("/optimize")
    def optimize_status() -> dict[str, Any]:
        write_heartbeat(root, SERVICE_NAME, interval_seconds=60.0, detail="optimize")
        return build_optimize_status(root)

    @app.post("/optimize/run")
    def optimize_run(body: OptimizeRunRequest | None = None) -> dict[str, Any]:
        payload = body or OptimizeRunRequest()
        job = enqueue_run(
            root,
            reason="manual",
            session_count=payload.session_count,
            snapshot_limit=payload.snapshot_limit,
        )
        write_heartbeat(root, SERVICE_NAME, interval_seconds=60.0, detail="optimize-run")
        return {"queued": True, "job": job, "status": build_optimize_status(root)}

    @app.get("/optimize/schedule")
    def optimize_schedule_get() -> dict[str, Any]:
        return load_schedule(root)

    @app.post("/optimize/schedule")
    def optimize_schedule_set(body: OptimizeScheduleRequest) -> dict[str, Any]:
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        schedule = save_schedule(updates, root)
        write_heartbeat(root, SERVICE_NAME, interval_seconds=60.0, detail="optimize-schedule")
        return {"schedule": schedule, "status": build_optimize_status(root)}

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iron-Spyder VPS desk/status API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument(
        "--state-root",
        default=None,
        help="VPS state root (default: $IRON_SPYDER_STATE_ROOT or /var/lib/iron-spyder)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    root = args.state_root or state_paths().root
    app = create_app(root)
    import uvicorn

    log.info("dashboard API listening on %s:%s state_root=%s", args.host, args.port, root)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
