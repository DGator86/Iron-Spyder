"""Read-only VPS status API.

Binds loopback by default. The Vercel frontend (or a tunnel) reaches it through
a reverse proxy, so the API is never directly internet-facing. It only reads
the state root — heartbeats, live_state, deploy.json — and never mutates the
decision pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from spy_der.vps.heartbeat import write_heartbeat
from spy_der.vps.paths import state_paths
from spy_der.vps.system_status import build_system_status

__all__ = ["create_app", "build_arg_parser", "main"]

log = logging.getLogger("spy_der.vps.dashboard_api")

SERVICE_NAME = "dashboard-api"


def create_app(state_root: str | Path) -> FastAPI:
    root = Path(state_root)
    app = FastAPI(
        title="Iron-Spyder VPS Status",
        version="0.2.0",
        description="Read-only status surface for the VPS deployment",
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

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iron-Spyder VPS read-only dashboard API")
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
