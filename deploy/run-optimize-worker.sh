#!/usr/bin/env bash
# run-optimize-worker.sh — oneshot wrapper used by iron-spyder-optimize.service.
set -euo pipefail

APP_DIR=${IRON_SPYDER_APP_DIR:-/opt/iron-spyder}
STATE_ROOT=${IRON_SPYDER_STATE_ROOT:-/var/lib/iron-spyder}
COMPOSE_FILE="$APP_DIR/docker-compose.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "optimize-worker: missing $COMPOSE_FILE" >&2
    exit 0
fi

cd "$APP_DIR"
# Reuse the built image; override entrypoint so we don't start uvicorn.
docker compose -f "$COMPOSE_FILE" run --rm --no-deps --entrypoint python \
    status-api -m scripts.optimize_worker --state-root "$STATE_ROOT"
