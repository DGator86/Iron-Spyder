#!/usr/bin/env bash
# remote-deploy.sh — idempotent deploy of Iron-Spyder on its dedicated CPU VPS.
#
# Runs ON the Iron-Spyder host as root. This host is CPU-only and separate from
# any SPY-DER / 0DTE / GPU machine. See deploy/CPU_VPS.md.
#
# Default path: Docker Compose live stack + pull-based self-update timer.
# The secrets file (/etc/iron-spyder/iron-spyder.env) is never overwritten.
set -euo pipefail

REPO_URL="${IRON_SPYDER_REPO_URL:-https://github.com/DGator86/Iron-Spyder.git}"
DEPLOY_REF="${DEPLOY_REF:-origin/main}"
APP_DIR="${IRON_SPYDER_APP_DIR:-/opt/iron-spyder}"
ENV_FILE=/etc/iron-spyder/iron-spyder.env
CONFIG_FILE=/etc/iron-spyder/config.yaml
STATE_DIR=/var/lib/iron-spyder
COMPOSE_ENV="$APP_DIR/.env"
SVC_USER=iron-spyder
# Must match the Dockerfile's `useradd --uid 10001`. A bind mount carries numeric
# uids, not names, so a host user with the same name but a different uid leaves the
# containers unable to write the state tree — and the supervisor logs that failure
# per cycle while continuing to run, so the dashboard silently serves stale state.
SVC_UID=10001

#: Compose-backed live unit + pull-based update timer.
SERVICES=(
    iron-spyder
)

TIMERS=(
    iron-spyder-update
)

STATE_DIRS=(
    health decisions positions audit reports configs
)

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo "remote-deploy.sh must run as root (got uid $(id -u))." >&2
    exit 1
fi

log "System packages (CPU host — no NVIDIA/CUDA packages)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl git openssl >/dev/null

# Docker Engine from Ubuntu's docker.io package is enough for Compose V2.
if ! command -v docker >/dev/null 2>&1; then
    apt-get install -y -qq docker.io docker-compose-v2 >/dev/null
    systemctl enable --now docker
fi
if ! docker compose version >/dev/null 2>&1; then
    apt-get install -y -qq docker-compose-v2 >/dev/null
fi

log "Service user + directories"
id -u "$SVC_USER" >/dev/null 2>&1 || \
    useradd --system --uid "$SVC_UID" --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
# A pre-existing account from before the uid was pinned still owns the state tree with
# the wrong number. Report it rather than leaving the containers silently unable to write.
have_uid=$(id -u "$SVC_USER")
if [ "$have_uid" != "$SVC_UID" ]; then
    log "NOTE: host user $SVC_USER is uid $have_uid, containers run as $SVC_UID;"
    log "      chowning the state tree numerically so the bind mount works."
fi
mkdir -p "$APP_DIR" /etc/iron-spyder "$STATE_DIR" /var/backups/iron-spyder

log "Code -> $DEPLOY_REF"
if [ ! -d "$APP_DIR/.git" ]; then
    git clone "$REPO_URL" "$APP_DIR"
fi
git -C "$APP_DIR" remote set-url origin "$REPO_URL"
git -C "$APP_DIR" fetch --prune origin
if git -C "$APP_DIR" rev-parse --verify -q "origin/${DEPLOY_REF#origin/}^{commit}" >/dev/null; then
    TARGET="origin/${DEPLOY_REF#origin/}"
else
    TARGET="$DEPLOY_REF"
fi
git -C "$APP_DIR" reset --hard "$TARGET"
echo "Deployed commit: $(git -C "$APP_DIR" rev-parse --short HEAD)"

log "State tree"
for sub in "${STATE_DIRS[@]}"; do
    mkdir -p "$STATE_DIR/$sub"
done
chown -R "$SVC_UID:$SVC_UID" "$STATE_DIR"
chmod 0755 "$STATE_DIR"
find "$STATE_DIR" -type d -exec chmod 0755 {} +
find "$STATE_DIR/reports" -type f -name '*.json' -exec chmod 0644 {} + 2>/dev/null || true
[ -f "$STATE_DIR/live_state.json" ] && chmod 0644 "$STATE_DIR/live_state.json"

cat > "$STATE_DIR/deploy.json" <<JSON
{
  "commit": "$(git -C "$APP_DIR" rev-parse HEAD)",
  "short_commit": "$(git -C "$APP_DIR" rev-parse --short HEAD)",
  "ref": "$TARGET",
  "deployed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "app_dir": "$APP_DIR",
  "repo_url": "$REPO_URL",
  "compute": "cpu",
  "runtime": "docker-compose"
}
JSON
chown "$SVC_UID:$SVC_UID" "$STATE_DIR/deploy.json"
chmod 0644 "$STATE_DIR/deploy.json"

log "Config template (only if absent)"
if [ ! -f "$CONFIG_FILE" ]; then
    install -D -m 644 -o root -g "$SVC_USER" \
        "$APP_DIR/deploy/config.yaml.example" "$CONFIG_FILE"
fi

log "Systemd units"
install -m 644 "$APP_DIR/deploy/iron-spyder.service" /etc/systemd/system/iron-spyder.service
install -m 644 "$APP_DIR/deploy/iron-spyder-update.service" /etc/systemd/system/iron-spyder-update.service
install -m 644 "$APP_DIR/deploy/iron-spyder-update.timer" /etc/systemd/system/iron-spyder-update.timer
# Remove legacy per-process units if a prior deploy installed them.
for legacy in iron-spyder-supervisor iron-spyder-dashboard-api; do
    systemctl disable --now "$legacy" 2>/dev/null || true
    rm -f "/etc/systemd/system/${legacy}.service"
done
systemctl daemon-reload
systemctl enable --now iron-spyder-update.timer

if [ ! -f "$ENV_FILE" ]; then
    log "Secrets file not found"
    cat <<EOF

$ENV_FILE is missing. Create it once by hand, then re-run (or wait for the
self-update timer):

  sudo install -D -m 640 -o root -g $SVC_USER \\
       $APP_DIR/deploy/iron-spyder.env.example $ENV_FILE
  sudo nano $ENV_FILE

This host is the dedicated Iron-Spyder CPU VPS (see deploy/CPU_VPS.md).
Do not install NVIDIA drivers or co-tenant SPY-DER / 0DTE here.
EOF
    exit 0
fi

# Compose reads $APP_DIR/.env; keep it in sync with the secrets file without
# ever inventing secrets from the example template.
install -m 640 -o root -g root "$ENV_FILE" "$COMPOSE_ENV"

log "Build and start CPU-only Compose stack"
cd "$APP_DIR"
docker compose -f "$APP_DIR/docker-compose.yml" pull || true
docker compose -f "$APP_DIR/docker-compose.yml" build
systemctl enable iron-spyder
systemctl restart iron-spyder

log "Deploy complete (compute=cpu, runtime=docker-compose)"
docker compose -f "$APP_DIR/docker-compose.yml" ps || true
