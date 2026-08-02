#!/usr/bin/env bash
# remote-deploy.sh — idempotent deploy of Iron-Spyder on a VPS.
#
# Runs ON the VPS as root. Provisions on first run (service user, checkout,
# venv, state tree, systemd units) and fast-forwards on every later run. Safe to
# re-run any number of times.
#
# The ONE thing this never touches is the secrets file
# (/etc/iron-spyder/iron-spyder.env): it is created by hand once and the script
# refuses to start units until it exists, so a key is never overwritten by a
# deploy.
set -euo pipefail

REPO_URL="${IRON_SPYDER_REPO_URL:-https://github.com/DGator86/Iron-Spyder.git}"
DEPLOY_REF="${DEPLOY_REF:-origin/main}"
APP_DIR="${IRON_SPYDER_APP_DIR:-/opt/iron-spyder}"
ENV_FILE=/etc/iron-spyder/iron-spyder.env
CONFIG_FILE=/etc/iron-spyder/config.yaml
STATE_DIR=/var/lib/iron-spyder
SVC_USER=iron-spyder

#: Long-running services. Installed and restarted every deploy.
SERVICES=(
    iron-spyder-supervisor
    iron-spyder-dashboard-api
)

#: Timer-driven units. Installed and enabled; the timer owns the schedule.
TIMERS=(
    iron-spyder-update
)

#: State subdirectories the VPS runtime owns (deploy/config.yaml.example).
STATE_DIRS=(
    health decisions positions audit reports configs
)

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo "remote-deploy.sh must run as root (got uid $(id -u))." >&2
    exit 1
fi

log "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git >/dev/null

log "Service user + directories"
id -u "$SVC_USER" >/dev/null 2>&1 || \
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
mkdir -p "$APP_DIR" /etc/iron-spyder "$STATE_DIR"

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

log "Virtualenv + package"
if [ ! -x "$APP_DIR/venv/bin/python" ]; then
    python3 -m venv "$APP_DIR/venv"
fi
"$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
# Editable: console scripts resolve against the checkout. Re-running the install
# is what picks up NEW entry points and NEW dependencies.
"$APP_DIR/venv/bin/pip" install --quiet -e "$APP_DIR[dashboard]"

log "State tree"
for sub in "${STATE_DIRS[@]}"; do
    mkdir -p "$STATE_DIR/$sub"
done
# Own the tree as the service user and leave it world-readable. Readable matters:
# live_state and reports are a published surface that other local readers must
# be able to open. Do not chown this tree to another user — the units declare
# StateDirectory=iron-spyder, and systemd resets ownership on start.
chown -R "$SVC_USER:$SVC_USER" "$STATE_DIR"
chmod 0755 "$STATE_DIR"
find "$STATE_DIR" -type d -exec chmod 0755 {} +
find "$STATE_DIR/reports" -type f -name '*.json' -exec chmod 0644 {} + 2>/dev/null || true
[ -f "$STATE_DIR/live_state.json" ] && chmod 0644 "$STATE_DIR/live_state.json"

# Publish what is deployed so the dashboard can answer "did my change land?"
cat > "$STATE_DIR/deploy.json" <<JSON
{
  "commit": "$(git -C "$APP_DIR" rev-parse HEAD)",
  "short_commit": "$(git -C "$APP_DIR" rev-parse --short HEAD)",
  "ref": "$TARGET",
  "deployed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "app_dir": "$APP_DIR",
  "repo_url": "$REPO_URL"
}
JSON
chown "$SVC_USER:$SVC_USER" "$STATE_DIR/deploy.json"
chmod 0644 "$STATE_DIR/deploy.json"

log "Config template (only if absent)"
if [ ! -f "$CONFIG_FILE" ]; then
    install -D -m 644 -o root -g "$SVC_USER" \
        "$APP_DIR/deploy/config.yaml.example" "$CONFIG_FILE"
fi

log "Systemd units"
for unit in "${SERVICES[@]}"; do
    install -m 644 "$APP_DIR/deploy/${unit}.service" "/etc/systemd/system/${unit}.service"
done
for unit in "${TIMERS[@]}"; do
    install -m 644 "$APP_DIR/deploy/${unit}.service" "/etc/systemd/system/${unit}.service"
    install -m 644 "$APP_DIR/deploy/${unit}.timer" "/etc/systemd/system/${unit}.timer"
done
systemctl daemon-reload

# Always enable the self-update timer so the first deploy is not also the last.
systemctl enable --now iron-spyder-update.timer

if [ ! -f "$ENV_FILE" ]; then
    log "Secrets file not found"
    cat <<EOF

$ENV_FILE is missing. Create it once by hand, then re-run (or wait for the
self-update timer):

  sudo install -D -m 640 -o root -g $SVC_USER \\
       $APP_DIR/deploy/iron-spyder.env.example $ENV_FILE
  sudo nano $ENV_FILE

Units were installed but not started. The next self-update will start them
once the secrets file exists.
EOF
    exit 0
fi

log "Restart runtime services"
for unit in "${SERVICES[@]}"; do
    systemctl enable --now "$unit"
    systemctl restart "$unit"
done

log "Deploy complete"
systemctl --no-pager --full status "${SERVICES[@]}" || true
