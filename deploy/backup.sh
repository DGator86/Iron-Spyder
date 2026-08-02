#!/usr/bin/env bash
# backup.sh — daily encrypted snapshot of Iron-Spyder state + config.
#
# Intended for the dedicated CPU live VPS (cron or a timer). Reads only
# Iron-Spyder paths; never touches legacy SPY-DER / 0DTE trees.
#
#   IRON_SPYDER_BACKUP_KEY_FILE=/root/iron-spyder-backup.key \
#     /opt/iron-spyder/deploy/backup.sh
set -euo pipefail

STATE_DIR=${IRON_SPYDER_STATE_ROOT:-/var/lib/iron-spyder}
CONFIG_DIR=${IRON_SPYDER_CONFIG_DIR:-/etc/iron-spyder}
BACKUP_ROOT=${IRON_SPYDER_BACKUP_ROOT:-/var/backups/iron-spyder}
KEY_FILE=${IRON_SPYDER_BACKUP_KEY_FILE:-}
KEEP_DAYS=${IRON_SPYDER_BACKUP_KEEP_DAYS:-14}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="$BACKUP_ROOT/$STAMP"
ARCHIVE="$BACKUP_ROOT/iron-spyder-$STAMP.tar.gz"

if [ "$(id -u)" -ne 0 ]; then
    echo "backup.sh must run as root" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$BACKUP_ROOT"
tar -C / -czf "$OUT_DIR/payload.tar.gz" \
    "${STATE_DIR#/}" \
    "${CONFIG_DIR#/}" \
    2>/dev/null || tar -C / -czf "$OUT_DIR/payload.tar.gz" "${STATE_DIR#/}"

if [ -n "$KEY_FILE" ] && [ -f "$KEY_FILE" ]; then
    if command -v openssl >/dev/null 2>&1; then
        openssl enc -aes-256-cbc -pbkdf2 -salt \
            -in "$OUT_DIR/payload.tar.gz" \
            -out "$ARCHIVE.enc" \
            -pass "file:$KEY_FILE"
        rm -f "$OUT_DIR/payload.tar.gz"
        rmdir "$OUT_DIR" 2>/dev/null || true
        echo "Wrote $ARCHIVE.enc"
    else
        echo "openssl not installed; leaving unencrypted $OUT_DIR/payload.tar.gz" >&2
        mv "$OUT_DIR/payload.tar.gz" "$ARCHIVE"
        rmdir "$OUT_DIR" 2>/dev/null || true
    fi
else
    mv "$OUT_DIR/payload.tar.gz" "$ARCHIVE"
    rmdir "$OUT_DIR" 2>/dev/null || true
    echo "Wrote $ARCHIVE (set IRON_SPYDER_BACKUP_KEY_FILE for encryption)"
fi

find "$BACKUP_ROOT" -type f \( -name 'iron-spyder-*.tar.gz' -o -name 'iron-spyder-*.tar.gz.enc' \) \
    -mtime "+$KEEP_DAYS" -delete 2>/dev/null || true
