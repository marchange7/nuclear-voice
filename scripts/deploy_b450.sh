#!/usr/bin/env bash
# deploy_b450.sh — Deploy nuclear-voice from M4 to b450
#
# Run from M4: bash scripts/deploy_b450.sh [--start]
#
# Steps:
#   1. rsync source to b450:/data/git/nuclear-voice
#   2. SSH → sudo bash deploy/install.sh  (venv + deps + systemd unit)
#   3. [--start] Start the service
#
# Prerequisites:
#   ssh alias b450 = crew@192.168.2.23 (in ~/.ssh/config)

set -euo pipefail

B450_HOST="${B450_HOST:-b450}"
REMOTE_DIR="${REMOTE_DIR:-/data/git/nuclear-voice}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SERVICES="${1:-}"

echo "🚀 nuclear-voice → b450 deploy"
echo "   src  : $ROOT_DIR"
echo "   dst  : $B450_HOST:$REMOTE_DIR"
echo ""

# ── 1. Rsync source ───────────────────────────────────────────────────────────
echo "📤 Syncing source to $B450_HOST..."
rsync -az --delete \
  --exclude ".git/" \
  --exclude "*.log" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "models/" \
  "$ROOT_DIR/" "$B450_HOST:$REMOTE_DIR/"
echo "   ✓ rsync done"

# ── 2. Install (venv deps + systemd unit) ────────────────────────────────────
echo ""
echo "🔨 Running install on $B450_HOST..."
ssh "$B450_HOST" "cd '$REMOTE_DIR' && sudo bash deploy/install.sh"
echo "   ✓ install done"

# ── 3. Start service (optional) ──────────────────────────────────────────────
if [[ "$START_SERVICES" == "--start" ]]; then
  echo ""
  echo "▶  Starting nuclear-voice on $B450_HOST..."
  ssh "$B450_HOST" "sudo systemctl start nuclear-voice"
  echo "   ✓ service started (first start loads Whisper+Kokoro — allow ~60s)"
  echo ""
  echo "📋 Status:"
  ssh "$B450_HOST" "sudo systemctl status nuclear-voice --no-pager -l | head -20"
else
  echo ""
  echo "✅ Deploy complete. To start service:"
  echo "   bash scripts/deploy_b450.sh --start"
  echo "   — or on b450 directly:"
  echo "   sudo systemctl start nuclear-voice"
fi
