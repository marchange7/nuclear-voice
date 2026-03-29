#!/usr/bin/env bash
# nuclear-voice systemd install script
# Run as: sudo bash deploy/install.sh
# Must be executed on b450 from /home/crew/git/nuclear-voice

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"
CONFIG_DIR="/home/crew/.config/nuclear-voice"
SERVICE_USER="crew"

# ── Ensure Python venv with deps ──────────────────────────────────────────
VENV="/home/crew/.venv/nuclear"
if [[ ! -f "${VENV}/bin/uvicorn" ]]; then
    echo "→ Installing Python deps into ${VENV}…"
    sudo -u "${SERVICE_USER}" python3 -m venv "${VENV}"
    sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install --upgrade pip
    sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install \
        fastapi uvicorn[standard] \
        faster-whisper \
        kokoro-onnx \
        soundfile numpy \
        webrtcvad-wheels \
        httpx
fi

# ── Ensure config directory and env file ─────────────────────────────────
echo "→ Ensuring config directory at ${CONFIG_DIR}…"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 750 "${CONFIG_DIR}"

if [[ ! -f "${CONFIG_DIR}/env" ]]; then
    echo "→ Creating stub env file…"
    cat > "${CONFIG_DIR}/env" <<'EOF'
# nuclear-voice environment
VOICE_MODEL=kokoro
STT_MODEL=base
FORTRESS_URL=http://192.168.2.23:7700
CONVERSE_AGENT=emile
ALERT_LANG=fr
EOF
    chown "${SERVICE_USER}:${SERVICE_USER}" "${CONFIG_DIR}/env"
    chmod 640 "${CONFIG_DIR}/env"
fi

# ── Install service file ──────────────────────────────────────────────────
echo "→ Installing nuclear-voice.service…"
install -m 644 "${DEPLOY_DIR}/nuclear-voice.service" "${SYSTEMD_DIR}/nuclear-voice.service"

# ── Reload and enable ─────────────────────────────────────────────────────
echo "→ Reloading systemd daemon…"
systemctl daemon-reload

echo "→ Enabling nuclear-voice.service…"
systemctl enable nuclear-voice.service

echo ""
echo "Done. Start with:"
echo "  sudo systemctl start nuclear-voice"
echo "  sudo systemctl status nuclear-voice"
echo ""
echo "First start loads Whisper+Kokoro models — allow ~60s."
