#!/usr/bin/env bash
# download_models.sh — Pull all models required by nuclear-voice
set -euo pipefail

VENV="${HOME}/.venv"
PYTHON="${VENV}/bin/python"

echo "==> Downloading faster-whisper 'small' model…"
"${PYTHON}" - <<'EOF'
from faster_whisper import WhisperModel
print("  Fetching model weights (this may take a minute)…")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("  faster-whisper 'small' cached.")
EOF

echo "==> Pre-fetching silero-VAD model…"
"${PYTHON}" - <<'EOF'
import torch
print("  Fetching silero-vad…")
model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)
print("  silero-vad cached.")
EOF

echo ""
echo "✅  All models downloaded. You can now start the service:"
echo "    sudo systemctl start nuclear-voice"
