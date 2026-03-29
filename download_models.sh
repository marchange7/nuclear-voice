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

echo '==> Downloading Kokoro ONNX TTS models...'
mkdir -p /data/models/kokoro
wget -q --show-progress   -O /data/models/kokoro/kokoro-v1.0.int8.onnx   'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx'
wget -q --show-progress   -O /data/models/kokoro/voices-v1.0.bin   'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin'
echo '  Kokoro models cached at /data/models/kokoro/'
