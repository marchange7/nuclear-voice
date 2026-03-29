# nuclear-voice

Shared STT / TTS / VAD microservice for the **Nuclear Fortress** ecosystem.

| Endpoint | Method | Description |
|---|---|---|
| `/transcribe` | POST | Speech-to-text via faster-whisper |
| `/speak` | POST | Text-to-speech via Voxtral-4B (pyttsx3/espeak fallback) |
| `/vad` | GET | Voice activity detection via silero-vad (webrtcvad fallback) |
| `/health` | GET | Service + model status |

## Quick start (b450)

```bash
cd ~/git/nuclear-voice
bash download_models.sh          # one-time model pull
sudo systemctl start nuclear-voice
curl http://localhost:8083/health
```

## Emotions (TTS)

`warm` · `decisive` · `uncertain` · `impulsive`

## Environment

| Variable | Default | Description |
|---|---|---|
| `VOICE_STT_MODEL` | `small` | faster-whisper model size |
| `VOICE_LANGUAGE` | `fr` | Default language |
