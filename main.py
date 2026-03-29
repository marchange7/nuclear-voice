"""nuclear-voice — Shared STT/TTS/VAD service for the Nuclear Fortress ecosystem."""

import asyncio
import base64
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nuclear-voice")

app = FastAPI(title="nuclear-voice", version="0.1.0")
executor = ThreadPoolExecutor(max_workers=4)

# ── Module-level model caches ──────────────────────────────────────────────────

_whisper_model = None
_tts_engine = None
_tts_backend = None  # "kokoro" | "voxtral" | "pyttsx3" | "espeak"

# Kokoro ONNX — paths configurable per machine (b450: /data/models, M4: ~/models)
KOKORO_MODEL  = os.environ.get("KOKORO_MODEL",  "/data/models/kokoro/kokoro-v1.0.int8.onnx")
KOKORO_VOICES = os.environ.get("KOKORO_VOICES", "/data/models/kokoro/voices-v1.0.bin")
KOKORO_VOICE  = os.environ.get("KOKORO_VOICE",  "ff_siwis")  # fr female; bf_emma = EN female"
_vad_model = None
_vad_backend = None  # "silero" | "webrtcvad"

STT_MODEL = os.environ.get("VOICE_STT_MODEL", "small")
VOICE_LANGUAGE = os.environ.get("VOICE_LANGUAGE", "fr")


# ── Model loaders ──────────────────────────────────────────────────────────────

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        log.info("Loading faster-whisper model: %s", STT_MODEL)
        _whisper_model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
        log.info("faster-whisper ready")
    return _whisper_model


def _load_tts():
    global _tts_engine, _tts_backend
    if _tts_engine is not None:
        return _tts_engine, _tts_backend

    # 1. Kokoro ONNX — fast, high quality, works on CPU (b450) and Apple Silicon (M4)
    try:
        if os.path.exists(KOKORO_MODEL) and os.path.exists(KOKORO_VOICES):
            from kokoro_onnx import Kokoro
            log.info("Loading Kokoro ONNX TTS: %s", KOKORO_MODEL)
            _tts_engine = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
            _tts_backend = "kokoro"
            log.info("Kokoro TTS ready (voice=%s)", KOKORO_VOICE)
            return _tts_engine, _tts_backend
        else:
            log.warning("Kokoro models not found at %s — run download_models.sh", KOKORO_MODEL)
    except Exception as e:
        log.warning("Kokoro unavailable (%s), trying Voxtral…", e)

    # 2. Voxtral
    try:
        from transformers import pipeline as hf_pipeline
        log.info("Loading Voxtral-4B TTS…")
        _tts_engine = hf_pipeline(
            "text-to-speech",
            model="mistralai/Voxtral-4B-TTS-2603",
            device="cpu",
        )
        _tts_backend = "voxtral"
        log.info("Voxtral TTS ready")
        return _tts_engine, _tts_backend
    except Exception as e:
        log.warning("Voxtral unavailable (%s), trying pyttsx3…", e)

    # Fallback: pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        _tts_engine = engine
        _tts_backend = "pyttsx3"
        log.info("pyttsx3 TTS ready")
        return _tts_engine, _tts_backend
    except Exception as e:
        log.warning("pyttsx3 unavailable (%s), using espeak…", e)

    # Fallback: espeak via subprocess
    _tts_engine = "espeak"
    _tts_backend = "espeak"
    log.info("espeak TTS backend selected")
    return _tts_engine, _tts_backend


def _load_vad():
    global _vad_model, _vad_backend
    if _vad_model is not None:
        return _vad_model, _vad_backend

    # Try silero-vad
    try:
        import torch
        log.info("Loading silero-VAD…")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        _vad_model = (model, utils)
        _vad_backend = "silero"
        log.info("silero-VAD ready")
        return _vad_model, _vad_backend
    except Exception as e:
        log.warning("silero-VAD unavailable (%s), falling back to webrtcvad…", e)

    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)  # aggressiveness 0–3
        _vad_model = vad
        _vad_backend = "webrtcvad"
        log.info("webrtcvad ready")
        return _vad_model, _vad_backend
    except Exception as e:
        raise RuntimeError(f"No VAD backend available: {e}") from e


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _b64_to_numpy(audio_b64: str) -> tuple[np.ndarray, int]:
    """Decode base64 WAV → (float32 array, sample_rate)."""
    raw = base64.b64decode(audio_b64)
    buf = io.BytesIO(raw)
    data, sr = sf.read(buf, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _numpy_to_b64(audio: np.ndarray, sr: int) -> str:
    """Encode float32 numpy array → base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Blocking inference workers ─────────────────────────────────────────────────

def _do_transcribe(audio_b64: str, language: Optional[str]) -> dict:
    model = _load_whisper()
    audio, sr = _b64_to_numpy(audio_b64)
    duration_ms = int(len(audio) / sr * 1000)

    t0 = time.monotonic()
    lang = language or VOICE_LANGUAGE
    segments, info = model.transcribe(audio, language=lang, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    detected = info.language if hasattr(info, "language") else lang
    elapsed = int((time.monotonic() - t0) * 1000)

    return {"text": text, "language": detected, "duration_ms": elapsed}


# Emotion → prosody hint (injected as prefix for Voxtral / rate for pyttsx3)
_EMOTION_RATE = {
    "warm": 150,
    "decisive": 175,
    "uncertain": 130,
    "impulsive": 200,
}


def _do_speak(text: str, emotion: str, language: str) -> dict:
    engine, backend = _load_tts()
    sr = 22050
    t0 = time.monotonic()

    if backend == "kokoro":
        lang_map = {"fr": "fr-fr", "en": "en-us", "fr-fr": "fr-fr", "en-us": "en-us"}
        lang = lang_map.get(language[:5] if language else "fr", "fr-fr")
        audio_array, sr = engine.create(text, voice=KOKORO_VOICE, speed=1.0, lang=lang)

    elif backend == "voxtral":
        result = engine(text)
        audio_array = np.array(result["audio"], dtype=np.float32).squeeze()
        sr = result.get("sampling_rate", 22050)

    elif backend == "pyttsx3":
        rate = _EMOTION_RATE.get(emotion, 150)
        engine.setProperty("rate", rate)
        buf = io.BytesIO()
        # pyttsx3 can save to a file path; use a temp buffer via save_to_file trick
        import tempfile, os, soundfile as _sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        audio_array, sr = _sf.read(tmp_path, dtype="float32")
        os.unlink(tmp_path)

    else:  # espeak
        import subprocess, tempfile, soundfile as _sf
        rate = _EMOTION_RATE.get(emotion, 150)
        lang_code = language[:2] if language else "fr"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        subprocess.run(
            ["espeak", "-v", lang_code, "-s", str(rate), "-w", tmp_path, text],
            check=True,
        )
        audio_array, sr = _sf.read(tmp_path, dtype="float32")
        os.unlink(tmp_path)

    duration_ms = int((time.monotonic() - t0) * 1000)
    audio_b64 = _numpy_to_b64(audio_array, sr)
    return {"audio_b64": audio_b64, "format": "wav", "duration_ms": duration_ms}


def _do_vad(audio_b64: str) -> dict:
    model_obj, backend = _load_vad()
    audio, sr = _b64_to_numpy(audio_b64)

    if backend == "silero":
        import torch
        model, utils = model_obj
        (get_speech_timestamps, _, _, _, _) = utils
        tensor = torch.from_numpy(audio).float()
        if sr != 16000:
            import torchaudio
            tensor = torchaudio.functional.resample(tensor, sr, 16000)
        timestamps = get_speech_timestamps(tensor, model, sampling_rate=16000)
        is_speech = len(timestamps) > 0
        confidence = float(len(timestamps)) / max(1, len(timestamps) + 1)
        if is_speech:
            confidence = min(0.99, 0.7 + 0.3 * min(1.0, len(timestamps) / 3))

    else:  # webrtcvad
        import webrtcvad
        vad_obj = model_obj
        # webrtcvad needs 16-bit PCM at 8/16/32/48 kHz, 10/20/30ms frames
        target_sr = 16000
        if sr != target_sr:
            # simple resample via numpy
            ratio = target_sr / sr
            n_new = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_new),
                np.arange(len(audio)),
                audio,
            )
        pcm = (audio * 32767).astype(np.int16).tobytes()
        frame_duration_ms = 20
        frame_len = int(target_sr * frame_duration_ms / 1000) * 2  # bytes (16-bit)
        frames = [pcm[i: i + frame_len] for i in range(0, len(pcm), frame_len) if len(pcm[i: i + frame_len]) == frame_len]
        speech_frames = sum(1 for f in frames if webrtcvad.Vad(2).is_speech(f, target_sr))
        is_speech = speech_frames > 0
        confidence = speech_frames / max(1, len(frames))

    return {"is_speech": is_speech, "confidence": round(confidence, 4)}


# ── Request / Response schemas ─────────────────────────────────────────────────

class TranscribeRequest(BaseModel):
    audio_b64: str
    language: Optional[str] = None


class SpeakRequest(BaseModel):
    text: str
    emotion: str = "warm"
    language: str = "fr"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor, _do_transcribe, req.audio_b64, req.language
        )
    except Exception as e:
        log.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return result


@app.post("/speak")
async def speak(req: SpeakRequest):
    if req.emotion not in ("warm", "decisive", "uncertain", "impulsive"):
        raise HTTPException(status_code=422, detail=f"Unknown emotion: {req.emotion}")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor, _do_speak, req.text, req.emotion, req.language
        )
    except Exception as e:
        log.exception("TTS failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return result


@app.get("/vad")
async def vad(audio_b64: str = Query(...)):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, _do_vad, audio_b64)
    except Exception as e:
        log.exception("VAD failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return result


@app.get("/health")
async def health():
    stt_info = f"faster-whisper-{STT_MODEL}"
    tts_info = _tts_backend or "not-loaded"
    vad_info = _vad_backend or "not-loaded"
    return {"status": "ok", "stt": stt_info, "tts": tts_info, "vad": vad_info}
