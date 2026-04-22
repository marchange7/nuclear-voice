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

import nuclear_client  # T-P1-12: process-wide HTTP session

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nuclear-voice")

app = FastAPI(title="nuclear-voice", version="0.1.0")
executor = ThreadPoolExecutor(max_workers=4)

# ── Module-level model caches ──────────────────────────────────────────────────

_whisper_model = None
_tts_engine = None
_tts_backend = None  # "kokoro" | "voxtral" | "pyttsx3" | "espeak"

# Kokoro ONNX paths — override via env vars on each machine (b450 or M4)
KOKORO_MODEL = os.environ.get("KOKORO_MODEL", "/data/models/kokoro/kokoro-v1.9.onnx")
KOKORO_VOICES = os.environ.get("KOKORO_VOICES", "/data/models/kokoro/voices-v1.0.bin")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "ff_siwis")  # French female — change to bf_emma for EN
_vad_model = None
_vad_backend = None  # "silero" | "webrtcvad"

STT_MODEL = os.environ.get("VOICE_STT_MODEL", "small")
VOICE_LANGUAGE = os.environ.get("VOICE_LANGUAGE", "fr")

# ── Conversational pipeline config ─────────────────────────────────────────────
# FORTRESS_URL moved to nuclear_client.py (T-P1-12)
CONVERSE_AGENT = os.environ.get("CONVERSE_AGENT", "emile")  # agent to route conversation to


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

    # Primary: Kokoro ONNX (82M params, CPU-efficient)
    if os.path.exists(KOKORO_MODEL) and os.path.exists(KOKORO_VOICES):
        try:
            from kokoro_onnx import Kokoro
            log.info("Loading Kokoro ONNX TTS: %s", KOKORO_MODEL)
            _tts_engine = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
            _tts_backend = "kokoro"
            log.info("Kokoro TTS ready (voice=%s)", KOKORO_VOICE)
            return _tts_engine, _tts_backend
        except Exception as e:
            log.warning("Kokoro unavailable (%s), trying Voxtral…", e)
    else:
        log.warning("Kokoro model files not found (%s / %s), trying Voxtral…", KOKORO_MODEL, KOKORO_VOICES)

    # Fallback: Voxtral (4B params)
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
        log.warning("Voxtral unavailable (%s), using espeak…", e)

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

# Emotion → Kokoro speed multiplier (1.0 = normal, range ~0.5–2.0)
_EMOTION_SPEED = {
    "warm": 1.0,
    "decisive": 1.15,
    "uncertain": 0.85,
    "impulsive": 1.3,
}


# ── Phoneme helpers ───────────────────────────────────────────────────────────

# Grapheme → phoneme label (maps to Oculus viseme names in the renderer).
# French + English digraphs first (order matters — longest match wins).
_GRAPHEME_RULES: list[tuple[str, str]] = [
    ("eau", "o"), ("ou", "u"), ("au", "o"), ("ai", "e"), ("ei", "e"),
    ("oi", "o"), ("eu", "u"), ("ch", "sh"), ("gn", "n"), ("qu", "k"),
    ("ph", "f"), ("th", "th"), ("sh", "sh"), ("oo", "u"), ("ee", "i"),
    ("ea", "i"), ("oa", "o"), ("ng", "n"), ("ck", "k"), ("wh", "f"),
    ("b", "b"), ("c", "k"), ("d", "d"), ("f", "f"), ("g", "g"),
    ("h", "sil"), ("j", "j"), ("k", "k"), ("l", "l"), ("m", "m"),
    ("n", "n"), ("p", "p"), ("q", "k"), ("r", "r"), ("s", "s"),
    ("t", "t"), ("v", "v"), ("w", "u"), ("x", "k"), ("y", "i"), ("z", "z"),
    ("a", "aa"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "u"),
]
# Phoneme → Oculus viseme name
_PHONEME_TO_VISEME: dict[str, str] = {
    "sil": "viseme_sil",
    "p": "viseme_PP", "b": "viseme_PP", "m": "viseme_PP",
    "f": "viseme_FF", "v": "viseme_FF",
    "th": "viseme_TH",
    "d": "viseme_DD", "t": "viseme_DD", "n": "viseme_nn",
    "k": "viseme_kk", "g": "viseme_kk",
    "ch": "viseme_CH", "j": "viseme_CH", "sh": "viseme_CH",
    "s": "viseme_SS", "z": "viseme_SS",
    "r": "viseme_RR", "l": "viseme_RR",
    "aa": "viseme_aa", "a": "viseme_aa", "e": "viseme_E",
    "i": "viseme_I", "o": "viseme_O", "u": "viseme_U",
}
_PH_DURATION_MS = 80
_WORD_GAP_MS = 40


def _text_to_phoneme_timeline(text: str, audio_duration_ms: float) -> list[dict]:
    """
    Convert text to a timed phoneme/viseme sequence scaled to actual audio duration.

    Returns list of:
        {"phoneme": str, "viseme": str, "start_ms": int, "end_ms": int}

    The timeline is derived via grapheme rules then linearly scaled so that
    the last phoneme ends exactly at audio_duration_ms, giving the renderer
    timing that stays in sync even when TTS speed varies.
    """
    import re
    words = re.findall(r"[a-zàâäéèêëïîôùûüÿçœæ]+", text.lower())

    raw: list[tuple[str, int, int]] = []  # (phoneme, start_ms, end_ms)
    cursor = 0
    for word in words:
        i = 0
        while i < len(word):
            for pattern, phoneme in _GRAPHEME_RULES:
                plen = len(pattern)
                if word[i:i + plen] == pattern:
                    if phoneme != "sil":
                        raw.append((phoneme, cursor, cursor + _PH_DURATION_MS))
                        cursor += _PH_DURATION_MS
                    i += plen
                    break
            else:
                i += 1
        cursor += _WORD_GAP_MS

    if not raw:
        return []

    # Scale to actual audio duration
    raw_total = raw[-1][2]  # end_ms of last phoneme
    scale = audio_duration_ms / raw_total if raw_total > 0 else 1.0

    return [
        {
            "phoneme": ph,
            "viseme": _PHONEME_TO_VISEME.get(ph, "viseme_sil"),
            "start_ms": int(start * scale),
            "end_ms": int(end * scale),
        }
        for ph, start, end in raw
    ]


def _do_speak(text: str, emotion: str, language: str, voice_id: Optional[str] = None) -> dict:
    engine, backend = _load_tts()
    sr = 22050
    t0 = time.monotonic()

    if backend == "kokoro":
        speed = _EMOTION_SPEED.get(emotion, 1.0)
        lang_code = language[:2] if language else "fr"
        # Kokoro lang codes: "en-us", "fr-fr", etc. — map 2-letter to Kokoro format
        _KOKORO_LANG = {"fr": "fr-fr", "en": "en-us", "de": "de-de", "es": "es-es"}
        kokoro_lang = _KOKORO_LANG.get(lang_code, f"{lang_code}-{lang_code}")
        voice = voice_id or KOKORO_VOICE
        samples, sr = engine.create(text, voice=voice, speed=speed, lang=kokoro_lang)
        audio_array = np.array(samples, dtype=np.float32)

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
    # Compute audio duration from samples (more accurate than wall-clock)
    audio_duration_ms = int(len(audio_array) / sr * 1000)
    phonemes = _text_to_phoneme_timeline(text, audio_duration_ms)
    return {
        "audio_b64": audio_b64,
        "format": "wav",
        "duration_ms": duration_ms,
        "audio_duration_ms": audio_duration_ms,
        "phonemes": phonemes,
    }


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


# ── Conversational round-trip worker ──────────────────────────────────────────

def _do_converse(audio_b64: str, user_id: str, language: str) -> dict:
    """VAD → STT → Fortress agent chat → TTS — full synchronous round-trip."""
    # 1. VAD: skip synthesis if no speech detected
    vad = _do_vad(audio_b64)
    if not vad["is_speech"]:
        return {"is_speech": False, "transcript": "", "reply": "", "audio_b64": "", "emotion": "warm", "stt_duration_ms": 0}

    # 2. STT
    stt = _do_transcribe(audio_b64, language)
    transcript = stt["text"].strip()
    if not transcript:
        return {"is_speech": True, "transcript": "", "reply": "", "audio_b64": "", "emotion": "warm", "stt_duration_ms": stt["duration_ms"]}

    # 3. Fortress agent chat — T-P1-12: routed through nuclear_client (aiohttp, chain-ready)
    reply = ""
    emotion = "warm"
    try:
        result = asyncio.get_event_loop().run_until_complete(
            nuclear_client.fortress_chat(
                agent=CONVERSE_AGENT,
                message=transcript,
                user_id=user_id,
                language=language,
                timeout=10.0,
            )
        )
        reply   = result["reply"]
        emotion = result["emotion"]
    except Exception as e:
        log.warning("Fortress chat call failed (%s): %s", CONVERSE_AGENT, e)

    # 4. TTS
    reply_audio = ""
    if reply:
        try:
            tts = _do_speak(reply, emotion, language)
            reply_audio = tts["audio_b64"]
        except Exception as e:
            log.warning("TTS failed for converse reply: %s", e)

    return {
        "is_speech": True,
        "transcript": transcript,
        "reply": reply,
        "audio_b64": reply_audio,
        "emotion": emotion,
        "stt_duration_ms": stt["duration_ms"],
    }


# ── Request / Response schemas ─────────────────────────────────────────────────

class TranscribeRequest(BaseModel):
    audio_b64: str
    language: Optional[str] = None


class SpeakRequest(BaseModel):
    text: str
    emotion: str = "warm"
    language: str = "fr"
    voice_id: Optional[str] = None  # Kokoro voice ID; falls back to KOKORO_VOICE env default


class ConversationRequest(BaseModel):
    audio_b64: str
    user_id: str = "user"
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
            executor, _do_speak, req.text, req.emotion, req.language, req.voice_id
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


@app.post("/converse")
async def converse(req: ConversationRequest):
    """Full conversational round-trip: audio → VAD → STT → agent → TTS → audio.

    Returns:
      is_speech: bool — false if VAD detected no speech (audio_b64 will be empty)
      transcript: str — what the user said
      reply: str — agent text response
      audio_b64: str — spoken reply as base64 WAV
      emotion: str — TTS prosody used (warm/decisive/uncertain/impulsive)
      stt_duration_ms: int
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor, _do_converse, req.audio_b64, req.user_id, req.language
        )
    except Exception as e:
        log.exception("Conversation round-trip failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return result


@app.get("/health")
async def health():
    stt_info = f"faster-whisper-{STT_MODEL}"
    tts_info = _tts_backend or "not-loaded"
    vad_info = _vad_backend or "not-loaded"
    return {"status": "ok", "stt": stt_info, "tts": tts_info, "vad": vad_info}
