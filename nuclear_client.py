"""
nuclear_client.py  — T-P1-12

Process-wide aiohttp.ClientSession singleton for nuclear-voice.
All outbound HTTP calls (Fortress /v1/agents/:agent/chat) go through
this module — no bare aiohttp.ClientSession() or urllib.request
calls elsewhere in this service.

Chain-ready: when CHAIN_ENABLED=true all Fortress calls are routed
through nuclear-chain (:9100) instead of hitting Fortress directly.
"""

import asyncio
import logging
import os
from typing import Any

import aiohttp

log = logging.getLogger("nuclear-voice.nuclear_client")

# ── Config ────────────────────────────────────────────────────────────────────

FORTRESS_URL   = os.environ.get("FORTRESS_URL",        "http://192.168.2.23:7700")
CHAIN_ENABLED  = os.environ.get("CHAIN_ENABLED",       "false").lower() == "true"
# T-P2-06: NUCLEAR_CHAIN_TRANSPORT_URL is the preferred env var for the proxy path.
# NUCLEAR_CHAIN_URL is the legacy fallback (deprecated for one release).
CHAIN_URL      = (
    os.environ.get("NUCLEAR_CHAIN_TRANSPORT_URL", "").strip()
    or os.environ.get("NUCLEAR_CHAIN_URL", "http://localhost:9100")
)
SERVICE_TOKEN  = os.environ.get("NUCLEAR_SERVICE_TOKEN", "")
SERVICE_NAME   = os.environ.get("NUCLEAR_SERVICE_NAME",  "nuclear-voice")

# ── Session singleton ─────────────────────────────────────────────────────────

_session: aiohttp.ClientSession | None = None


def get_session() -> aiohttp.ClientSession:
    """Return (or create) the process-wide aiohttp.ClientSession."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
        log.debug("nuclear_client: session created")
    return _session


async def close() -> None:
    """Gracefully close the session at process shutdown."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None
        log.debug("nuclear_client: session closed")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fortress_url(path: str) -> str:
    """Build a Fortress URL, routing through nuclear-chain when CHAIN_ENABLED."""
    if CHAIN_ENABLED and CHAIN_URL:
        base = CHAIN_URL.rstrip("/")
    else:
        base = FORTRESS_URL.rstrip("/")
    return f"{base}{path}"


def _service_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if SERVICE_TOKEN:
        headers["X-Nuclear-Token"] = SERVICE_TOKEN
    if SERVICE_NAME:
        headers["X-Nuclear-Service"] = SERVICE_NAME
    return headers


# ── Fortress API wrappers ─────────────────────────────────────────────────────

_VALID_EMOTIONS = frozenset({"warm", "decisive", "uncertain", "impulsive"})


async def fortress_chat(
    agent: str,
    message: str,
    user_id: str = "user",
    language: str = "fr",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """
    POST /v1/agents/:agent/chat → Fortress.

    Returns dict with at minimum:
      {"reply": str, "emotion": str}

    Raises on HTTP error; caller is responsible for catching.
    """
    url = _fortress_url(f"/v1/agents/{agent}/chat")
    payload = {"message": message, "user_id": user_id, "language": language}
    async with get_session().post(
        url,
        json=payload,
        headers=_service_headers(),
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        resp.raise_for_status()
        data: dict[str, Any] = await resp.json()

    reply   = data.get("reply") or data.get("text") or data.get("message", "")
    emotion = data.get("emotion", "warm")
    if emotion not in _VALID_EMOTIONS:
        emotion = "warm"
    return {"reply": reply, "emotion": emotion}
