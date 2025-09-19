from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import torch
import whisper
from fastapi import APIRouter, Header, HTTPException, Request

router = APIRouter()

# Configuration
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v2")
if "WHISPER_DEVICE" in os.environ:
    WHISPER_DEVICE = os.environ["WHISPER_DEVICE"]
elif torch.cuda.is_available():
    WHISPER_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda"
else:
    WHISPER_DEVICE = "cpu"
SESSION_TTL = timedelta(minutes=5)
MAX_BUFFER_SECONDS = 30
TARGET_SAMPLE_RATE = 16000

_sessions: Dict[str, "SessionState"] = {}
_sessions_lock = asyncio.Lock()
_model_lock = asyncio.Lock()
_model = None


@dataclass
class SessionState:
    audio_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    last_text: str = ""
    sample_rate: int = TARGET_SAMPLE_RATE
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def append_audio(self, chunk: np.ndarray) -> None:
        """Append chunk and trim to the most recent window."""
        if not isinstance(chunk, np.ndarray) or chunk.dtype != np.float32:
            raise ValueError("chunk must be np.float32")
        if chunk.size == 0:
            return
        self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
        max_samples = TARGET_SAMPLE_RATE * MAX_BUFFER_SECONDS
        if self.audio_buffer.size > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
        self.last_updated = datetime.utcnow()


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
    return _model


def _resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)
    if orig_sr <= 0:
        raise ValueError("sample rate must be positive")
    new_len = int(round(audio.shape[0] * TARGET_SAMPLE_RATE / float(orig_sr)))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.interp(
        np.linspace(0.0, 1.0, num=new_len, endpoint=False),
        np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False),
        audio.astype(np.float32),
    ).astype(np.float32)


async def _transcribe_audio(buffer: np.ndarray) -> str:
    model = _get_model()
    if buffer.size == 0:
        return ""

    loop = asyncio.get_running_loop()

    def _run() -> str:
        result = model.transcribe(
            buffer,
            fp16=getattr(model, "device", torch.device("cpu")).type == "cuda",
            language="tr",         
            task="transcribe",      
        )

        return result.get("text", "").strip()

    async with _model_lock:
        return await loop.run_in_executor(None, _run)


async def _get_session(session_id: str) -> SessionState:
    async with _sessions_lock:
        state = _sessions.get(session_id)
        if state is None:
            state = SessionState()
            _sessions[session_id] = state
        return state


async def _remove_session(session_id: str) -> None:
    async with _sessions_lock:
        _sessions.pop(session_id, None)


async def _cleanup_sessions() -> None:
    now = datetime.utcnow()
    async with _sessions_lock:
        expired = [sid for sid, state in _sessions.items() if now - state.last_updated > SESSION_TTL]
        for sid in expired:
            _sessions.pop(sid, None)


@router.post("/speech/stream")
async def stream_speech(
    request: Request,
    session_id: str = Header(..., alias="X-Session-Id"),
    sample_rate: int = Header(..., alias="X-Sample-Rate"),
    finalize: bool = Header(False, alias="X-Finalize"),
):
    raw = await request.body()
    if not raw:
        session = await _get_session(session_id)
        text = session.last_text
        if finalize:
            await _remove_session(session_id)
        else:
            await _cleanup_sessions()
        return {"text": text, "delta": "", "is_final": bool(finalize)}
    if len(raw) % 2 != 0:
        raise HTTPException(status_code=400, detail="Audio payload must be 16-bit PCM")

    try:
        pcm16 = np.frombuffer(raw, dtype=np.int16)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid audio payload") from exc

    audio = pcm16.astype(np.float32) / 32768.0
    try:
        audio = _resample_to_16k(audio, sample_rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session = await _get_session(session_id)
    session.append_audio(audio)

    text = await _transcribe_audio(session.audio_buffer)
    delta_text = text[len(session.last_text) :].lstrip() if text.startswith(session.last_text) else text
    session.last_text = text

    if finalize:
        await _remove_session(session_id)
    else:
        await _cleanup_sessions()

    return {"text": text, "delta": delta_text, "is_final": bool(finalize)}
