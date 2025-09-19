from __future__ import annotations

import os
import subprocess
import uuid
from contextlib import contextmanager
from typing import Generator, Iterable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter()

MODEL_PATH = "backend/voices/tr_TR-fahrettin-medium.onnx"
CHUNK_SIZE = 4096
_SENTENCE_ENDINGS = ".!?"
# Piper stops mid-stream when hitting end-of-sentence punctuation, so map them to commas.
_PIPER_TEXT_TRANSLATION = str.maketrans({char: "," for char in _SENTENCE_ENDINGS})


@contextmanager
def _piper_pipeline(text: str) -> Generator[subprocess.Popen, None, None]:
    if not text:
        raise ValueError("text must be non-empty")

    fifo_path = f"/tmp/piper_{uuid.uuid4().hex}.wav"
    os.mkfifo(fifo_path)

    try:
        try:
            piper_proc = subprocess.Popen(
                ["piper", "--model", os.path.abspath(MODEL_PATH), "--output_file", fifo_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail="piper binary not found") from exc

        assert piper_proc.stdin is not None  # for type checkers
        try:
            sanitized_text = text.translate(_PIPER_TEXT_TRANSLATION)
            payload = sanitized_text.encode("utf-8")
            piper_proc.stdin.write(payload)
            piper_proc.stdin.flush()
            piper_proc.stdin.close()
        except BrokenPipeError as exc:
            piper_proc.kill()
            raise HTTPException(status_code=500, detail="Failed to feed TTS pipeline") from exc

        try:
            ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    fifo_path,
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "64k",
                    "-f",
                    "ogg",
                    "-",
                ],
                stdout=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            piper_proc.kill()
            raise HTTPException(status_code=500, detail="ffmpeg binary not found") from exc

        assert ffmpeg_proc.stdout is not None
        try:
            yield ffmpeg_proc
        finally:
            ffmpeg_proc.stdout.close()
            ffmpeg_proc.wait()
            piper_proc.wait()
    finally:
        try:
            os.remove(fifo_path)
        except FileNotFoundError:
            pass


def stream_tts_chunks(text: str) -> Iterable[bytes]:
    with _piper_pipeline(text) as ffmpeg_proc:
        assert ffmpeg_proc.stdout is not None
        while True:
            data = ffmpeg_proc.stdout.read(CHUNK_SIZE)
            if not data:
                break
            yield data


def synthesize_tts_bytes(text: str) -> bytes:
    audio = bytearray()
    for chunk in stream_tts_chunks(text):
        audio.extend(chunk)
    return bytes(audio)


@router.get("/tts")
def tts_stream(text: str = Query(..., min_length=1)):
    return StreamingResponse(stream_tts_chunks(text), media_type="audio/ogg")
