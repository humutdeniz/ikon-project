import base64
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

# Import the agent from the sibling package `model`
from ..newModel.talk import talkToAgent
from .tts import synthesize_tts_bytes


router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None
    context: Optional[Dict[str, Any]] = None
    isReset: bool = False


class ChatResponse(BaseModel):
    reply: str
    history: List[Dict[str, str]]
    context: Optional[Dict[str, Any]] = None
    historyCleared: bool = False
    audio: Optional[str] = None
    audioMimeType: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    agent_resp = talkToAgent(req.message, req.isReset)
    history: List[Dict[str, str]] = []
    if isinstance(agent_resp, dict):
        reply = str(agent_resp.get("reply", ""))
        print(reply)
        cleared = bool(agent_resp.get("history_was_reset", False))
        history = agent_resp.get("history", [])
    else:
        reply = str(agent_resp)
        cleared = False

    audio_b64: Optional[str] = None
    audio_mime = "audio/ogg"
    if reply.strip():
        try:
            audio_bytes = synthesize_tts_bytes(reply)
        except Exception:
            logger.exception("Failed to synthesize TTS audio")
        else:
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    return {
        "reply": reply,
        "context": None,
        "history": history,
        "historyCleared": cleared,
        "audio": audio_b64,
        "audioMimeType": audio_mime if audio_b64 else None,
    }
