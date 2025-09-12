from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

# Import the agent from the sibling package `model`
from ..newModel.talk import talkToAgent


router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    reply: str
    history: List[Dict[str, str]]
    context: Optional[Dict[str, Any]] = None


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # Simplified flow: ignore context and do not pass backend-side history
    reply = talkToAgent(req.message)

    history = req.history[:] if req.history else []
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})

    return {"reply": reply, "history": history, "context": None}
