from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

# Import the agent from the sibling package `model`
from ..model.talk import talkToAgent
from ..agent.runner import step as agent_step


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
    # New structured agent path if context provided
    if req.context is not None:
        res = agent_step(req.message, req.context)
        return {
            "reply": res.reply,
            "history": res.context.turns,
            "context": res.context.dict(),
        }

    # Legacy path using LLM chat history
    history = req.history[:] if req.history else []
    reply = talkToAgent(req.message, history)
    return {"reply": reply, "history": history}
