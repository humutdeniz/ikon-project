from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Context(BaseModel):
    # Conversation turns for UI display only; LLM must not rely on these
    turns: List[Turn] = Field(default_factory=list)

    # Shared slots (persisted context across turns)
    role: Optional[Literal["employee", "courier", "meeting", "unknown"]] = None
    employeeName: Optional[str] = None
    password: Optional[str] = None
    company: Optional[str] = None
    recipient: Optional[str] = None
    host: Optional[str] = None
    guest: Optional[str] = None
    time: Optional[str] = None

    # Safety and action
    threat: Optional[bool] = None
    decision: Optional[Literal["allow", "deny", "lock", "ask"]] = None
    doorAction: Optional[Literal["unlock", "lock"]] = None
    note: Optional[str] = None


class StepResult(BaseModel):
    reply: str
    context: Context

