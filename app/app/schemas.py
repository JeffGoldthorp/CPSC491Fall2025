from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


ChatMode = Literal["base", "finetuned", "rag", "hybrid"]


class ChatRequest(BaseModel):
    question: str = Field(min_length=3)
    namespace: str | None = None
    filters: dict[str, Any] | None = None
    mode: ChatMode = "rag"
    chat_model_override: str | None = None


class Citation(BaseModel):
    rank: int
    source_id: str
    title: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None
    url: str | None = None
    authority_level: str | None = None
    allowed_for_answers: bool | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    mode: ChatMode
    chat_model_used: str


class HealthResponse(BaseModel):
    status: str
