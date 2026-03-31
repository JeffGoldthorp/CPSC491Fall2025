from __future__ import annotations

from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.chat_service import answer_question

app = FastAPI(
    title="PSAP 911 Cyber Risk Assistant API",
    description="Supports base, fine-tuned, RAG, and hybrid answer modes for 911 call center cybersecurity guidance.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return answer_question(
        question=payload.question,
        namespace=payload.namespace,
        filters=payload.filters,
        mode=payload.mode,
        model_override=payload.model_override,
    )
