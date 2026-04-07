from __future__ import annotations

from typing import Iterable

from app.clients import get_openai_client
from app.config import settings
from app.schemas import ChatMode, ChatResponse, Citation
from app.services.retriever import retrieve
from app.services.web_enrichment import external_search, fetch_full_text

SYSTEM_PROMPT = (
    "You are a cybersecurity assistant for 911 call centers and PSAPs. "
    "Be practical, specific, and calm. "
    "Never invent requirements, laws, or standards. "
    "When evidence is insufficient, say so clearly and recommend a safe next step."
)

RAG_PROMPT = (
    "Use the supplied excerpts as evidence. "
    "Prefer the indexed corpus first. "
    "If live web fallback excerpts are provided, treat them as secondary evidence. "
    "Cite claims inline using [1], [2], and so on. "
    "Do not add any extra sections such as 'What the indexed evidence shows' or 'What the corpus doesn\'t explicitly list'. "
    "Answer only from the evidence and do not speculate beyond it. "
    "If evidence is partial, say what is known and what is uncertain."
)

FT_PROMPT = (
    "Stay within PSAP, 911 operations, cyber risk, emergency communications, and public-safety policy. "
    "If asked outside scope, state the scope limit clearly. "
    "Do not invent analysis sections beyond the answer itself."
)

WEB_FALLBACK_MIN_RESULTS = 3
WEB_FALLBACK_MIN_SCORE = 0.35


def _md_value(md: dict, *keys, default=None):
    for key in keys:
        value = md.get(key)
        if value is not None and value != "":
            return value
    return default


def _format_context(results: list[dict]) -> str:
    lines = []
    for i, item in enumerate(results, start=1):
        md = item.get("metadata", {})
        lines.append(
            f"[{i}] title={_md_value(md, 'title', default='Untitled')} | "
            f"source_id={_md_value(md, 'source_id', 'source_key', 'source', 'location', default='unknown_source')} | "
            f"namespace={_md_value(md, 'namespace', default='unknown')} | "
            f"authority_level={_md_value(md, 'authority_level', default='unknown')} | "
            f"page={_md_value(md, 'page_start', 'page', default='n/a')}\n"
            f"content={_md_value(md, 'content', 'text', default='')}"
        )
    return "\n\n".join(lines)


def _format_web_context(web_docs: list[dict], start_rank: int) -> str:
    lines = []
    for offset, doc in enumerate(web_docs, start=0):
        rank = start_rank + offset
        lines.append(
            f"[{rank}] title={doc.get('title', 'External Source')} | "
            f"source_id={doc.get('url', '')} | namespace=live-web | authority_level=external_live | page=n/a\n"
            f"content={doc.get('content', '') or doc.get('snippet', '')}"
        )
    return "\n\n".join(lines)


def _to_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _citations_from_results(results: Iterable[dict]) -> list[Citation]:
    citations = []

    for rank, item in enumerate(results, start=1):
        md = item.get("metadata", {})
        citations.append(
            Citation(
                rank=rank,
                source_id=str(_md_value(md, "source_id", "source_key", "source", "location", default="")),
                title=str(_md_value(md, "title", default="Untitled")),
                section=str(_md_value(md, "section", "chunk_index")) if _md_value(md, "section", "chunk_index") is not None else None,
                page_start=_to_int_or_none(_md_value(md, "page_start", "page")),
                page_end=_to_int_or_none(_md_value(md, "page_end", "page")),
                score=_to_float_or_none(item.get("score")),
                url=_md_value(md, "url", "source"),
                authority_level=_md_value(md, "authority_level"),
                allowed_for_answers=md.get("allowed_for_answers"),
            )
        )

    return citations


def _citations_from_web_docs(web_docs: list[dict], start_rank: int) -> list[Citation]:
    citations = []
    for idx, doc in enumerate(web_docs, start=start_rank):
        citations.append(
            Citation(
                rank=idx,
                source_id=doc.get("url", ""),
                title=doc.get("title", "External Source"),
                section=None,
                page_start=None,
                page_end=None,
                score=None,
                url=doc.get("url"),
                authority_level="external_live",
                allowed_for_answers=True,
            )
        )
    return citations


def _chat(model: str, messages: list[dict]) -> str:
    client = get_openai_client()
    response = client.responses.create(model=model, input=messages)
    return response.output_text.strip()


def _resolve_model(mode: ChatMode, model_override: str | None) -> str:
    if model_override:
        return model_override
    if mode in {"finetuned", "hybrid"}:
        return settings.openai_finetuned_model or settings.openai_chat_model
    return settings.openai_chat_model


def _needs_web_fallback(results: list[dict]) -> bool:
    if not results:
        return True
    top_score = max(float(item.get("score") or 0.0) for item in results)
    return len(results) < WEB_FALLBACK_MIN_RESULTS or top_score < WEB_FALLBACK_MIN_SCORE


def answer_question(
    question: str,
    namespace: str | None = None,
    filters: dict | None = None,
    mode: ChatMode = "rag",
    model_override: str | None = None,
    allow_web_fallback: bool = False,
) -> ChatResponse:
    model = _resolve_model(mode, model_override)

    if mode == "base":
        answer = _chat(
            model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        return ChatResponse(answer=answer, citations=[], mode=mode, chat_model_used=model)

    if mode == "finetuned":
        answer = _chat(
            model,
            [
                {"role": "system", "content": f"{SYSTEM_PROMPT} {FT_PROMPT}"},
                {"role": "user", "content": question},
            ],
        )
        return ChatResponse(answer=answer, citations=[], mode=mode, chat_model_used=model)

    results = retrieve(question=question, namespace=namespace, filters=filters)

    web_docs: list[dict] = []
    if allow_web_fallback and _needs_web_fallback(results):
        web_docs = external_search(question, max_results=3)
        for doc in web_docs:
            full = fetch_full_text(doc["url"])
            if full:
                doc["content"] = full

    if not results and not web_docs:
        return ChatResponse(
            answer="I could not find enough trusted source material for that question in the current corpus.",
            citations=[],
            mode=mode,
            chat_model_used=model,
        )

    corpus_context = _format_context(results) if results else ""
    web_context = _format_web_context(web_docs, start_rank=len(results) + 1) if web_docs else ""

    prompt = (
        f"Question:\n{question}\n\n"
        f"Indexed corpus excerpts:\n{corpus_context}\n\n"
        f"Live web fallback excerpts:\n{web_context}\n\n"
        "Write a grounded answer with inline citations like [1] and [2]. "
        "Prefer indexed corpus evidence first. "
        "Use live web fallback only if the indexed corpus is insufficient. "
        "If evidence is partial, say what is known and what is uncertain."
    )

    if mode == "rag":
        answer = _chat(
            model,
            [
                {"role": "system", "content": f"{SYSTEM_PROMPT} {RAG_PROMPT}"},
                {"role": "user", "content": prompt},
            ],
        )
    else:
        answer = _chat(
            model,
            [
                {"role": "system", "content": f"{SYSTEM_PROMPT} {FT_PROMPT} {RAG_PROMPT}"},
                {"role": "user", "content": prompt},
            ],
        )

    citations = _citations_from_results(results)
    if web_docs:
        citations.extend(_citations_from_web_docs(web_docs, start_rank=len(citations) + 1))

    return ChatResponse(
        answer=answer,
        citations=citations,
        mode=mode,
        chat_model_used=model,
    )
