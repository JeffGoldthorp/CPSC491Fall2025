from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone

from app.services.embeddings import embed_text

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "")
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "psap-911-curated")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "8"))
DEFAULT_MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))

ALL_NAMESPACES = ["psap-911-curated", "psap-911-web"]


def _get_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set in .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_HOST:
        return pc.Index(host=PINECONE_INDEX_HOST)
    if PINECONE_INDEX:
        return pc.Index(PINECONE_INDEX)

    raise RuntimeError("Set either PINECONE_INDEX_HOST or PINECONE_INDEX in .env")


def _resolve_namespaces(namespace: str | None) -> list[str]:
    if not namespace:
        return [DEFAULT_NAMESPACE]

    if namespace == "all":
        return ALL_NAMESPACES

    if "," in namespace:
        values = [item.strip() for item in namespace.split(",") if item.strip()]
        return values or [DEFAULT_NAMESPACE]

    return [namespace]


def _match_to_dict(match: Any, namespace: str) -> dict:
    if isinstance(match, dict):
        metadata = match.get("metadata", {}) or {}
        score = match.get("score")
    else:
        metadata = getattr(match, "metadata", {}) or {}
        score = getattr(match, "score", None)

    metadata = dict(metadata)
    metadata["namespace"] = namespace

    return {
        "score": float(score) if score is not None else 0.0,
        "metadata": metadata,
    }


def _dedupe_key(item: dict) -> tuple:
    md = item["metadata"]
    return (
        md.get("namespace"),
        md.get("source_key") or md.get("source_id") or md.get("source") or md.get("location"),
        md.get("page"),
        md.get("chunk_index"),
        (md.get("text") or md.get("content") or "")[:120],
    )


def retrieve(
    question: str,
    namespace: str | None = None,
    filters: dict | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict]:
    index = _get_index()
    q_emb = embed_text(question)
    namespaces = _resolve_namespaces(namespace)

    all_results: list[dict] = []

    per_namespace_top_k = max(top_k * 2, 8) if len(namespaces) > 1 else max(top_k, 8)

    for ns in namespaces:
        query_kwargs = {
            "vector": q_emb,
            "top_k": per_namespace_top_k,
            "include_metadata": True,
            "namespace": ns,
        }
        if filters:
            query_kwargs["filter"] = filters

        results = index.query(**query_kwargs)
        matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", []) or []

        for match in matches:
            item = _match_to_dict(match, ns)
            md = item["metadata"]

            allowed = md.get("allowed_for_answers")
            if allowed is False or str(allowed).lower() == "false":
                continue

            if item["score"] < min_score:
                continue

            all_results.append(item)

    all_results.sort(key=lambda x: x["score"], reverse=True)

    deduped: list[dict] = []
    seen = set()

    for item in all_results:
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[:top_k]