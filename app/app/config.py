from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    return value or ""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = _get_env("OPENAI_API_KEY")
    openai_chat_model: str = _get_env("OPENAI_CHAT_MODEL", "gpt-5-mini")
    openai_embedding_model: str = _get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_embedding_dimensions: int = int(_get_env("OPENAI_EMBEDDING_DIMENSIONS", "1536"))
    openai_fine_tune_base_model: str = _get_env("OPENAI_FINE_TUNE_BASE_MODEL", "gpt-4o-mini")
    openai_fine_tune_suffix: str = _get_env("OPENAI_FINE_TUNE_SUFFIX", "psap-911-cyber-sft")
    openai_finetuned_model: str = _get_env("OPENAI_FINETUNED_MODEL")
    pinecone_api_key: str = _get_env("PINECONE_API_KEY")
    pinecone_index_host: str = _get_env("PINECONE_INDEX_HOST")
    pinecone_namespace: str = _get_env("PINECONE_NAMESPACE", "public_authoritative")
    pinecone_top_k: int = int(_get_env("PINECONE_TOP_K", "8"))
    system_name: str = _get_env("SYSTEM_NAME", "PSAP Cyber Risk Assistant")
    allow_web_enrichment: bool = _get_env("ALLOW_WEB_ENRICHMENT", "false").lower() == "true"
    max_context_chunks: int = int(_get_env("MAX_CONTEXT_CHUNKS", "8"))
    min_source_score: float = float(_get_env("MIN_SOURCE_SCORE", "0.35"))


settings = Settings()


def require_openai_api_key() -> None:
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")



def require_pinecone_config() -> None:
    missing = []
    if not settings.pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if not settings.pinecone_index_host:
        missing.append("PINECONE_INDEX_HOST")
    if missing:
        raise RuntimeError(f"Missing Pinecone configuration: {', '.join(missing)}")
