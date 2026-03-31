from __future__ import annotations

from functools import lru_cache

from openai import OpenAI
from pinecone import Pinecone

from app.config import require_openai_api_key, require_pinecone_config, settings


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    require_openai_api_key()
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    require_pinecone_config()
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache(maxsize=1)
def get_pinecone_index():
    require_pinecone_config()
    return get_pinecone_client().Index(host=settings.pinecone_index_host)
