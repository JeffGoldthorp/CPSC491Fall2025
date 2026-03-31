from __future__ import annotations

import os
import time
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


def _clean_text(text: str, max_chars: int = 12000) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: List[str], batch_size: int = 64, max_retries: int = 5) -> List[list[float]]:
    cleaned: List[str] = []
    for text in texts:
        value = _clean_text(text)
        if value:
            cleaned.append(value)

    if not cleaned:
        return []

    vectors: List[list[float]] = []

    for start in range(0, len(cleaned), batch_size):
        batch = cleaned[start:start + batch_size]

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=batch,
                    dimensions=OPENAI_EMBEDDING_DIMENSIONS,
                )
                vectors.extend([item.embedding for item in response.data])
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                sleep_seconds = min(2 ** attempt, 20)
                print(f"[WARN] Embedding batch retry {attempt + 1}/{max_retries} after error: {exc}")
                time.sleep(sleep_seconds)

    return vectors