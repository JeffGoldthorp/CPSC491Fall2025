from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import fitz

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "psap-911-curated")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
EMBED_BATCH = 64
UPSERT_BATCH = 100


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract_pdf_text(path: Path) -> List[dict]:
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


def embed_texts(texts: List[str]) -> List[list[float]]:
    all_vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        resp = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
            dimensions=OPENAI_EMBEDDING_DIMENSIONS,
        )
        all_vectors.extend([item.embedding for item in resp.data])
    return all_vectors


def make_id(source_key: str, page: int, chunk_index: int, text: str) -> str:
    h = hashlib.sha1(f"{source_key}|{page}|{chunk_index}|{text[:300]}".encode("utf-8")).hexdigest()[:20]
    return f"{source_key}-{page}-{chunk_index}-{h}"


def ingest_folder(input_dir: str):
    records = []

    for path in sorted(Path(input_dir).glob("*.pdf")):
        source_key = path.stem.lower().replace(" ", "_")
        pages = extract_pdf_text(path)

        for page_obj in pages:
            page_num = page_obj["page"]
            chunks = chunk_text(page_obj["text"])
            for chunk_index, chunk in enumerate(chunks, start=1):
                records.append({
                    "id": make_id(source_key, page_num, chunk_index, chunk),
                    "text": chunk,
                    "metadata": {
                        "source_key": source_key,
                        "title": path.stem,
                        "source": str(path.name),
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "text": chunk[:1000],
                    },
                })

    print(f"Built {len(records)} chunks")

    for start in range(0, len(records), EMBED_BATCH):
        batch = records[start:start + EMBED_BATCH]
        vectors = embed_texts([r["text"] for r in batch])

        pinecone_vectors = []
        for record, vector in zip(batch, vectors):
            pinecone_vectors.append({
                "id": record["id"],
                "values": vector,
                "metadata": record["metadata"],
            })

        index.upsert(vectors=pinecone_vectors, namespace=PINECONE_NAMESPACE)
        print(f"Upserted {min(start + EMBED_BATCH, len(records))}/{len(records)}")


if __name__ == "__main__":
    ingest_folder("sources_curated")