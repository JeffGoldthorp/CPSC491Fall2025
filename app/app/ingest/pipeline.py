from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Any

from dotenv import load_dotenv
from pinecone import Pinecone

from app.ingest.loaders import iter_sources
from app.services.embeddings import embed_texts

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "public_authoritative")

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 100


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
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


def load_source_register(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}

    rows: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_key = (row.get("source_key") or "").strip()
            if source_key:
                rows[source_key] = row
    return rows


def _stable_chunk_id(source_key: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_key}|{chunk_index}|{text[:300]}".encode("utf-8")).hexdigest()[:20]
    return f"{source_key}-{chunk_index}-{digest}"


def _iter_text_units(source: dict) -> Iterable[dict]:
    source_key = source.get("source_key", "unknown_source")
    title = source.get("title", source_key)
    doc_type = source.get("doc_type", "text")
    location = source.get("path") or source.get("url") or ""

    if source.get("pages"):
        for page in source["pages"]:
            page_num = page.get("page")
            text = page.get("text", "")
            if text and text.strip():
                yield {
                    "source_key": source_key,
                    "title": title,
                    "doc_type": doc_type,
                    "location": location,
                    "page": page_num,
                    "text": text,
                }
    else:
        text = source.get("text", "")
        if text and text.strip():
            yield {
                "source_key": source_key,
                "title": title,
                "doc_type": doc_type,
                "location": location,
                "page": None,
                "text": text,
            }


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
            continue

        if isinstance(value, list):
            string_list = [str(v) for v in value if v is not None]
            if string_list:
                clean[key] = string_list
            continue

        clean[key] = str(value)

    return clean


def build_records(input_dir: Path, source_register: Dict[str, dict], max_records: int | None = None) -> List[dict]:
    records: List[dict] = []
    seen_hashes = set()

    for source in iter_sources(input_dir):
        for unit in _iter_text_units(source):
            source_key = unit["source_key"]
            title = unit["title"]
            doc_type = unit["doc_type"]
            location = unit["location"]
            page = unit["page"]

            register_row = source_register.get(source_key, {})
            allowed_for_answers = str(register_row.get("allowed_for_answers", "true")).strip().lower() == "true"
            authority_level = (register_row.get("authority_level") or "reference").strip()
            source_type = (register_row.get("source_type") or doc_type).strip()
            jurisdiction = (register_row.get("jurisdiction") or "").strip()
            freshness_reviewed_on = (register_row.get("freshness_reviewed_on") or "").strip()

            chunks = chunk_text(unit["text"])
            for chunk_index, chunk in enumerate(chunks, start=1):
                text_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
                if text_hash in seen_hashes:
                    continue
                seen_hashes.add(text_hash)

                record_id = _stable_chunk_id(source_key, chunk_index, chunk)

                metadata = sanitize_metadata(
                    {
                        "source_key": source_key,
                        "title": title,
                        "doc_type": doc_type,
                        "source_type": source_type,
                        "location": location,
                        "source": location,
                        "page": page,
                        "chunk_index": chunk_index,
                        "authority_level": authority_level,
                        "allowed_for_answers": allowed_for_answers,
                        "jurisdiction": jurisdiction,
                        "freshness_reviewed_on": freshness_reviewed_on,
                        "text": chunk[:1000],
                    }
                )

                records.append(
                    {
                        "id": record_id,
                        "text": chunk,
                        "metadata": metadata,
                    }
                )

                if max_records and len(records) >= max_records:
                    return records

    return records


def write_jsonl(records: List[dict], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_pinecone_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set in .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_HOST:
        return pc.Index(host=PINECONE_INDEX_HOST)

    if PINECONE_INDEX:
        return pc.Index(PINECONE_INDEX)

    raise RuntimeError("Set either PINECONE_INDEX_HOST or PINECONE_INDEX in .env")


def upsert_records(records: List[dict]) -> None:
    index = get_pinecone_index()

    for start in range(0, len(records), EMBED_BATCH_SIZE):
        batch = records[start:start + EMBED_BATCH_SIZE]
        texts = [record["text"] for record in batch]
        embeddings = embed_texts(texts, batch_size=EMBED_BATCH_SIZE)

        vectors = []
        for record, embedding in zip(batch, embeddings):
            vectors.append(
                {
                    "id": record["id"],
                    "values": embedding,
                    "metadata": record["metadata"],
                }
            )

        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            sub_batch = vectors[i:i + UPSERT_BATCH_SIZE]
            index.upsert(vectors=sub_batch, namespace=PINECONE_NAMESPACE)

        print(f"Embedded/upserted {min(start + EMBED_BATCH_SIZE, len(records))}/{len(records)} chunks")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="sources")
    parser.add_argument("--output-jsonl", default="data/staged/corpus.jsonl")
    parser.add_argument("--source-register", default="data/source_register.csv")
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_jsonl = Path(args.output_jsonl)
    source_register_path = Path(args.source_register)

    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    source_register = load_source_register(source_register_path)
    records = build_records(input_dir, source_register, max_records=args.max_records)

    if not records:
        raise RuntimeError("No records were built. Check input sources and source register.")

    write_jsonl(records, output_jsonl)
    print(f"Wrote {len(records)} normalized chunks to {output_jsonl}")

    if args.upsert:
        upsert_records(records)
        print(f"Finished upserting to Pinecone namespace '{PINECONE_NAMESPACE}'")


if __name__ == "__main__":
    main()