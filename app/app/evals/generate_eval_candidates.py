from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from app.clients import get_openai_client
from app.config import settings
from app.ingest.chunking import normalize_text, split_into_chunks
from app.ingest.loaders import iter_sources
from app.ingest.source_register import load_source_register

PROMPT = (
    "You are creating evaluation examples for a PSAP cybersecurity assistant. "
    "Use only the provided excerpt. Return ONLY a JSON array. "
    "Each array item must contain: question, gold_answer. "
    "The gold answer should be concise, factual, and fully grounded in the excerpt. "
    "If the excerpt is weak, return []."
)


def _apply_register(source: dict, source_register: dict[str, dict]) -> dict:
    key = source.get("source_key") or source.get("source_id")
    row = source_register.get(str(key), {})
    source["authority_level"] = row.get("authority_level", "reference")
    source["allowed_for_answers"] = str(row.get("allowed_for_answers", "true")).lower() == "true"
    return source


def _extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            loaded = json.loads(match.group(0))
            return loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            return []


def _iter_excerpts(input_dir: Path, source_register: dict[str, dict], chunk_size: int, max_chunks_per_source: int):
    for source in iter_sources(input_dir):
        source = _apply_register(source, source_register)
        if not source.get("allowed_for_answers", True):
            continue
        count = 0
        if source.get("doc_type") == "pdf":
            for page in source.get("pages", []):
                for chunk_index, chunk in enumerate(split_into_chunks(normalize_text(page.get("text", "")), chunk_size=chunk_size), start=1):
                    if len(chunk) < 500:
                        continue
                    yield {
                        "source_key": source["source_key"],
                        "source_title": source["title"],
                        "section": f"page-{page['page']}-chunk-{chunk_index}",
                        "excerpt": chunk,
                    }
                    count += 1
                    if count >= max_chunks_per_source:
                        break
                if count >= max_chunks_per_source:
                    break
        else:
            for chunk_index, chunk in enumerate(split_into_chunks(normalize_text(source.get("text", "")), chunk_size=chunk_size), start=1):
                if len(chunk) < 500:
                    continue
                yield {
                    "source_key": source["source_key"],
                    "source_title": source["title"],
                    "section": f"chunk-{chunk_index}",
                    "excerpt": chunk,
                }
                count += 1
                if count >= max_chunks_per_source:
                    break


def _generate_for_excerpt(excerpt: dict, model: str, items_per_excerpt: int) -> list[dict]:
    client = get_openai_client()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source title: {excerpt['source_title']}\n"
                    f"Source key: {excerpt['source_key']}\n"
                    f"Section: {excerpt['section']}\n\n"
                    f"Excerpt:\n{excerpt['excerpt']}\n\n"
                    f"Generate up to {items_per_excerpt} eval items."
                ),
            },
        ],
    )
    loaded = _extract_json_array(response.output_text)
    cleaned = []
    for item in loaded:
        question = str(item.get("question", "")).strip()
        gold_answer = str(item.get("gold_answer", "")).strip()
        if len(question) < 12 or len(gold_answer) < 50:
            continue
        cleaned.append(
            {
                "question": question,
                "gold_answer": gold_answer,
                "filters": None,
                "source_key": excerpt["source_key"],
                "source_title": excerpt["source_title"],
                "section": excerpt["section"],
            }
        )
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="sources")
    parser.add_argument("--source-register", default="data/source_register.csv")
    parser.add_argument("--output-jsonl", default="data/staged/eval_candidates.jsonl")
    parser.add_argument("--model", default=settings.openai_chat_model)
    parser.add_argument("--target-items", type=int, default=30)
    parser.add_argument("--items-per-excerpt", type=int, default=1)
    parser.add_argument("--max-chunks-per-source", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_register = load_source_register(Path(args.source_register) if args.source_register else None)
    excerpts = list(_iter_excerpts(Path(args.input_dir), source_register, args.chunk_size, args.max_chunks_per_source))
    random.Random(args.seed).shuffle(excerpts)

    items: list[dict] = []
    for excerpt in excerpts:
        generated = _generate_for_excerpt(excerpt, args.model, args.items_per_excerpt)
        items.extend(generated)
        print(f"Generated {len(generated)} eval candidates from {excerpt['source_key']} {excerpt['section']}")
        if len(items) >= args.target_items:
            break

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} eval candidates to {output_path}")


if __name__ == "__main__":
    main()
