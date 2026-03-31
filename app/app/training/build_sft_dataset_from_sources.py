from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

from app.clients import get_openai_client
from app.config import settings
from app.ingest.chunking import normalize_text, split_into_chunks
from app.ingest.loaders import iter_sources
from app.ingest.source_register import load_source_register

SYSTEM_PROMPT = (
    "You are a cybersecurity assistant for 911 call centers and PSAPs. "
    "Provide practical, grounded, concise guidance. "
    "Do not invent requirements, laws, or standards. "
    "Do not include phone numbers, email addresses, URLs, personal contact details, or promotional language. "
    "When evidence is insufficient, say so clearly and recommend a safe next step."
)

SYNTHESIS_PROMPT = (
    "You are generating supervised fine-tuning training examples for a PSAP cybersecurity assistant. "
    "Use ONLY the excerpt that is provided. Do not invent standards, product names, numbers, or facts not present in the excerpt. "
    "Do not include names of private individuals, phone numbers, email addresses, URLs, web domains, legal advertisements, marketing copy, 'contact us' language, or vendor promotion. "
    "If the excerpt is mostly a table of contents, bibliography, legal disclaimer, contact block, footer, index, or marketing material, return []. "
    "Return ONLY a JSON array. Each item must have exactly these keys: instruction, context, response. "
    "Make the instruction realistic for a 911 call center leader, analyst, or technical operator. "
    "The context should be a short neutral summary of the excerpt, not copied boilerplate. "
    "The response should be practical and concise, usually 100 to 180 words."
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.I)
PHONE_RE = re.compile(
    r"(?:(?<!\d)(?:\+?1[\s.-]*)?(?:\(?\d{3}\)?[\s.-]*)\d{3}[\s.-]*\d{4}(?!\d))"
)
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|org|gov|edu|net|mil|io)\b", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
PAGE_NUM_RE = re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.I)
SHORT_NUMERIC_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s*$")
TOC_LEADER_RE = re.compile(r"\.{4,}|_{4,}")

BLOCKLIST_PHRASES = [
    "contact us",
    "for more information",
    "call us",
    "email us",
    "please contact",
    "attorney advertising",
    "all rights reserved",
    "this publication is available free of charge from",
    "table of contents",
    "copyright",
    "privacy policy",
    "terms of use",
    "subscribe",
    "follow us",
    "advertisement",
]

GOOD_AUTHORITY_LEVELS = {"primary", "secondary", "local_policy"}


def _apply_register(source: dict, source_register: dict[str, dict]) -> dict:
    key = source.get("source_key") or source.get("source_id")
    row = source_register.get(str(key), {})
    source["authority_level"] = row.get("authority_level", "reference")
    source["allowed_for_answers"] = str(row.get("allowed_for_answers", "true")).lower() == "true"
    source["source_type"] = row.get("source_type", source.get("doc_type", "unknown"))
    source["jurisdiction"] = row.get("jurisdiction", "general")
    source["freshness_reviewed_on"] = row.get("freshness_reviewed_on")
    return source


def _sanitize_text(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = URL_RE.sub("[REDACTED_URL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = IPV4_RE.sub("[REDACTED_IP]", text)
    text = DOMAIN_RE.sub("[REDACTED_DOMAIN]", text)
    text = DOI_RE.sub("[REDACTED_DOI]", text)
    text = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+,?\s+(Secretary|Director|Attorney|Partner)\b", "[REDACTED_NAME]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if PAGE_NUM_RE.match(stripped):
        return True
    if SHORT_NUMERIC_LINE_RE.match(stripped):
        return True
    if TOC_LEADER_RE.search(stripped):
        return True
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return True
    if EMAIL_RE.search(stripped) or PHONE_RE.search(stripped):
        return True
    if any(phrase in lowered for phrase in BLOCKLIST_PHRASES):
        return True
    if len(stripped) < 3:
        return True
    return False


def _clean_excerpt(text: str) -> str:
    text = normalize_text(text)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if _line_is_noise(line):
            continue
        cleaned_lines.append(line.strip())
    text = "\n".join(cleaned_lines)
    text = _sanitize_text(text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _looks_low_quality(text: str) -> bool:
    if len(text) < 500:
        return True
    lowered = text.lower()
    bad_hits = sum(1 for phrase in BLOCKLIST_PHRASES if phrase in lowered)
    short_lines = sum(1 for line in text.splitlines() if 0 < len(line.strip()) < 35)
    total_lines = max(1, len(text.splitlines()))
    alpha_chars = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_chars / max(1, len(text))
    if bad_hits >= 2:
        return True
    if short_lines / total_lines > 0.55:
        return True
    if alpha_ratio < 0.55:
        return True
    return False


def _iter_excerpt_units(
    input_dir: Path,
    source_register: dict[str, dict],
    chunk_size: int,
    max_chunks_per_source: int,
    allowed_authorities: set[str],
) -> Iterable[dict]:
    for source in iter_sources(input_dir):
        source = _apply_register(source, source_register)
        if not source.get("allowed_for_answers", True):
            continue
        if allowed_authorities and source.get("authority_level", "reference") not in allowed_authorities:
            continue

        yielded = 0
        if source.get("doc_type") == "pdf":
            for page in source.get("pages", []):
                text = _clean_excerpt(page.get("text", ""))
                if _looks_low_quality(text):
                    continue
                for chunk_index, chunk in enumerate(split_into_chunks(text, chunk_size=chunk_size), start=1):
                    chunk = _clean_excerpt(chunk)
                    if _looks_low_quality(chunk):
                        continue
                    yield {
                        "source_key": source["source_key"],
                        "title": source["title"],
                        "doc_type": source["doc_type"],
                        "authority_level": source.get("authority_level"),
                        "page_start": page["page"],
                        "page_end": page["page"],
                        "section": f"page-{page['page']}-chunk-{chunk_index}",
                        "excerpt": chunk,
                    }
                    yielded += 1
                    if yielded >= max_chunks_per_source:
                        break
                if yielded >= max_chunks_per_source:
                    break
        else:
            text = _clean_excerpt(source.get("text", ""))
            if _looks_low_quality(text):
                continue
            for chunk_index, chunk in enumerate(split_into_chunks(text, chunk_size=chunk_size), start=1):
                chunk = _clean_excerpt(chunk)
                if _looks_low_quality(chunk):
                    continue
                yield {
                    "source_key": source["source_key"],
                    "title": source["title"],
                    "doc_type": source["doc_type"],
                    "authority_level": source.get("authority_level"),
                    "page_start": None,
                    "page_end": None,
                    "section": f"chunk-{chunk_index}",
                    "excerpt": chunk,
                }
                yielded += 1
                if yielded >= max_chunks_per_source:
                    break


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


def _synthesize_examples(unit: dict, synthesis_model: str, examples_per_excerpt: int) -> list[dict]:
    client = get_openai_client()
    prompt = (
        f"Source title: {unit['title']}\n"
        f"Source key: {unit['source_key']}\n"
        f"Authority level: {unit.get('authority_level', 'unknown')}\n"
        f"Section: {unit.get('section')}\n\n"
        f"Excerpt:\n{unit['excerpt']}\n\n"
        f"Create up to {examples_per_excerpt} examples. Return only JSON."
    )
    response = client.responses.create(
        model=synthesis_model,
        input=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw_examples = _extract_json_array(response.output_text)
    cleaned: list[dict] = []
    for item in raw_examples:
        instruction = _sanitize_text(str(item.get("instruction", "")).strip())
        context = _sanitize_text(str(item.get("context", "")).strip())
        answer = _sanitize_text(str(item.get("response", "")).strip())
        joined = f"{instruction} {context} {answer}".lower()
        if len(instruction) < 12 or len(answer) < 80:
            continue
        if any(token in joined for token in ["[redacted_email]", "[redacted_phone]", "[redacted_url]", "contact us"]):
            continue
        if any(phrase in joined for phrase in BLOCKLIST_PHRASES):
            continue
        cleaned.append(
            {
                "instruction": instruction,
                "context": context,
                "response": answer,
                "source_key": unit["source_key"],
                "source_title": unit["title"],
                "section": unit.get("section"),
                "page_start": unit.get("page_start"),
                "page_end": unit.get("page_end"),
                "authority_level": unit.get("authority_level"),
            }
        )
    return cleaned


def _build_record(example: dict) -> dict:
    instruction = example["instruction"].strip()
    response = example["response"].strip()
    context = example.get("context", "").strip()
    user_content = instruction if not context else f"Context:\n{context}\n\nQuestion:\n{instruction}"
    user_content = _sanitize_text(user_content)
    response = _sanitize_text(response)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


def _fingerprint(example: dict) -> str:
    material = "||".join(
        [
            example.get("instruction", "").strip().lower(),
            example.get("context", "").strip().lower(),
            example.get("response", "").strip().lower(),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="sources")
    parser.add_argument("--source-register", default="data/source_register.csv")
    parser.add_argument("--candidate-output", default="data/staged/sft_candidates.jsonl")
    parser.add_argument("--train-output", default="data/staged/sft_train.jsonl")
    parser.add_argument("--val-output", default="data/staged/sft_val.jsonl")
    parser.add_argument("--synthesis-model", default=settings.openai_chat_model)
    parser.add_argument("--target-examples", type=int, default=120)
    parser.add_argument("--examples-per-excerpt", type=int, default=2)
    parser.add_argument("--max-chunks-per-source", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1400)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allowed-authorities",
        default=",".join(sorted(GOOD_AUTHORITY_LEVELS)),
        help="Comma-separated authority levels allowed for SFT generation.",
    )
    args = parser.parse_args()

    allowed_authorities = {item.strip() for item in args.allowed_authorities.split(",") if item.strip()}
    source_register = load_source_register(Path(args.source_register) if args.source_register else None)
    units = list(
        _iter_excerpt_units(
            Path(args.input_dir),
            source_register,
            args.chunk_size,
            args.max_chunks_per_source,
            allowed_authorities,
        )
    )
    if not units:
        raise RuntimeError("No usable source excerpts found. Check the input folder and source register.")

    random.Random(args.seed).shuffle(units)

    candidate_examples: list[dict] = []
    fingerprints: set[str] = set()
    for unit in units:
        generated = _synthesize_examples(unit, args.synthesis_model, args.examples_per_excerpt)
        accepted_this_round = 0
        for item in generated:
            fp = _fingerprint(item)
            if fp in fingerprints:
                continue
            fingerprints.add(fp)
            candidate_examples.append(item)
            accepted_this_round += 1
        print(
            f"Generated {accepted_this_round} clean candidate examples from {unit['source_key']} {unit.get('section')} "
            f"(running total={len(candidate_examples)})"
        )
        if len(candidate_examples) >= args.target_examples:
            break

    if len(candidate_examples) < 10:
        raise RuntimeError("Too few clean candidate examples were generated. Increase target size or review source quality.")

    candidate_path = Path(args.candidate_output)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with candidate_path.open("w", encoding="utf-8") as out:
        for item in candidate_examples:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    records = [_build_record(example) for example in candidate_examples]
    random.Random(args.seed).shuffle(records)
    split_index = max(1, int(len(records) * (1 - args.val_ratio)))
    train_records = records[:split_index]
    val_records = records[split_index:]

    for output_path, output_records in [
        (Path(args.train_output), train_records),
        (Path(args.val_output), val_records),
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as out:
            for record in output_records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(candidate_examples)} clean SFT candidates to {candidate_path}")
    print(f"Wrote {len(train_records)} train examples to {args.train_output}")
    print(f"Wrote {len(val_records)} validation examples to {args.val_output}")


if __name__ == "__main__":
    main()
