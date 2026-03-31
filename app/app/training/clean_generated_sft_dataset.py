from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.I)
PHONE_RE = re.compile(r"(?:(?<!\d)(?:\+?1[\s.-]*)?(?:\(?\d{3}\)?[\s.-]*)\d{3}[\s.-]*\d{4}(?!\d))")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|org|gov|edu|net|mil|io)\b", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)

BLOCKLIST_PHRASES = [
    "contact us",
    "for more information",
    "call us",
    "email us",
    "please contact",
    "attorney advertising",
    "all rights reserved",
    "privacy policy",
    "terms of use",
    "subscribe",
    "follow us",
    "advertisement",
]


def sanitize_text(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = URL_RE.sub("[REDACTED_URL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = IPV4_RE.sub("[REDACTED_IP]", text)
    text = DOMAIN_RE.sub("[REDACTED_DOMAIN]", text)
    text = DOI_RE.sub("[REDACTED_DOI]", text)
    text = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+,?\s+(Secretary|Director|Attorney|Partner)\b", "[REDACTED_NAME]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def record_fingerprint(record: dict) -> str:
    messages = record.get("messages", [])
    joined = "||".join(f"{m.get('role','')}::{m.get('content','')}" for m in messages)
    return hashlib.sha256(joined.lower().encode("utf-8")).hexdigest()


def clean_record(record: dict) -> dict | None:
    messages = record.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 3:
        return None
    cleaned_messages = []
    for msg in messages:
        role = msg.get("role")
        content = sanitize_text(str(msg.get("content", "")))
        if not content:
            return None
        cleaned_messages.append({"role": role, "content": content})
    joined = " ".join(m["content"].lower() for m in cleaned_messages)
    if any(phrase in joined for phrase in BLOCKLIST_PHRASES):
        return None
    if any(token in joined for token in ["[redacted_email]", "[redacted_phone]", "[redacted_url]"]):
        return None
    assistant = next((m["content"] for m in cleaned_messages if m["role"] == "assistant"), "")
    if len(assistant) < 80:
        return None
    return {"messages": cleaned_messages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    fingerprints: set[str] = set()
    with in_path.open("r", encoding="utf-8") as inp, out_path.open("w", encoding="utf-8") as out:
        for line in inp:
            total += 1
            record = json.loads(line)
            cleaned = clean_record(record)
            if not cleaned:
                continue
            fp = record_fingerprint(cleaned)
            if fp in fingerprints:
                continue
            fingerprints.add(fp)
            out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Cleaned {in_path} -> {out_path}")
    print(f"Kept {kept} of {total} records")


if __name__ == "__main__":
    main()
