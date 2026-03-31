from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a cybersecurity assistant for 911 call centers and PSAPs. "
    "Provide practical, grounded, concise guidance. "
    "Do not invent requirements, laws, or standards. "
    "When evidence is insufficient, say so clearly and recommend a safe next step."
)



def build_record(example: dict) -> dict:
    instruction = example["instruction"].strip()
    response = example["response"].strip()
    context = example.get("context", "").strip()
    user_content = instruction if not context else f"Context:\n{context}\n\nQuestion:\n{instruction}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-dir", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    records = []
    for path in sorted(examples_dir.rglob("*.json")):
        example = json.loads(path.read_text(encoding="utf-8"))
        records.append(build_record(example))

    if len(records) < 5:
        raise RuntimeError("Add at least 5 approved examples before splitting train/val datasets.")

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

    print(f"Wrote {len(train_records)} train examples to {args.train_output}")
    print(f"Wrote {len(val_records)} validation examples to {args.val_output}")
