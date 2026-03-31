from __future__ import annotations

import argparse
import csv
from pathlib import Path

from app.ingest.loaders import discover_source_descriptors

SUGGESTED_AUTHORITY = {
    "pdf": "reference",
    "docx": "reference",
    "md": "reference",
    "txt": "reference",
    "html": "reference",
    "htm": "reference",
    "web": "reference",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="sources")
    parser.add_argument("--output-csv", default="data/staged/source_inventory.csv")
    args = parser.parse_args()

    descriptors = list(discover_source_descriptors(Path(args.input_dir)))
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_key",
                "title",
                "location",
                "doc_type",
                "suggested_authority_level",
                "suggested_allowed_for_answers",
                "suggested_source_type",
                "jurisdiction",
                "notes",
            ],
        )
        writer.writeheader()
        for item in descriptors:
            writer.writerow(
                {
                    "source_key": item["source_key"],
                    "title": item["title"],
                    "location": item.get("location") or item.get("path") or item.get("url"),
                    "doc_type": item["doc_type"],
                    "suggested_authority_level": SUGGESTED_AUTHORITY.get(item["doc_type"], "reference"),
                    "suggested_allowed_for_answers": "true",
                    "suggested_source_type": item["doc_type"],
                    "jurisdiction": "general",
                    "notes": "review manually before production use",
                }
            )

    print(f"Wrote {len(descriptors)} inventory rows to {output_path}")


if __name__ == "__main__":
    main()
