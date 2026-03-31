from __future__ import annotations

import csv
from pathlib import Path



def load_source_register(register_path: Path | None) -> dict[str, dict]:
    if not register_path or not register_path.exists():
        return {}

    rows: dict[str, dict] = {}
    with register_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("source_key") or "").strip()
            if key:
                rows[key] = row
    return rows
