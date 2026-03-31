from __future__ import annotations

import json
import sys
from pathlib import Path



def validate_jsonl(file_path: str) -> bool:
    path = Path(file_path)
    valid = True
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                json_obj = json.loads(line.strip())
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON on line {line_number}: {exc}")
                valid = False
                continue
            if "messages" not in json_obj or not isinstance(json_obj["messages"], list):
                print(f"Invalid structure on line {line_number}: missing messages array")
                valid = False
                continue
            for message in json_obj["messages"]:
                if "role" not in message or "content" not in message:
                    print(f"Invalid message on line {line_number}: missing role or content")
                    valid = False
    return valid


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.training.validate_jsonl path/to/file.jsonl")
    ok = validate_jsonl(sys.argv[1])
    print("VALID" if ok else "INVALID")
    raise SystemExit(0 if ok else 1)
