from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "source"


def source_key_for_path(path: Path) -> str:
    return slugify(path.stem)


def source_key_for_url(url: str) -> str:
    parsed = urlparse(url)
    pieces = [parsed.netloc, parsed.path]
    if parsed.query:
        pieces.append(parsed.query)
    return slugify("_".join(piece for piece in pieces if piece))
