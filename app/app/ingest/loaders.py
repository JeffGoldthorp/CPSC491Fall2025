from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import fitz
import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

try:
    from docx import Document
except ImportError:  # pragma: no cover
    Document = None

from app.ingest.source_keys import source_key_for_path, source_key_for_url

URL_MANIFEST_HINTS = ("url", "urls", "link", "links", "web")
TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm"}
URL_HEADERS = {"url", "urls", "link", "links", "website", "web_url"}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.google.com/",
}


def _looks_like_url_manifest(path: Path) -> bool:
    lowered = path.name.lower()
    return any(hint in lowered for hint in URL_MANIFEST_HINTS)


def _iter_urls_from_text(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        candidate = line.strip()
        if candidate.startswith("http://") or candidate.startswith("https://"):
            yield candidate


def _iter_urls_from_json(path: Path) -> Iterable[str]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                yield item
    elif isinstance(content, dict):
        for key in ("urls", "links", "websites"):
            value = content.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.startswith(("http://", "https://")):
                        yield item


def _iter_urls_from_csv(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for header in URL_HEADERS:
                candidate = (row.get(header) or "").strip()
                if candidate.startswith(("http://", "https://")):
                    yield candidate
                    break


def iter_manifest_urls(path: Path) -> Iterable[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        yield from _iter_urls_from_text(path)
    elif suffix == ".json":
        yield from _iter_urls_from_json(path)
    elif suffix == ".csv":
        yield from _iter_urls_from_csv(path)


def discover_source_descriptors(input_dir: Path) -> Iterable[dict]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            yield {
                "source_key": source_key_for_path(path),
                "source_id": path.stem,
                "title": path.stem,
                "doc_type": "pdf",
                "path": str(path),
                "location": str(path),
            }
        elif suffix == ".docx":
            yield {
                "source_key": source_key_for_path(path),
                "source_id": path.stem,
                "title": path.stem,
                "doc_type": "docx",
                "path": str(path),
                "location": str(path),
            }
        elif suffix in TEXT_EXTENSIONS:
            if _looks_like_url_manifest(path):
                for url in iter_manifest_urls(path):
                    yield {
                        "source_key": source_key_for_url(url),
                        "source_id": url,
                        "title": url,
                        "doc_type": "web",
                        "url": url,
                        "location": url,
                    }
            else:
                yield {
                    "source_key": source_key_for_path(path),
                    "source_id": path.stem,
                    "title": path.stem,
                    "doc_type": suffix.replace('.', '') or 'text',
                    "path": str(path),
                    "location": str(path),
                }
        elif suffix in {".json", ".csv"} and _looks_like_url_manifest(path):
            for url in iter_manifest_urls(path):
                yield {
                    "source_key": source_key_for_url(url),
                    "source_id": url,
                    "title": url,
                    "doc_type": "web",
                    "url": url,
                    "location": url,
                }


def load_pdf(path: Path) -> dict:
    doc = fitz.open(path)
    pages = []
    for index, page in enumerate(doc, start=1):
        pages.append({"page": index, "text": page.get_text("text")})
    doc.close()
    return {
        "source_key": source_key_for_path(path),
        "source_id": path.stem,
        "title": path.stem,
        "doc_type": "pdf",
        "path": str(path),
        "pages": pages,
    }


def load_docx(path: Path) -> dict:
    if Document is None:
        raise RuntimeError("python-docx is required to load DOCX files. Install requirements.txt first.")
    document = Document(path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return {
        "source_key": source_key_for_path(path),
        "source_id": path.stem,
        "title": path.stem,
        "doc_type": "docx",
        "path": str(path),
        "text": "\n".join(paragraphs),
    }


def load_text_like(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "source_key": source_key_for_path(path),
        "source_id": path.stem,
        "title": path.stem,
        "doc_type": path.suffix.replace('.', '') or 'text',
        "path": str(path),
        "text": text,
    }


def load_url(url: str) -> dict:
    errors: list[str] = []
    for candidate_url in (url, url.rstrip("/")):
        try:
            response = requests.get(candidate_url, timeout=30, headers=BROWSER_HEADERS)
            response.raise_for_status()
            extracted = (
                trafilatura.extract(response.text, include_comments=False, include_tables=True)
                if trafilatura
                else None
            )
            if not extracted:
                soup = BeautifulSoup(response.text, "html.parser")
                extracted = soup.get_text("\n", strip=True)
            return {
                "source_key": source_key_for_url(candidate_url),
                "source_id": candidate_url,
                "title": candidate_url,
                "doc_type": "web",
                "url": candidate_url,
                "text": extracted or "",
            }
        except requests.RequestException as exc:
            errors.append(f"{candidate_url} -> {exc}")
    raise RuntimeError(" ; ".join(errors))


def iter_sources(input_dir: Path) -> Iterable[dict]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            yield load_pdf(path)
        elif suffix == ".docx":
            yield load_docx(path)
        elif suffix in TEXT_EXTENSIONS:
            if _looks_like_url_manifest(path):
                for url in iter_manifest_urls(path):
                    try:
                        yield load_url(url)
                    except Exception as exc:  # pragma: no cover
                        print(f"[WARN] Skipping blocked or failed URL: {url} | {exc}")
                        continue
            else:
                yield load_text_like(path)
        elif suffix in {".json", ".csv"} and _looks_like_url_manifest(path):
            for url in iter_manifest_urls(path):
                try:
                    yield load_url(url)
                except Exception as exc:  # pragma: no cover
                    print(f"[WARN] Skipping blocked or failed URL: {url} | {exc}")
                    continue
