from __future__ import annotations

import os
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from serpapi import GoogleSearch
except Exception:
    GoogleSearch = None

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

BLOCKLIST = {
    "researchgate.net",
}


def _blocked(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(domain in host for domain in BLOCKLIST)


def external_search(query: str, max_results: int = 3) -> list[dict]:
    if not SERPAPI_API_KEY or GoogleSearch is None:
        return []

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": max_results,
        "hl": "en",
        "gl": "us",
    }

    try:
        result = GoogleSearch(params).get_dict()
    except Exception:
        return []

    docs = []
    for item in result.get("organic_results", [])[:max_results]:
        url = item.get("link")
        title = item.get("title") or "Untitled"
        snippet = item.get("snippet") or ""

        if not url or _blocked(url):
            continue

        docs.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )

    return docs


def fetch_full_text(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()

        if trafilatura:
            extracted = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=True,
            )
            if extracted and extracted.strip():
                return extracted.strip()

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text("\n", strip=True)
        return text.strip()
    except Exception:
        return ""