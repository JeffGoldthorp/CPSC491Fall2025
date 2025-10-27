"""
Ingestion pipeline with similarity filtering for ChromaDB.

Features:
- Uses OpenAI embeddings to embed scraped pages (chunked) using same model as chat.
- Checks similarity against existing vectors in ChromaDB; skips near-duplicates.
- Chunks large documents before embedding with overlap.
- Supports URL acquisition via: hardcoded SEARCH_QUERIES (SerpAPI), a --urls-file, or direct --url args.
- Provides summary report at end.
- Graceful handling of timeouts / rate limits with simple exponential backoff.

Usage examples:
  python ingestion/ingest_with_similarity.py
  python ingestion/ingest_with_similarity.py --urls-file urls.txt
  python ingestion/ingest_with_similarity.py --url https://www.fcc.gov/example1 --url https://www.fcc.gov/example2

Environment variables required:
  OPENAI_API_KEY
Optional:
  SERPAPI_API_KEY, CHROMA_PERSIST_PATH

Result: Adds novel chunks (by similarity threshold) to the configured Chroma collection.
"""

import os
import time
import re
import argparse
import datetime
import logging
import html
import tempfile
from uuid import uuid4
from typing import List, Iterable, Optional, Tuple
from urllib.parse import urlparse

import mimetypes
import requests
import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm
from newspaper import Article
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import PersistentClient

from config import get_api_key as get_openai_key, get_serpapi_key
from openai import OpenAI

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPENAI_CLIENT = None
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.90
SIMILARITY_TOP_K = 5
MIN_ARTICLE_LENGTH = 300
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
SEARCH_RESULTS_PER_QUERY = 30
BACKOFF_BASE = 2.0
BACKOFF_MAX_SLEEP = 30

DOWNLOAD_TIMEOUT = 10
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

DEFAULT_SEARCH_QUERIES = [
    "emergency alert systems academic research site:gov OR site:edu OR site:org -site:fcc.gov",
    "public safety communications peer-reviewed articles site:ncbi.nlm.nih.gov OR site:sciencedirect.com -site:fcc.gov",
    "cybersecurity policy academic papers site:acm.org OR site:ieee.org -site:fcc.gov",
    "disaster response frameworks white papers site:mit.edu OR site:nist.gov OR site:rand.org -site:fcc.gov",
    "regulatory principles in public safety communications site:law.stanford.edu OR site:brookings.edu -site:fcc.gov",
    "non-FCC emergency alerting regulation case studies site:gov OR site:edu -site:fcc.gov",
    "cyber threats to alert systems site:csis.org OR site:rand.org OR site:arpa-e.energy.gov -site:fcc.gov",
    "resilience of emergency communications systems site:sciencedirect.com OR site:springer.com -site:fcc.gov",
    "machine learning in emergency alert reliability site:ieee.org OR site:arxiv.org -site:fcc.gov",
    "comparative regulation of alerting systems site:oecd.org OR site:gov.uk OR site:who.int -site:fcc.gov",
    "public comment analysis for emergency alerts site:regulations.gov -site:fcc.gov",
    "academic literature on administrative procedures in emergency policy site:jstor.org -site:fcc.gov",
]


PERSIST_PATH = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_fcc_storage")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "fcc_documents")

client = PersistentClient(path=PERSIST_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def backoff_sleep(attempt: int) -> None:
    sleep_time = min(BACKOFF_BASE ** attempt + (0.1 * attempt), BACKOFF_MAX_SLEEP)
    time.sleep(sleep_time)

def ensure_openai_client():
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        key = get_openai_key()
        OPENAI_CLIENT = OpenAI(api_key=key)
    return OPENAI_CLIENT

def embed_texts(texts: List[str], max_retries: int = 5) -> List[List[float]]:
    for attempt in range(max_retries):
        try:
            client = ensure_openai_client()
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [r.embedding for r in resp.data]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"   ⚠️ Embed batch retry {attempt+1}/{max_retries} after error: {e}")
            backoff_sleep(attempt)
    raise RuntimeError("Unreachable: embedding loop exhausted")

def is_similar_to_existing(embedding: List[float], threshold: float = SIMILARITY_THRESHOLD, top_k: int = SIMILARITY_TOP_K) -> bool:
    try:
        results = collection.query(query_embeddings=[embedding], n_results=top_k, include=["embeddings"])
        existing_embeddings = results.get("embeddings", [[]])[0]
        if existing_embeddings is None or len(existing_embeddings) == 0:
            return False
        existing = np.array(existing_embeddings)
        query_vec = np.array(embedding).reshape(1, -1)
        sims = cosine_similarity(query_vec, existing)[0]
        return float(np.max(sims)) >= threshold
    except Exception as e:
        print(f"   ⚠️ similarity check failed: {e}")
        return False

def fetch_search_results(query: str, limit: int = SEARCH_RESULTS_PER_QUERY) -> List[str]:
    if not SERPAPI_AVAILABLE:
        return []
    api_key = get_serpapi_key()
    if not api_key:
        return []
    params = {"engine": "google", "q": query, "api_key": api_key, "num": limit}
    try:
        search = GoogleSearch(params)
        res = search.get_dict()
        return [r.get("link") for r in res.get("organic_results", []) if r.get("link")]
    except Exception as e:
        print(f"⚠️ Search failed for '{query}': {e}")
        return []

def is_pdf_url(url: str) -> bool:
    try:
        head = requests.head(url, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
        content_type = head.headers.get("Content-Type", "").lower()
        return "application/pdf" in content_type or url.lower().endswith(".pdf")
    except Exception:
        return url.lower().endswith(".pdf")

def scrape_article(url: str) -> Optional[dict]:
    try:
        if is_pdf_url(url):
            print(f"   📄 Detected PDF: {url}")
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            if resp.status_code != 200:
                print(f"   ✖ PDF download failed: {url}")
                return None
            if len(resp.content) > MAX_DOWNLOAD_BYTES:
                print(f"   ✖ PDF exceeds max download size: {url}")
                return None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(resp.content)
                tmp_path = tmp_file.name
            doc = fitz.open(tmp_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            os.unlink(tmp_path)
            if len(text) < MIN_ARTICLE_LENGTH:
                return None
            return {"url": url, "title": os.path.basename(urlparse(url).path), "text": text}

        art = Article(url)
        art.download()
        art.parse()
        text = (art.text or "").strip()
        if not text or len(text) < MIN_ARTICLE_LENGTH:
            return None
        return {"url": url, "title": art.title or "Untitled", "text": text}
    except Exception as e:
        print(f"   ✖ Failed to scrape {url}: {e}")
        return None

def url_already_ingested(url: str) -> bool:
    try:
        results = collection.query(query_texts=[url], n_results=1, include=["metadatas"])
        metadatas = results.get("metadatas", [[]])[0]
        return any(meta.get("source") == url for meta in metadatas)
    except Exception as e:
        print(f"   ⚠️ URL ingest check failed for {url}: {e}")
        return False

def ingest_from_urls(urls: Iterable[str]) -> Tuple[int, int]:
    added_chunks = 0
    skipped_chunks = 0

    for url in tqdm(list(urls), desc="URLs"):
        if url_already_ingested(url):
            continue
        scraped = scrape_article(url)
        if not scraped:
            continue
        chunks = chunk_text(scraped["text"])
        try:
            chunk_embs = embed_texts(chunks)
        except Exception as e:
            print(f"   ⚠️ Batch embedding failed for {url}: {e}")
            skipped_chunks += len(chunks)
            continue

        batched_ids = []
        batched_docs = []
        batched_embs = []
        batched_meta = []

        for idx, (chunk, emb) in enumerate(zip(chunks, chunk_embs)):
            if is_similar_to_existing(emb):
                print(f"   ⛔ Similar chunk skipped ({url} :: chunk {idx})")
                skipped_chunks += 1
                continue
            metadata = {
                "source": url,
                "title": scraped["title"],
                "retrieved": str(datetime.date.today()),
                "chunk_index": idx,
            }
            batched_ids.append(str(uuid4()))
            batched_docs.append(chunk)
            batched_embs.append(emb)
            batched_meta.append(metadata)
            added_chunks += 1

        if batched_ids:
            print(f"✅ Adding {len(batched_ids)} novel chunks to collection '{COLLECTION_NAME}' ...")
            collection.add(ids=batched_ids, documents=batched_docs, embeddings=batched_embs, metadatas=batched_meta)
        else:
            print(f"ℹ️ No new chunks to add for URL: {url}")

    return added_chunks, skipped_chunks

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents with similarity filtering into ChromaDB.")
    parser.add_argument("--url", dest="urls", action="append", help="One or more URLs to ingest", default=[])
    parser.add_argument("--urls-file", dest="urls_file", help="Path to file containing URLs (one per line)")
    parser.add_argument("--no-search", dest="no_search", action="store_true", help="Disable SerpAPI search; only use provided URLs")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Similarity threshold (default 0.85)")
    parser.add_argument("--top-k", type=int, default=SIMILARITY_TOP_K, help="Top K neighbors for similarity (default 5)")
    parser.add_argument("--dry-run", action="store_true", help="Do everything except adding to Chroma")
    return parser.parse_args()

def load_urls_from_file(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    except Exception as e:
        print(f"⚠️ Could not read URLs file '{path}': {e}")
        return []

def main():
    args = parse_args()

    global SIMILARITY_THRESHOLD, SIMILARITY_TOP_K
    SIMILARITY_THRESHOLD = args.threshold
    SIMILARITY_TOP_K = args.top_k

    url_set = set(args.urls or [])
    if args.urls_file:
        url_set.update(load_urls_from_file(args.urls_file))

    if not args.no_search and not url_set:
        print("🔍 Performing search queries (SerpAPI)...")
        if not SERPAPI_AVAILABLE:
            print("⚠️ SerpAPI package not installed.")
        elif not get_serpapi_key():
            print("⚠️ No SerpAPI key found.")
        else:
            for q in DEFAULT_SEARCH_QUERIES:
                print(f"   Query: {q}")
                results = fetch_search_results(q)
                if results:
                    url_set.update(results)

    urls = sorted(url_set)
    if not urls:
        print("No URLs to ingest. Provide --url, --urls-file, or enable search.")
        return

    try:
        ensure_openai_client()
    except Exception as e:
        print("✖ OpenAI configuration error:", e)
        return

    added, skipped = ingest_from_urls(urls)

    if args.dry_run:
        print("(Dry run) Skipped writing to DB.")
    else:
        print("Ingestion complete.")

    print("Summary:")
    print(f"  Added chunks:   {added}")
    print(f"  Skipped chunks: {skipped}")
    print(f"  Threshold:      {SIMILARITY_THRESHOLD}")
    print(f"  Top-K:          {SIMILARITY_TOP_K}")
    print(f"  Collection:     {COLLECTION_NAME}")
    print(f"  Persist path:   {PERSIST_PATH}")

if __name__ == "__main__":
    main()
