# to run: python3 -m VectordB.ChromaChat --auto-ingest-externals (optional)

import os
import sys
import logging
import time
import re
from typing import List, Dict, Tuple

from chromadb import PersistentClient
from openai import OpenAI
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
from uuid import uuid4
from config import get_api_key as get_openai_key, get_serpapi_key



# === Config ===
PERSIST_PATH = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_storage")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "fcc_documents")
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500
SIMILARITY_THRESHOLD = 0.90
FALLBACK_TEXT = "No information available in the dataset or external sources for that question."
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

# Configure logging: keep backend/internal logs hidden by default.
logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# === Clients ===
client = PersistentClient(path=PERSIST_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

openai_client = OpenAI(api_key=get_openai_key())
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

# === Embedding & Retrieval ===
def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def retrieve_relevant_chunks(query: str, top_k: int = SIMILARITY_TOP_K) -> List[Dict]:
    q_emb = embed_text(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"document": doc, "metadata": meta} for doc, meta in zip(docs, metas)]

# === Trusted Source Filtering ===
def is_trusted_source(url: str) -> bool:
    return any(domain in url for domain in DEFAULT_SEARCH_QUERIES)

# === External Search and Ingestion ===
def external_search(query: str, max_results: int = 5) -> List[Dict]:
    params = {
        "q": query,
        "engine": "google",
        "api_key": SERPAPI_API_KEY,
        "num": max_results,
        "hl": "en",
        "gl": "us"
    }
    results = GoogleSearch(params).get_dict()
    external = []
    for r in results.get("organic_results", []):
        url = r.get("link", "")
        if is_trusted_source(url):
            external.append({
                "title": r.get("title", "Untitled"),
                "url": url,
                "content": r.get("snippet", "")
            })
    return external

def fetch_full_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return "\n".join(p.get_text() for p in soup.find_all("p"))
    except Exception:
        return ""

def ingest_external_document_to_chroma(doc: Dict):
    content = doc.get("content", "")
    if not content or len(content) < 300:
        return

    embedding = embed_text(content)
    uid = str(uuid4())
    metadata = {
        "source": doc.get("url", ""),
        "title": doc.get("title", ""),
        "retrieved": str(datetime.date.today())
    }
    collection.add(
        ids=[uid],
        documents=[content],
        embeddings=[embedding],
        metadatas=[metadata]
    )

# === Prompt Construction ===

def build_prompt(query: str, embedded_chunks: List[Dict], external_docs: List[Dict]) -> str:
    system_instructions = ( "You are a helpful assistant that specializes in emergency alert systems, public safety communications, cybersecurity policy, "
        "disaster response frameworks, and regulatory principles.\n\n"
        "Do not make up sources. Cite your information using markdown links at the end.")

    parts = []

    for doc in external_docs:
        title = doc.get("title", "External Source")
        url = doc.get("url", "")
        parts.append(f"Title: {title}" + (f" (URL: {url})" if url else "") + f"\n{doc.get('content', '')}")

    context_text = "\n---\n".join(parts)

    return (
        f"{system_instructions}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer (with markdown citations under 'Sources:'):"
    )

def parse_sources(answer: str) -> Tuple[str, List[Tuple[str, str]]]:
    marker = "\n Sources:"
    if marker in answer:
        ans_part, src_part = answer.split(marker, 1)
        sources = []
        for line in src_part.strip().splitlines():
            if line.startswith("- [") and "](" in line:
                try:
                    title = line.split("[", 1)[1].split("]")[0]
                    url = line.split("(", 1)[1].split(")")[0]
                    sources.append((title, url))
                except Exception:
                    continue
        return ans_part.strip(), sources
    return answer.strip(), []

# === Main Chat Loop ===

def chat():
    print("🔎 Regulatory Assistant (type 'exit' or Ctrl-C to quit)")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            embedded_chunks = retrieve_relevant_chunks(user_input)

            external_docs = external_search(user_input)
            for doc in external_docs:
                full_text = fetch_full_text(doc["url"])
                if full_text:
                    doc["content"] = full_text
                    ingest_external_document_to_chroma(doc)

            if not embedded_chunks and not external_docs:
                print("Assistant: No relevant information found in trusted sources.")
                continue

            prompt = build_prompt(user_input, embedded_chunks, external_docs)

            response = None
            for attempt in range(3):
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=MAX_RESPONSE_TOKENS,
                        temperature=0.3,
                    )
                    break
                except Exception as e:
                    time.sleep(1)

            if not response:
                print("Assistant: Sorry, couldn't get a response.")
                continue

            full_answer = response.choices[0].message.content.strip()
            ans_text, sources = parse_sources(full_answer)

            print(f"\nAssistant: {ans_text}\n")
            if sources:
                print("Sources:")
                for title, url in sources:
                    print(f"- [{title}]({url})")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    chat()
