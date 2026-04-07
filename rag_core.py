from __future__ import annotations

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "psap-911-curated")
TOP_K = int(os.getenv("TOP_K", "8"))

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)


def embed_query(text: str) -> list[float]:
    resp = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
        dimensions=OPENAI_EMBEDDING_DIMENSIONS,
    )
    return resp.data[0].embedding


def retrieve_chunks(query: str, top_k: int = TOP_K):
    q = embed_query(query)
    results = index.query(
        vector=q,
        top_k=top_k * 2,  # Fetch more to deduplicate by source
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )
    out = []
    seen_sources = set()
    
    for match in results.matches:
        md = match.metadata or {}
        source = md.get("source", "")
        
        # Keep only the top-scoring result per unique source
        if source in seen_sources:
            continue
        seen_sources.add(source)
        
        out.append({
            "score": match.score,
            "title": md.get("title", "Unknown"),
            "source": source,
            "page": md.get("page"),
            "text": md.get("text", ""),
        })
        
        if len(out) >= top_k:
            break
    
    return out


def answer_question(question: str):
    chunks = retrieve_chunks(question)
    context = "\n\n---\n\n".join(
        f"[{i+1}] {c['title']} (source={c['source']}, page={c['page']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    prompt = f"""
You are a cybersecurity assistant for 911 call centers and PSAPs.
Answer using only the retrieved context below.
If the context is insufficient, say so clearly.
Cite sources inline using [1], [2], etc.

Context:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model=OPENAI_CHAT_MODEL,
        input=prompt,
    )

    return response.output_text, chunks