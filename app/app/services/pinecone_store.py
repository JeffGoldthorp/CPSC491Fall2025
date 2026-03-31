from __future__ import annotations

from app.clients import get_pinecone_index


def upsert_records(records: list[dict], namespace: str) -> None:
    if records:
        get_pinecone_index().upsert(vectors=records, namespace=namespace)



def query_records(vector: list[float], namespace: str, top_k: int, metadata_filter: dict | None = None):
    return get_pinecone_index().query(
        namespace=namespace,
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )
