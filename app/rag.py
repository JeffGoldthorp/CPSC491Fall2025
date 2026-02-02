# app/rag.py
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

EMBED_MODEL_DEFAULT = "text-embedding-3-small"

THEME_QUERY_HINTS: List[Tuple[str, str]] = [
    ("clean", "cleanliness_atmosphere"),
    ("dirty", "cleanliness_atmosphere"),
    ("sanitation", "health_safety"),
    ("food poisoning", "health_safety"),
    ("sick", "health_safety"),
    ("wait", "speed_wait_time"),
    ("slow", "speed_wait_time"),
    ("service", "service_friendliness"),
    ("friendly", "service_friendliness"),
    ("rude", "service_friendliness"),
    ("portion", "portion_value"),
    ("expensive", "portion_value"),
    ("price", "portion_value"),
    ("spicy", "spice_level"),
    ("heat", "spice_level"),
    ("flavor", "food_quality"),
    ("soggy", "food_quality"),
    ("crispy", "food_quality"),
]

CHUNK_REVIEW_RE = re.compile(r"^(r\d+)(?:-owner)?-c\d{2}$")

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing env var: {name}")
    return v

def _match_to_dict(m: Any) -> Dict[str, Any]:
    """Normalize Pinecone match objects across client versions."""
    if isinstance(m, dict):
        return m
    out: Dict[str, Any] = {}
    for k in ("id", "score", "metadata"):
        if hasattr(m, k):
            out[k] = getattr(m, k)
    return out

def _extract_json_object(text: str) -> str:
    """
    Handles raw JSON, fenced ```json ...```, and extra chatter.
    Returns best-effort JSON object string.
    """
    t = (text or "").strip()

    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last != -1 and last > first:
        t = t[first:last + 1].strip()
    return t

def embed_text(client: OpenAI, text: str, embed_model: str) -> List[float]:
    return client.embeddings.create(model=embed_model, input=[text]).data[0].embedding

def _query_hint_themes(query: str) -> List[str]:
    q = (query or "").lower()
    hinted = []
    for needle, theme in THEME_QUERY_HINTS:
        if needle in q:
            hinted.append(theme)
    return list(dict.fromkeys(hinted))

def chunk_id_to_review_id(chunk_id: str) -> Optional[str]:
    """
    Extract review_id from a chunk_id like:
      - r00037-c00  -> r00037
      - r00037-owner-c00 -> r00037
    """
    if not chunk_id:
        return None
    m = CHUNK_REVIEW_RE.match(chunk_id.strip())
    return m.group(1) if m else None

def unique_review_ids_from_chunks(chunk_ids: List[str]) -> Set[str]:
    s: Set[str] = set()
    for cid in chunk_ids or []:
        rid = chunk_id_to_review_id(cid)
        if rid:
            s.add(rid)
    return s

def retrieve(
    query: str,
    top_k: int = 8,
    exclude_owner_responses: bool = True,
    candidate_multiplier: int = 3,
    dedupe_by_review: bool = True,
) -> List[Dict[str, Any]]:
    openai_key = _env("OPENAI_API_KEY")
    pinecone_key = _env("PINECONE_API_KEY")
    index_name = _env("PINECONE_INDEX")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", EMBED_MODEL_DEFAULT)

    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    qvec = embed_text(client, query, embed_model)

    filt = {"is_owner_response": False} if exclude_owner_responses else None

    # pull more candidates, then dedupe/rerank locally
    res = index.query(
        vector=qvec,
        top_k=max(top_k * candidate_multiplier, top_k),
        include_metadata=True,
        include_values=False,
        filter=filt,
    )

    matches = getattr(res, "matches", None) or (res.get("matches", []) if isinstance(res, dict) else []) or []
    hinted_themes = set(_query_hint_themes(query))

    rows: List[Dict[str, Any]] = []
    for m in matches:
        mdict = _match_to_dict(m)
        md = mdict.get("metadata", {}) or {}
        rows.append({
            "id": mdict.get("id"),
            "score": float(mdict.get("score", 0.0)),
            "text": md.get("text", ""),
            "rating": md.get("rating"),
            "date": md.get("date"),
            "themes": md.get("themes", []) or [],
            "compound": md.get("compound"),
            "sentiment_label": md.get("sentiment_label") or md.get("sentiment", {}).get("label"),
            "review_id": md.get("review_id"),
        })

    # optional soft rerank: boost if query-hinted theme appears in chunk themes
    if hinted_themes:
        for r in rows:
            if any(t in hinted_themes for t in (r.get("themes") or [])):
                r["score"] = float(r["score"]) + 0.02  # tiny deterministic bump

    rows.sort(key=lambda x: x["score"], reverse=True)

    if not dedupe_by_review:
        return rows[:top_k]

    # dedupe by review_id to avoid N chunks from one review
    out: List[Dict[str, Any]] = []
    seen_reviews = set()
    for r in rows:
        rid = r.get("review_id") or ""
        if rid and rid in seen_reviews:
            continue
        if rid:
            seen_reviews.add(rid)
        out.append(r)
        if len(out) >= top_k:
            break

    return out

def _words(s: str) -> List[str]:
    return re.findall(r"\b\w[\w'-]*\b", s or "")

def _excerpt_from_text(text: str, max_words: int = 25) -> str:
    """
    Deterministic excerpt: first max_words words from the chunk.
    (This avoids model hallucinated quotes.)
    """
    ws = _words((text or "").replace("\n", " ").strip())
    if not ws:
        return ""
    return " ".join(ws[:max_words])

def _validate_sms(out: Dict[str, Any]) -> None:
    sms = out.get("sms_draft") or {}
    msgs = sms.get("messages") or []
    if not isinstance(msgs, list):
        return
    counts = [len(m) for m in msgs]
    sms["character_counts"] = counts
    out["sms_draft"] = sms

def _enforce_recurring_threshold(out: Dict[str, Any], min_unique_reviews: int) -> None:
    """
    Deterministic policy:
    - recurring issue requires evidence from >= min_unique_reviews distinct review_ids
    - else it moves to isolated_issues
    """
    recurring = out.get("recurring_issues", [])
    if not isinstance(recurring, list):
        out["recurring_issues"] = []
        out["isolated_issues"] = []
        return

    kept: List[Dict[str, Any]] = []
    moved: List[Dict[str, Any]] = []

    for item in recurring:
        if not isinstance(item, dict):
            continue
        ev = item.get("evidence_chunk_ids") or []
        if not isinstance(ev, list):
            ev = []
        review_ids = unique_review_ids_from_chunks(ev)

        item2 = dict(item)
        item2["unique_review_count"] = len(review_ids)
        item2["unique_review_ids"] = sorted(review_ids)

        if len(review_ids) >= min_unique_reviews:
            kept.append(item2)
        else:
            moved.append(item2)

    out["recurring_issues"] = kept
    out["isolated_issues"] = moved

def _filter_to_known_chunk_ids(out: Dict[str, Any], known_ids: Set[str]) -> None:
    """
    Hard validation: any referenced chunk id must exist in retrieved contexts.
    Unknown ids are removed deterministically.
    """
    # top_themes evidence
    if isinstance(out.get("top_themes"), list):
        for th in out["top_themes"]:
            if not isinstance(th, dict):
                continue
            ev = th.get("evidence", [])
            if not isinstance(ev, list):
                th["evidence"] = []
                continue
            th["evidence"] = [e for e in ev if isinstance(e, dict) and e.get("chunk_id") in known_ids]

    # recurring issues
    if isinstance(out.get("recurring_issues"), list):
        for it in out["recurring_issues"]:
            if not isinstance(it, dict):
                continue
            ids = it.get("evidence_chunk_ids") or []
            if not isinstance(ids, list):
                it["evidence_chunk_ids"] = []
            else:
                it["evidence_chunk_ids"] = [cid for cid in ids if cid in known_ids]

    # ops recommendations
    if isinstance(out.get("ops_recommendations"), list):
        for it in out["ops_recommendations"]:
            if not isinstance(it, dict):
                continue
            ids = it.get("grounding_chunk_ids") or []
            if not isinstance(ids, list):
                it["grounding_chunk_ids"] = []
            else:
                it["grounding_chunk_ids"] = [cid for cid in ids if cid in known_ids]

def _populate_missing_excerpts(out: Dict[str, Any], id_to_text: Dict[str, str]) -> None:
    """
    If evidence excerpts are missing or empty, fill them deterministically from chunk text.
    (Keeps your schema but removes dependency on model quoting.)
    """
    if not isinstance(out.get("top_themes"), list):
        return
    for th in out["top_themes"]:
        if not isinstance(th, dict):
            continue
        ev = th.get("evidence", [])
        if not isinstance(ev, list):
            continue
        for e in ev:
            if not isinstance(e, dict):
                continue
            cid = e.get("chunk_id")
            if not cid:
                continue
            if not (e.get("excerpt") or "").strip():
                e["excerpt"] = _excerpt_from_text(id_to_text.get(cid, ""), max_words=25)

def build_prompt(query: str, contexts: List[Dict[str, Any]], max_chunk_chars: int = 650) -> List[Dict[str, str]]:
    """
    IMPORTANT CHANGE:
    - We still *allow* the model to output excerpts, but we will overwrite/fill them deterministically.
    - We also explicitly forbid chunk ids outside the provided list.
    """
    ctx_lines = []
    allowed_ids = []
    for c in contexts:
        cid = c.get("id")
        if cid:
            allowed_ids.append(cid)

        txt = (c.get("text") or "").replace("\n", " ").strip()
        if len(txt) > max_chunk_chars:
            txt = txt[:max_chunk_chars].rstrip() + "…"

        ctx_lines.append(
            f"- [chunk_id={c['id']}] (rating={c.get('rating')}, date={c.get('date')}, score={c.get('score'):.3f}) {txt}"
        )

    ctx_block = "\n".join(ctx_lines)
    allowed_block = ", ".join(allowed_ids)

    system = (
        "You are a customer feedback analyst. Return ONLY a single JSON object.\n"
        "Use ONLY the provided review chunks as evidence.\n"
        "CRITICAL: You may ONLY cite chunk_id values that appear in the retrieved chunks list.\n"
        "If a claim is not supported by the chunks, write exactly:\n"
        "\"Not enough evidence in retrieved reviews.\""
    )

    user = f"""
USER REQUEST:
{query}

ALLOWED CHUNK IDS (you may ONLY cite from this list):
{allowed_block}

RETRIEVED REVIEW CHUNKS:
{ctx_block}

Return ONLY valid JSON with this schema:

{{
  "answer_summary": "string",
  "top_themes": [
    {{
      "theme": "string",
      "sentiment": "positive|mixed|negative",
      "evidence": [
        {{"chunk_id": "string", "excerpt": "string"}}
      ]
    }}
  ],
  "overall_sentiment": {{"label": "positive|mixed|negative", "rationale": "string"}},
  "recurring_issues": [
    {{"issue": "string", "evidence_chunk_ids": ["string"]}}
  ],
  "sms_draft": {{
    "messages": ["string"],
    "character_counts": [0]
  }},
  "ops_recommendations": [
    {{"recommendation": "string", "grounding_chunk_ids": ["string"]}}
  ]
}}

Rules:
- Provide 3–5 themes max.
- Evidence excerpts must be short (<= 25 words) and MUST come from the chunk text.
- recurring_issues: include 0–3 issues max.
- sms_draft: 2–3 messages, warm/professional, NO emojis, NO discounts, NO legal language, each <= 160 chars.
- ops_recommendations: 2–3 items; each must cite grounding_chunk_ids.
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def generate_grounded_response(
    query: str,
    top_k: int = 8,
    min_recurring_reviews: int = 2,
    include_debug: bool = False,
) -> Dict[str, Any]:
    openai_key = _env("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=openai_key)

    contexts = retrieve(query, top_k=top_k, exclude_owner_responses=True)

    # If retrieval is empty, return deterministic "not enough evidence"
    if not contexts:
        return {
            "answer_summary": "Not enough evidence in retrieved reviews.",
            "top_themes": [],
            "overall_sentiment": {"label": "mixed", "rationale": "Not enough evidence in retrieved reviews."},
            "recurring_issues": [],
            "isolated_issues": [],
            "sms_draft": {"messages": ["Thank you for your feedback. We’re looking into this and appreciate the details."],
                         "character_counts": [len("Thank you for your feedback. We’re looking into this and appreciate the details.")]},
            "ops_recommendations": [],
            "debug": {"retrieved": 0, "top_k": top_k} if include_debug else {},
        }

    id_to_text = {c["id"]: (c.get("text") or "") for c in contexts if c.get("id")}
    known_ids = set(id_to_text.keys())

    messages = build_prompt(query, contexts)

    resp = client.responses.create(
        model=model,
        input=messages,
        text={"format": {"type": "json_object"}},
        temperature=0,
    )

    raw = (resp.output_text or "").strip()
    cooked = _extract_json_object(raw)

    try:
        out = json.loads(cooked)
        if not isinstance(out, dict):
            return {"error": "Model returned non-object JSON", "raw_output": raw[:4000]}

        # 1) validate sms counts (deterministic)
        _validate_sms(out)

        # 2) hard filter citations to retrieved chunk ids (deterministic)
        _filter_to_known_chunk_ids(out, known_ids)

        # 3) populate/overwrite missing excerpts deterministically (deterministic)
        _populate_missing_excerpts(out, id_to_text)

        # 4) enforce "recurring" threshold deterministically
        _enforce_recurring_threshold(out, min_unique_reviews=min_recurring_reviews)

        # 5) add debug block (optional)
        if include_debug:
            out.setdefault("debug", {})
            out["debug"]["retrieved"] = len(contexts)
            out["debug"]["top_k"] = top_k
            out["debug"]["min_recurring_reviews"] = min_recurring_reviews
            out["debug"]["allowed_chunk_ids"] = sorted(list(known_ids))

        return out

    except json.JSONDecodeError as e:
        return {
            "error": "Model did not return valid JSON",
            "raw_output": raw[:4000],
            "exception": str(e),
        }
