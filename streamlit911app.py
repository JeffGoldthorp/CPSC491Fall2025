# streamlit_app.py
# 🍗 Customer Review Insights Bot — Chat UI + Conversation Dashboard
#
# Sidebar:
#   - New chat
#   - Conversation history (click to open)
#   - Rename + Delete
# Main page:
#   - Chat
#   - Advanced settings (collapsed)
#   - Debug (collapsed, optional)

import json
import time
import uuid
import streamlit as st
from app.rag import generate_grounded_response

st.set_page_config(
    page_title="🍗 Customer Review Insights Bot",
    page_icon="🍗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Session state helpers
# =========================
def _init_state() -> None:
    if "conversations" not in st.session_state:
        # { chat_id: { "title": str, "messages": [{"role": "user|assistant", "content": str}], "created": float } }
        st.session_state.conversations = {}

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "top_k": 12,
            "min_recurring_reviews": 2,
            "debug_on": False,
        }

def _new_chat() -> None:
    chat_id = str(uuid.uuid4())
    st.session_state.conversations[chat_id] = {
        "title": "New chat",
        "messages": [],
        "created": time.time(),
    }
    st.session_state.active_chat_id = chat_id

def _ensure_active_chat() -> None:
    if not st.session_state.conversations:
        _new_chat()
        return
    if st.session_state.active_chat_id not in st.session_state.conversations:
        # Fall back to newest chat
        newest = max(st.session_state.conversations.items(), key=lambda kv: kv[1]["created"])[0]
        st.session_state.active_chat_id = newest

def _set_title_from_first_user_message(chat_id: str) -> None:
    convo = st.session_state.conversations[chat_id]
    if convo["title"] != "New chat":
        return
    for m in convo["messages"]:
        if m["role"] == "user" and m["content"].strip():
            t = m["content"].strip()
            convo["title"] = t[:40] + ("…" if len(t) > 40 else "")
            return

def _delete_chat(chat_id: str) -> None:
    st.session_state.conversations.pop(chat_id, None)
    if not st.session_state.conversations:
        _new_chat()
    else:
        newest = max(st.session_state.conversations.items(), key=lambda kv: kv[1]["created"])[0]
        st.session_state.active_chat_id = newest

def _sorted_chat_ids_newest_first():
    items = sorted(
        st.session_state.conversations.items(),
        key=lambda kv: kv[1]["created"],
        reverse=True,
    )
    return [cid for cid, _ in items]


# =========================
# Init
# =========================
_init_state()
if st.session_state.active_chat_id is None:
    _new_chat()
_ensure_active_chat()

# =========================
# Sidebar — Conversation dashboard
# =========================
with st.sidebar:
    st.header("💬 Conversations")

    # New chat button
    if st.button("➕ New chat", use_container_width=True):
        _new_chat()
        st.rerun()

    st.divider()

    chat_ids = _sorted_chat_ids_newest_first()
    active = st.session_state.active_chat_id

    # Conversation list (click to open)
    selected = st.radio(
        "History",
        options=chat_ids,
        index=chat_ids.index(active) if active in chat_ids else 0,
        format_func=lambda cid: st.session_state.conversations[cid]["title"],
        label_visibility="collapsed",
    )
    st.session_state.active_chat_id = selected
    _ensure_active_chat()

    st.divider()

    # Optional rename
    convo = st.session_state.conversations[st.session_state.active_chat_id]
    with st.expander("✏️ Rename / Manage", expanded=False):
        new_title = st.text_input("Conversation title", value=convo["title"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save title", use_container_width=True):
                convo["title"] = (new_title.strip() or "Untitled")
                st.rerun()
        with col2:
            if st.button("Delete chat", use_container_width=True):
                _delete_chat(st.session_state.active_chat_id)
                st.rerun()

# =========================
# Main — Chat UI
# =========================
st.title("🍗 Customer Review Insights Bot")
st.caption("Ask questions about your review dataset. The bot retrieves relevant chunks and generates grounded insights.")

cid = st.session_state.active_chat_id
convo = st.session_state.conversations[cid]

# Advanced settings expander (collapsed by default)
with st.expander("⚙️ Advanced settings", expanded=False):
    st.session_state.settings["top_k"] = st.slider(
        "Retrieved chunks (top_k)",
        min_value=3,
        max_value=50,
        value=int(st.session_state.settings["top_k"]),
        step=1,
    )
    st.session_state.settings["min_recurring_reviews"] = st.slider(
        "Min recurring reviews",
        min_value=1,
        max_value=10,
        value=int(st.session_state.settings["min_recurring_reviews"]),
        step=1,
    )
    st.session_state.settings["debug_on"] = st.toggle(
        "Enable debug output",
        value=bool(st.session_state.settings["debug_on"]),
    )

# Render history
for msg in convo["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask a question about the reviews…")

def _render_details(out: dict) -> None:
    """Collapsed details (themes/issues/SMS/ops + sources)."""
    with st.expander("Details (themes, issues, SMS, ops)", expanded=False):
        # Overall sentiment
        osent = out.get("overall_sentiment") or {}
        st.markdown(f"**Overall sentiment:** `{osent.get('label', 'mixed')}`")
        if osent.get("rationale"):
            st.write(osent["rationale"])

        # Themes
        st.subheader("Themes")
        themes = out.get("top_themes", []) or []
        if not themes:
            st.write("_No themes returned._")
        else:
            for th in themes:
                st.markdown(f"**{th.get('theme', 'Theme')}** • sentiment: `{th.get('sentiment', 'mixed')}`")
                for ev in th.get("evidence", []) or []:
                    st.markdown(f"- `{ev.get('chunk_id','')}` — {ev.get('excerpt','')}")

        # Issues
        st.subheader("Recurring issues")
        rec = out.get("recurring_issues", []) or []
        if not rec:
            st.write("_None._")
        else:
            for i, it in enumerate(rec, start=1):
                st.markdown(f"**{i}. {it.get('issue','')}**")
                ids = it.get("evidence_chunk_ids", []) or []
                if ids:
                    st.code("\n".join(ids))

        iso = out.get("isolated_issues", []) or []
        if iso:
            st.subheader("Isolated issues")
            for i, it in enumerate(iso, start=1):
                st.markdown(f"**{i}. {it.get('issue','')}**")
                ids = it.get("evidence_chunk_ids", []) or []
                if ids:
                    st.code("\n".join(ids))

        # SMS
        st.subheader("Draft SMS")
        sms = out.get("sms_draft") or {}
        msgs = sms.get("messages", []) or []
        if not msgs:
            st.write("_None._")
        else:
            for i, m in enumerate(msgs, start=1):
                st.markdown(f"**Message {i}**")
                st.code(m)

        # Ops recommendations
        st.subheader("Ops recommendations")
        recs = out.get("ops_recommendations", []) or []
        if not recs:
            st.write("_None._")
        else:
            for i, r in enumerate(recs, start=1):
                st.markdown(f"**{i}. {r.get('recommendation','')}**")
                ids = r.get("grounding_chunk_ids", []) or []
                if ids:
                    st.caption("Grounding chunk IDs")
                    st.code("\n".join(ids))

        # Sources
        st.subheader("Sources (chunk IDs)")
        src = set()
        for th in out.get("top_themes", []) or []:
            for ev in th.get("evidence", []) or []:
                if ev.get("chunk_id"):
                    src.add(ev["chunk_id"])
        for r in out.get("ops_recommendations", []) or []:
            for x in r.get("grounding_chunk_ids", []) or []:
                src.add(x)
        for it in out.get("recurring_issues", []) or []:
            for x in it.get("evidence_chunk_ids", []) or []:
                src.add(x)
        if src:
            st.code("\n".join(sorted(src)))
        else:
            st.write("_No sources provided._")

def _render_debug(out: dict) -> None:
    """Collapsed debug expander (only if enabled)."""
    if not st.session_state.settings["debug_on"]:
        return
    with st.expander("🧪 Debug", expanded=False):
        st.code(json.dumps(out, ensure_ascii=False, indent=2), language="json")

if prompt:
    # Append user message
    convo["messages"].append({"role": "user", "content": prompt})
    _set_title_from_first_user_message(cid)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating grounded insights…"):
            out = generate_grounded_response(
                query=prompt,
                top_k=int(st.session_state.settings["top_k"]),
                min_recurring_reviews=int(st.session_state.settings["min_recurring_reviews"]),
                include_debug=bool(st.session_state.settings["debug_on"]),
            )

        if not isinstance(out, dict):
            st.error("Unexpected output type from backend (expected dict).")
            st.write(out)
        else:
            # Main assistant reply: keep it clean
            answer = out.get("answer_summary", "").strip() or "No answer summary returned."
            st.markdown(answer)

            # Details + debug collapsed
            _render_details(out)
            _render_debug(out)

            # Store assistant message in history (store clean text, not the whole JSON)
            convo["messages"].append({"role": "assistant", "content": answer})
