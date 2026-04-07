from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
APP_PARENT = ROOT / "app"   # because your file is in repo root, and package is in app/app/

if str(APP_PARENT) not in sys.path:
    sys.path.insert(0, str(APP_PARENT))

import streamlit as st
from app.services.chat_service import answer_question

st.set_page_config(
    page_title="PSAP 911 Cyber Risk Assistant",
    page_icon="📞",
    layout="wide"
)

st.title("📞 PSAP 911 Cyber Risk Assistant")
st.caption("RAG chatbot for 911 call centers using curated PDFs and web-enriched cybersecurity sources")

namespace = st.sidebar.selectbox(
    "Knowledge source",
    ["all", "psap-911-curated", "psap-911-web"],
    index=0,
)

allow_web_fallback = st.sidebar.checkbox(
    "Enable live web fallback (SerpAPI)",
    value=False,
)

mode = st.sidebar.selectbox(
    "Mode",
    ["rag"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.write("**all** = curated PDFs + ingested web URLs")
st.sidebar.write("Turn on live web fallback only when the indexed corpus is thin.")

if st.sidebar.button("New chat"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_citations(citations, expanded: bool = False):
    if not citations:
        return

    with st.expander("Sources used", expanded=expanded):
        st.caption("This answer is grounded in retrieved source excerpts. Page numbers are only available for documents with fixed pagination.")
        for citation in citations:
            page_display = "n/a"
            if citation["page_start"] is not None and citation["page_end"] is not None:
                page_display = f"{citation['page_start']}–{citation['page_end']}"
            elif citation["page_start"] is not None:
                page_display = str(citation["page_start"])
            elif citation["page_end"] is not None:
                page_display = str(citation["page_end"])

            authority_display = citation.get("authority_level") or "unknown"
            source_type = "Web" if citation.get("url") and str(citation.get("url")).startswith(("http://", "https://")) else "Document"
            page_note = " (web sources have no page numbers)" if page_display == "n/a" else ""

            st.markdown(
                f"**[{citation['rank']}] {citation['title']}**  \n"
                f"Source Type: `{source_type}`  \n"
                f"Source ID: `{citation['source_id']}`  \n"
                f"Pages: `{page_display}`{page_note}  \n"
                f"Authority: `{authority_display}`  \n"
                f"Score: `{citation.get('score')}`"
            )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            _render_citations(msg.get("citations"), expanded=False)

question = st.chat_input("Ask a question about PSAP cybersecurity, NG911, incident response, or vendor risk...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching sources..."):
            result = answer_question(
                question=question,
                namespace=namespace,
                mode=mode,
                allow_web_fallback=allow_web_fallback,
            )

            st.markdown(result.answer)
            _render_citations([citation.dict() for citation in result.citations], expanded=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": [citation.dict() for citation in result.citations],
        }
    )
