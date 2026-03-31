from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="PSAP Cyber Risk Assistant", layout="wide")
st.title("PSAP Cyber Risk Assistant")
st.caption("Experiment with base, fine-tuned, RAG, and hybrid modes")

with st.sidebar:
    namespace = st.text_input("Namespace", value="public_authoritative")
    mode = st.selectbox("Mode", ["base", "finetuned", "rag", "hybrid"], index=2)
    model_override = st.text_input("Model override", value="")
    st.markdown("Use Streamlit for the class demo or internal pilot.")

question = st.text_area(
    "Ask a question",
    placeholder="What should a small PSAP prioritize first to reduce ransomware risk?",
    height=140,
)

if st.button("Ask") and question.strip():
    with st.spinner("Generating answer..."):
        response = requests.post(
            API_URL,
            json={
                "question": question,
                "namespace": namespace,
                "filters": None,
                "mode": mode,
                "model_override": model_override or None,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()

    st.subheader(f"Answer ({payload['mode']})")
    st.caption(f"Model used: {payload['model_used']}")
    st.write(payload["answer"])

    st.subheader("Citations")
    if payload["citations"]:
        for citation in payload["citations"]:
            st.markdown(
                f"**[{citation['rank']}] {citation['title']}**  \n"
                f"source_id: `{citation['source_id']}`  \n"
                f"authority: `{citation.get('authority_level')}`  \n"
                f"section: `{citation.get('section')}`  \n"
                f"pages: `{citation.get('page_start')} - {citation.get('page_end')}`  \n"
                f"score: `{citation.get('score')}`"
            )
    else:
        st.info("This mode does not use retrieval citations.")
