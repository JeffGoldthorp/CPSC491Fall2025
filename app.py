import streamlit as st
from app.services.chat_service import answer_question

st.set_page_config(page_title="PSAP 911 Cyber Risk Assistant", page_icon="📞", layout="wide")

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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

            with st.expander("Sources used"):
                if result.citations:
                    for citation in result.citations:
                        st.markdown(
                            f"**[{citation.rank}] {citation.title}**  \n"
                            f"Source ID: `{citation.source_id}`  \n"
                            f"Pages: `{citation.page_start}`–`{citation.page_end}`  \n"
                            f"Authority: `{citation.authority_level}`  \n"
                            f"Score: `{citation.score}`"
                        )
                else:
                    st.write("No citations returned.")

    st.session_state.messages.append({"role": "assistant", "content": result.answer})