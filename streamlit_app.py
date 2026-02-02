import json
import streamlit as st
from app.rag import generate_grounded_response

st.set_page_config(page_title="Review Insights Bot", layout="wide")

st.title("Customer Review Insights Bot")
st.caption("Grounded RAG over your review dataset (Pinecone + OpenAI).")

with st.sidebar:
    st.header("Controls")
    top_k = st.slider("top_k (retrieved chunks)", min_value=3, max_value=50, value=25, step=1)
    min_recurring = st.slider("min_recurring_reviews", min_value=1, max_value=5, value=2, step=1)
    include_debug = st.checkbox("Include debug block", value=True)

    st.divider()
    st.markdown("**Environment required:**")
    st.code(
        "OPENAI_API_KEY\nPINECONE_API_KEY\nPINECONE_INDEX\n"
        "OPENAI_MODEL (optional)\nOPENAI_EMBED_MODEL (optional)\n"
        "PINECONE_CLOUD (optional)\nPINECONE_REGION (optional)",
        language="text"
    )

q = st.text_input("Ask a question about the reviews", value="What themes come up about service speed?")
go = st.button("Run analysis", type="primary")

def render_theme(theme_obj: dict):
    st.subheader(theme_obj.get("theme", "Theme"))
    st.write(f"**Sentiment:** `{theme_obj.get('sentiment', 'mixed')}`")
    evidence = theme_obj.get("evidence", []) or []
    if evidence:
        st.write("**Evidence**")
        for e in evidence:
            cid = e.get("chunk_id", "")
            excerpt = e.get("excerpt", "")
            st.markdown(f"- `{cid}` — {excerpt}")
    else:
        st.write("_No evidence returned._")

def render_issue_list(title: str, issues: list, key_prefix: str):
    st.subheader(title)
    if not issues:
        st.write("_None._")
        return
    for i, it in enumerate(issues):
        issue = it.get("issue", "")
        ids = it.get("evidence_chunk_ids", []) or []
        ucnt = it.get("unique_review_count")
        st.markdown(f"**{i+1}. {issue}**")
        if ucnt is not None:
            st.write(f"Unique review count: `{ucnt}`")
        if ids:
            st.write("Evidence chunk IDs:")
            st.code("\n".join(ids))

def render_ops_recs(recs: list):
    st.subheader("Ops recommendations")
    if not recs:
        st.write("_None._")
        return
    for i, r in enumerate(recs):
        rec = r.get("recommendation", "")
        ids = r.get("grounding_chunk_ids", []) or []
        st.markdown(f"**{i+1}. {rec}**")
        if ids:
            st.caption("Grounding chunk IDs")
            st.code("\n".join(ids))

def render_sms(sms: dict):
    st.subheader("Draft SMS messages")
    msgs = (sms or {}).get("messages", []) or []
    counts = (sms or {}).get("character_counts", []) or []
    if not msgs:
        st.write("_None._")
        return
    for i, m in enumerate(msgs):
        c = counts[i] if i < len(counts) else len(m)
        st.markdown(f"**Message {i+1}** ({c} chars)")
        st.code(m)

if go:
    with st.spinner("Running retrieval + grounded generation..."):
        out = generate_grounded_response(
            query=q,
            top_k=int(top_k),
            min_recurring_reviews=int(min_recurring),
            include_debug=bool(include_debug),
        )

    if not isinstance(out, dict):
        st.error("Unexpected output type (expected dict).")
        st.write(out)
        st.stop()

    # Top summary
    st.success("Done.")
    st.subheader("Answer summary")
    st.write(out.get("answer_summary", ""))

    # Overall sentiment
    st.subheader("Overall sentiment")
    osent = out.get("overall_sentiment") or {}
    st.write(f"**Label:** `{osent.get('label', 'mixed')}`")
    st.write(osent.get("rationale", ""))

    col1, col2 = st.columns(2)

    with col1:
        st.header("Themes")
        themes = out.get("top_themes", []) or []
        if not themes:
            st.write("_No themes returned._")
        else:
            for th in themes:
                st.divider()
                render_theme(th)

    with col2:
        render_issue_list("Recurring issues", out.get("recurring_issues", []) or [], "rec")
        st.divider()
        render_issue_list("Isolated issues", out.get("isolated_issues", []) or [], "iso")

    st.divider()
    render_sms(out.get("sms_draft") or {})
    st.divider()
    render_ops_recs(out.get("ops_recommendations", []) or [])

    if include_debug:
        st.divider()
        st.subheader("Debug")
        st.json(out.get("debug", {}))

        # Show full JSON for copy/paste
        st.subheader("Raw JSON output")
        st.code(json.dumps(out, ensure_ascii=False, indent=2), language="json")
