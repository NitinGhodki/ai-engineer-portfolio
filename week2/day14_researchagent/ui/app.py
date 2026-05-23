import streamlit as st
import requests

API = "http://localhost:8000"

st.set_page_config(page_title="ResearchAgent", page_icon="🔬", layout="wide")
st.title("ResearchAgent")
st.caption("Multi-agent document intelligence — Week 2 portfolio project")

# ── Sidebar 

with st.sidebar:
    st.header("System Status")
    try:
        h = requests.get(f"{API}/health", timeout=3).json()
        st.success(f"API connected")
        st.metric("Chunks indexed", h.get("chunks_indexed", 0))
        st.metric("Queries recorded", h.get("queries_recorded", 0))
    except Exception:
        st.error("API not reachable. Run: uvicorn api.app:app --reload")
        st.stop()

    st.divider()
    st.header("Upload Document")
    uploaded = st.file_uploader("Add to knowledge base (.txt)", type=["txt"])
    if uploaded and st.button("Ingest"):
        with st.spinner("Ingesting..."):
            r = requests.post(
                f"{API}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/plain")},
            )
            if r.ok:
                d = r.json()
                st.success(f"Ingested {d['chunks']} chunks from {d['filename']}")
            else:
                st.error(r.text)

    st.divider()
    st.header("RAGAS Evaluation")
    n_eval = st.slider("Samples to evaluate", 1, 10, 3)
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            r = requests.post(f"{API}/evaluate?n_samples={n_eval}")
            if r.ok:
                scores = r.json()
                st.metric("Faithfulness", f"{scores.get('faithfulness', 0):.3f}")
                st.metric("Answer Relevancy", f"{scores.get('answer_relevancy', 0):.3f}")
                st.metric("Context Recall", f"{scores.get('context_recall', 0):.3f}")
                st.metric("Context Precision", f"{scores.get('context_precision', 0):.3f}")
                st.metric("Overall", f"{scores.get('overall', 0):.3f}")
            else:
                st.error(r.json().get("detail", r.text))

# ── Main chat 

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("meta"):
            with st.expander("Execution details"):
                for step in msg["meta"].get("execution_log", []):
                    st.text(step)
                col1, col2 = st.columns(2)
                col1.metric("Latency", f"{msg['meta'].get('latency_ms', 0):.0f}ms")
                col2.metric("Session", msg["meta"].get("session_id", ""))

col1, col2 = st.columns([3, 1])
with col1:
    question = st.chat_input("Ask a question about your documents...")
with col2:
    fmt = st.selectbox("Format", ["paragraph", "bullets", "table"], label_visibility="collapsed")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            try:
                r = requests.post(
                    f"{API}/query",
                    json={"question": question, "output_format": fmt},
                    timeout=120,
                )
                if r.ok:
                    data = r.json()
                    st.write(data["answer"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "meta": data,
                    })
                else:
                    st.error(f"Error: {r.text}")
            except requests.Timeout:
                st.error("Request timed out (agents took >120s)")
            except Exception as e:
                st.error(f"Error: {e}")