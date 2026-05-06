"""
Streamlit frontend — chat interface for DocuAgent.
Run: streamlit run ui/app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="DocuAgent", page_icon="🤖", layout="wide")
st.title("DocuAgent")
st.caption("Document-aware AI assistant — Week 1 portfolio project")

# ── Sidebar: document upload

with st.sidebar:
    st.header("Documents")

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API connected — {health['chunks_in_store']} chunks indexed")
    except Exception:
        st.error("API not reachable. Start with: uvicorn api.app:app --reload")
        st.stop()

    # File upload
    uploaded = st.file_uploader("Upload a document (.txt)", type=["txt"])
    if uploaded and st.button("Ingest document"):
        with st.spinner("Ingesting..."):
            resp = requests.post(
                f"{API_URL}/ingest/file",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/plain")},
            )
            if resp.ok:
                data = resp.json()
                st.success(f"Ingested {data['chunks_created']} chunks from {data['doc_name']}")
            else:
                st.error(f"Failed: {resp.text}")

    if st.button("Clear all documents", type="secondary"):
        resp = requests.post(f"{API_URL}/clear")
        if resp.ok:
            st.success("Documents cleared")

# ── Main: chat interface ──────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("meta"):
            with st.expander("Details"):
                st.json(msg["meta"])

# Input
if question := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()

                    if data["is_blocked"]:
                        st.error(f"Blocked: {data['answer']}")
                        answer = data["answer"]
                    else:
                        st.write(data["answer"])
                        answer = data["answer"]

                        # Show metadata
                        col1, col2 = st.columns(2)
                        col1.metric("Tool calls", data["tool_call_count"])
                        col2.metric("Latency", f"{data['latency_ms']:.0f}ms")

                        if data["sources"]:
                            with st.expander(f"Sources ({len(data['sources'])})"):
                                for s in data["sources"]:
                                    st.markdown(f"**Query:** {s['query']}")
                                    st.text(s["result_preview"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": {
                            "tool_calls": data.get("tool_call_count", 0),
                            "latency_ms": data.get("latency_ms", 0),
                            "blocked": data.get("is_blocked", False),
                        },
                    })
                else:
                    st.error(f"API error: {resp.text}")
            except requests.Timeout:
                st.error("Request timed out. The agent is taking too long.")
            except Exception as e:
                st.error(f"Error: {e}")