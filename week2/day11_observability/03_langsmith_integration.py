"""
Day 11 — LangSmith integration.

LangSmith = hosted observability platform by LangChain team.
Free tier available. Gives you a UI for all your traces.

Setup:
1. Go to smith.langchain.com
2. Create a free account
3. Create a project called "ai-engineer-portfolio"
4. Get your API key from Settings
5. Add to .env:
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key_here
   LANGCHAIN_PROJECT=ai-engineer-portfolio

After running this file:
- Go to smith.langchain.com
- Open your project
- See every LLM call traced with full input/output
- Screenshot it for your portfolio
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith activates automatically when these env vars are set
# No code changes needed in your pipeline — it just works
TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def check_langsmith_setup():
    """Verify LangSmith is configured correctly."""
    print("="*55)
    print("LANGSMITH SETUP CHECK")
    print("="*55)

    checks = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGSMITH_TRACING"),
        "LANGCHAIN_API_KEY": "SET" if os.getenv("LANGSMITH_API_KEY") else "NOT SET",
        "LANGCHAIN_PROJECT": os.getenv("LANGSMITH_PROJECT", "default"),
    }

    all_good = True
    for key, value in checks.items():
        status = "✓" if value and value != "NOT SET" else "✗"
        print(f"  {status} {key}: {value}")
        if value is None or value == "NOT SET":
            all_good = False

    if all_good:
        print("\nLangSmith is configured. All traces will be sent automatically.")
    else:
        print("\nLangSmith not fully configured.")
        print("Add these to your .env file:")
        print("  LANGCHAIN_TRACING_V2=true")
        print("  LANGCHAIN_API_KEY=your_key_from_smith.langchain.com")
        print("  LANGCHAIN_PROJECT=ai-engineer-portfolio")

    return all_good


def run_traced_pipeline():
    """
    Run a RAG pipeline. If LangSmith is configured,
    every call is automatically traced — no code changes.
    This is the zero-instrumentation value of LangSmith.
    """
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.tracers.context import tracing_v2_enabled

    print("\n" + "="*55)
    print("RUNNING TRACED PIPELINE")
    print("="*55)

    # Build mini RAG
    docs = [
        Document(page_content="LangSmith is an observability platform for LLM applications. It traces every call."),
        Document(page_content="LangGraph is a framework for building stateful multi-actor LLM applications."),
        Document(page_content="RAGAS is an evaluation framework that measures RAG quality with four metrics."),
    ]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Chroma.from_documents(docs, embeddings, collection_name="langsmith_demo")
    retriever = store.as_retriever(search_kwargs={"k": 2})

    llm = ChatHuggingFace(
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=200,
            temperature=0.1,
        )
    )

    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY this context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    def fmt(d): return "\n".join(doc.page_content for doc in d)

    chain = (
        RunnableParallel({"context": retriever | fmt, "question": RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )

    questions = [
        "What is LangSmith used for?",
        "What metrics does RAGAS measure?",
        "What is LangGraph?",
    ]

    # Use tracing_v2_enabled context manager for explicit project naming
    # This works even without env vars — useful for testing
    with tracing_v2_enabled(project_name=os.getenv("LANGCHAIN_PROJECT", "ai-engineer-portfolio")):
        for q in questions:
            print(f"\nQ: {q}")
            answer = chain.invoke(q)
            print(f"A: {answer.strip()[:100]}")

    if TRACING_ENABLED:
        print("\n✓ All calls traced to LangSmith.")
        print(f"  View at: https://smith.langchain.com")
        print(f"  Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
        print("\nWhat to look for in LangSmith UI:")
        print("  - Each question = one trace")
        print("  - Inside each trace: retrieval call + LLM call")
        print("  - Click any LLM call: see full prompt + response + latency")
        print("  - This is what you screenshot for your portfolio")
    else:
        print("\nLangSmith not enabled — traces not sent.")
        print("Configure .env to see traces in UI.")


def demo_custom_metadata():
    """
    Add custom metadata to traces — useful for filtering in LangSmith UI.
    Tag traces by user, feature, experiment version, etc.
    """
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    from langchain_core.tracers.context import tracing_v2_enabled
    from langsmith import traceable

    print("\n" + "="*55)
    print("CUSTOM METADATA IN TRACES")
    print("="*55)

    llm = ChatHuggingFace(
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=200,
            temperature=0.1,
        )
    )

    # @traceable adds a named span to LangSmith trace
    # metadata dict appears in the LangSmith UI for filtering
    @traceable(name="rag_query", metadata={"version": "v1.2", "feature": "document_qa"})
    def traced_query(question: str, user_id: str) -> str:
        return llm.invoke(question)

    print("Running traced function with metadata...")
    result = traced_query(
        question="What is observability in AI systems?",
        user_id="user_nitin_001",
    )
    print(f"Result: {result.content[:100]!r}")
    print("\nIn LangSmith UI:")
    print("  - Filter by metadata.version = 'v1.2'")
    print("  - Filter by metadata.feature = 'document_qa'")
    print("  - Compare v1.2 vs v1.3 traces side by side")
    print("  - This is how you run A/B tests on prompts")


if __name__ == "__main__":
    setup_ok = check_langsmith_setup()
    run_traced_pipeline()
    if setup_ok:
        demo_custom_metadata()