"""
FastAPI wrapper around the LangChain RAG pipeline.
This makes your RAG system callable over HTTP — 
the same way it would work in a real product.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rag_pipeline import build_rag_pipeline

# app State
pipeline_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the RAG pipeline once when the server starts.
    Reusing it across requests is critical — 
    rebuilding on every request would take 10-30 seconds each time.
    """
    print("Starting up: building pipeline")
    chain, vectorstore = build_rag_pipeline()
    pipeline_state["chain"] = chain
    pipeline_state["vectorstore"] = vectorstore

    print("RAG pipeline ready. Server accepting requests.")
    yield
    # Cleanup on shutdown
    pipeline_state.clear()
    print("Server shut down.")

app =  FastAPI(
    title="RAG API",
    description="Langchain RAG pipeline exposed via FastAPI",
    version="1.0.0",
    lifespan=lifespan   
)


#  Request / Response schemas

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to answer using the document store",
        examples=["What is RAG?"],
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrive",
    )

class SourceChunk(BaseModel):
    source: str
    preview: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    latency_ms: float


# Endpoints

@app.get("/health")
def health_check():
    """Basic health check — always implement this in production APIs."""
    return {
        "status": "healthy",
        "pipeline_loaded": "chain" in pipeline_state,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main RAG query endpoint.
    
    POST /query
    {
        "question": "What is the refund policy?",
        "top_k": 3
    }
    """
    if "chain" not in pipeline_state:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    chain = pipeline_state["chain"]
    vectorstore = pipeline_state["vectorstore"]

    start = time.perf_counter()

    # Get answer from RAG chain
    answer = chain.invoke(request.question)

    # Get source chunks for citation
    retrieved_docs = vectorstore.similarity_search_with_score(
        request.question,
        k=request.top_k,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    sources = [
        SourceChunk(
            source=doc.metadata.get("source", "unknown").split("/")[-1],
            preview=doc.page_content[:120],
            relevance_score=round(float(1 - score), 4),
        )
        for doc, score in retrieved_docs
    ]

    return QueryResponse(
        question=request.question,
        answer=answer.strip(),
        sources=sources,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """
    Streaming endpoint — returns tokens as they're generated.
    Uses Server-Sent Events (SSE) format.
    
    This is the answer to Day 1 Q3:
    "What protocol sends streaming chunks to a browser?"
    → Server-Sent Events (text/event-stream content type)
    """
    if "chain" not in pipeline_state:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    chain = pipeline_state["chain"]

    def generate():
        for chunk in chain.stream(request.question):
            if chunk:
                # SSE format: "data: <content>\n\n"
                yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sources")
def list_sources():
    """Show all ingested documents in the vector store."""
    if "vectorstore" not in pipeline_state:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    vectorstore = pipeline_state["vectorstore"]
    collection = vectorstore._collection
    results = collection.get()

    sources = list(set([
        meta.get("source", "unknown").split("/")[-1]
        for meta in results["metadatas"]
    ]))

    return {
        "total_chunks": len(results["ids"]),
        "documents": sources,
    }
