"""
FastAPI app — HTTP interface to DocuAgent.
Endpoints: health, ingest, query, clear, history.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from core.rag import RAGPipeline
from core.agent import DocuAgent

# ── App state

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initializing RAG pipeline...")
    rag = RAGPipeline(persist_dir="./chroma_db")

    print("[STARTUP] Loading sample document...")
    try:
        chunks = rag.ingest_file("data/sample_doc.txt")
        print(f"[STARTUP] Sample doc ingested: {chunks} chunks")
    except FileNotFoundError:
        print("[STARTUP] No sample doc found - skipping")

    print("[STARTUP] Building DocuAgent...........")
    agent = DocuAgent(rag=rag)

    state["rag"] = rag
    state["agent"] = agent
    print("[STARTUP] DocuAgent ready. \n")

    yield

    state.clear()
    print("[SHUTDOWN] CLeaned up........")


app = FastAPI(
    title="DocuAgent API",
    description="Document-aware AI agent with RAG + tools",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)

class QueryResponse(BaseModel):
    question: str
    answer: str
    tool_call_count: int
    sources: list[dict]
    is_blocked: bool
    latency_ms: float

class IngestResponse(BaseModel):
    doc_name: str
    chunks_created: int
    total_chunks: int


# ── Endpoints
@app.get("/health")
def health():
    rag: RAGPipeline = state.get("rag")

    return{
        "status": "healthy",
        "chunks_in_store": rag.chunk_count() if rag else 0,
        "agent_ready": "agent" in state,
    }

@app.post("/ingest/text")
def ingest_text(doc_name: str, text: str) -> IngestResponse:
    """Ingest raw text as a document."""
    rag: RAGPipeline = state["rag"]
    chunks = rag.ingest_text(text, doc_name=doc_name)
    return IngestResponse(
        doc_name=doc_name,
        chunks_created=chunks,
        total_chunks=rag.chunk_count(),
    )

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and ingest a text file."""
    rag: RAGPipeline = state["rag"]

    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .md files supported",
        )

    content = await file.read()
    text = content.decode("utf-8")
    chunks = rag.ingest_text(text, doc_name=file.filename)

    return IngestResponse(
        doc_name=file.filename,
        chunks_created=chunks,
        total_chunks=rag.chunk_count(),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Ask a question. Agent searches documents and uses tools."""
    agent: DocuAgent = state["agent"]

    start = time.perf_counter()
    result = agent.query(request.question)
    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        tool_call_count=result.get("tool_call_count", 0),
        sources=result["sources"],
        is_blocked=result["is_blocked"],
        latency_ms=round(latency_ms, 2),
    )

@app.post("/clear")
def clear_documents():
    """Clear all ingested documents."""
    rag: RAGPipeline = state["rag"]
    rag.clear()
    return {"status": "cleared", "chunks_remaining": rag.chunk_count()}