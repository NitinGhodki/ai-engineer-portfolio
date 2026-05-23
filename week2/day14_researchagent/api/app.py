import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from core.rag import HybridRAG
from core.evaluator import record_query, run_evaluation, get_history_count
from graph.workflow import build_workflow, WorkflowState

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Building RAG pipeline...")
    rag = HybridRAG(persist_dir="./chroma_db")

    # Load default knowledge base
    try:
        chunks = rag.ingest_file("data/knowledge_base.txt")
        print(f"[STARTUP] Knowledge base loaded: {chunks} chunks")
    except FileNotFoundError:
        print("[STARTUP] No default knowledge base found")

    workflow = build_workflow(rag)
    state["rag"] = rag
    state["workflow"] = workflow
    print("[STARTUP] ResearchAgent ready.\n")
    yield
    print("[END] ResearchAgent now stop.\n")
    state.clear()


app = FastAPI(
    title="ResearchAgent API",
    description="Multi-agent document intelligence system",
    version="2.0.0",
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
    output_format: str = Field(default="paragraph", pattern="^(paragraph|bullets|table)$")


class QueryResponse(BaseModel):
    question: str
    answer: str
    output_format: str
    execution_log: list[str]
    latency_ms: float
    session_id: str


# ── Endpoints 

@app.get("/health")
def health():
    rag: HybridRAG = state.get("rag")
    return {
        "status": "healthy",
        "chunks_indexed": rag.chunk_count() if rag else 0,
        "queries_recorded": get_history_count(),
        "workflow_ready": "workflow" in state,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if "workflow" not in state:
        raise HTTPException(status_code=503, detail="Workflow not ready")

    session_id = str(uuid.uuid4())[:8]
    workflow = state["workflow"]
    rag = state["rag"]

    start = time.perf_counter()

    initial: WorkflowState = {
        "user_request": request.question,
        "output_format": request.output_format,
        "research_findings": "",
        "written_draft": "",
        "critique_result": "",
        "revision_count": 0,
        "next_agent": "",
        "final_output": "",
        "session_id": session_id,
        "execution_log": [],
        "cost_summary": {},
    }

    result = workflow.invoke(initial)
    latency_ms = (time.perf_counter() - start) * 1000

    answer = result["final_output"]

    # Record for RAGAS evaluation
    contexts = [r["text"] for r in rag.search(request.question, top_k=3)]
    record_query(
        question=request.question,
        answer=answer,
        contexts=contexts,
    )

    return QueryResponse(
        question=request.question,
        answer=answer,
        output_format=request.output_format,
        execution_log=result.get("execution_log", []),
        latency_ms=round(latency_ms, 2),
        session_id=session_id,
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .txt and .md files supported")
    rag: HybridRAG = state["rag"]
    content = await file.read()
    chunks = rag.ingest(content.decode("utf-8"), doc_name=file.filename)
    return {"filename": file.filename, "chunks": chunks, "total": rag.chunk_count()}


@app.post("/evaluate")
def evaluate(n_samples: int = 5):
    """Run RAGAS evaluation on last n_samples queries."""
    if get_history_count() == 0:
        raise HTTPException(status_code=400, detail="No queries recorded yet")
    return run_evaluation(n_samples=min(n_samples, get_history_count()))


@app.get("/search")
def search(q: str, top_k: int = 3):
    """Direct hybrid search — useful for debugging retrieval."""
    rag: HybridRAG = state["rag"]
    results = rag.search(q, top_k=top_k)
    return {"query": q, "results": results}