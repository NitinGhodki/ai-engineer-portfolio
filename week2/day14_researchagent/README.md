1. Architecture diagram (ASCII):
   User → Streamlit UI → FastAPI → LangGraph Supervisor
                                        ↓         ↑
                              Researcher  Writer  Critic
                                   ↓
                              HybridRAG (BM25 + Vector)

2. Setup instructions from scratch

3. Screenshot of Streamlit UI showing:
   - A query answered
   - Execution log expanded
   - RAGAS scores visible in sidebar

4. Screenshot of all 5 curl commands and responses

5. Section: "What I built in Week 2"
   - LangGraph fundamentals and human-in-the-loop
   - RAG evaluation with RAGAS
   - LLM observability and cost tracking
   - Advanced RAG (hybrid search, reranking, HyDE)
   - 4-agent multi-agent system
   - Deployed production application

6. Section: "What I would improve" — 5 GitHub Issues
   Example issues:
   - Add Redis cache for rerank scores
   - Replace MemorySaver with SqliteSaver for multi-worker support
   - Add streaming endpoint for real-time agent updates
   - Implement per-user session isolation
   - Add cross-encoder reranking to HybridRAG