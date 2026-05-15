"""
Day 12 — Reranking retrieved chunks.

Problem with retrieval:
  top_k=5 returns 5 chunks ordered by embedding similarity.
  Embedding similarity ≠ true relevance.
  A chunk can be semantically similar to the query
  without actually answering it.

Reranking:
  After retrieving top_k candidates, run a second model
  (cross-encoder) that jointly evaluates query + document together.
  Cross-encoder is slower but more accurate than bi-encoder.
  Use it on a small candidate set (top 10-20), not all documents.

Bi-encoder (what you use for retrieval):
  embed(query) → vector A
  embed(document) → vector B
  similarity = cosine(A, B)
  Fast: embeddings computed independently and cached.
  Less accurate: query and document processed separately.

Cross-encoder (what you use for reranking):
  score = model(query + "[SEP]" + document)
  Slow: must process query+document pair together, cannot cache.
  More accurate: model sees both at the same time, understands context.

Production pattern:
  Retrieve top 20 with bi-encoder (fast) → rerank top 20 with cross-encoder → return top 3
"""

import os
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import numpy as np

load_dotenv()

DOCUMENTS = [
    {"id": "d01", "text": "The Starter plan costs 999 rupees per month and includes 100 AI queries per day."},
    {"id": "d02", "text": "The Professional plan costs 2999 rupees per month with unlimited queries and priority support."},
    {"id": "d03", "text": "The Enterprise plan is custom priced with dedicated infrastructure and SLA guarantees."},
    {"id": "d04", "text": "All plans come with a 14-day free trial. No credit card required for trial."},
    {"id": "d05", "text": "Refunds are available within 30 days of first payment for annual plans."},
    {"id": "d06", "text": "Monthly plans can be cancelled anytime but are not eligible for partial refunds."},
    {"id": "d07", "text": "Our API supports REST and WebSocket connections."},
    {"id": "d08", "text": "Rate limits are 10 requests per second for Starter, 50 for Professional."},
    {"id": "d09", "text": "Maximum document size for upload is 10MB. Supported formats are PDF and TXT."},
    {"id": "d10", "text": "Response latency SLA is under 2 seconds for 95th percentile on Professional plan."},
    {"id": "d11", "text": "Starter plan support is via email with 48-hour response time."},
    {"id": "d12", "text": "Professional plan includes live chat support with 4-hour response time."},
    {"id": "d13", "text": "Enterprise customers get a dedicated support engineer and 1-hour response SLA."},
]


@dataclass
class RankedResult:
    doc_id: str
    text: str
    initial_rank: int
    initial_score: float
    rerank_score: float = 0.0
    final_rank: int = 0


class BiEncoderRetriever:
    """Standard vector retrieval — fast, used for initial candidate set."""

    def __init__(self, documents: list[dict]):
        self._docs = documents
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._embeddings = self._model.encode(
            [d["text"] for d in documents],
            normalize_embeddings=True,
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[RankedResult]:
        query_vec = self._model.encode(query, normalize_embeddings=True)
        scores = np.dot(self._embeddings, query_vec)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RankedResult(
                doc_id=self._docs[idx]["id"],
                text=self._docs[idx]["text"],
                initial_rank=rank + 1,
                initial_score=round(float(score), 4),
            )
            for rank, (idx, score) in enumerate(ranked)
        ]

class CrossEncoderReranker:
    """
    Reranks candidates using a cross-encoder model.
    Cross-encoder: processes (query, document) pair jointly.
    Output: relevance score — higher is more relevant.

    Model: ms-marco-MiniLM-L-6-v2
    Trained on Microsoft MARCO passage retrieval dataset.
    Specifically designed to judge query-passage relevance.
    """

    def __init__(self):
        print("  Loading cross-encoder reranker...")
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, candidates: list[RankedResult], top_k: int = 3) -> list[RankedResult]:
        """
        Score all candidates with cross-encoder.
        Return top_k by rerank score.
        """
        # Build (query, document) pairs
        pairs = [(query, c.text) for c in candidates]

        # Cross-encoder scores all pairs
        scores = self._model.predict(pairs)

        # Assign scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = round(float(score), 4)

        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)[:top_k]

        # Assign final ranks
        for i, r in enumerate(reranked):
            r.final_rank = i + 1

        return reranked

class RerankedRAG:
    """Full pipeline: retrieve → rerank → answer."""

    def __init__(self):
        self._retriever = BiEncoderRetriever(DOCUMENTS)
        self._reranker = CrossEncoderReranker()

    def search(
        self,
        query: str,
        initial_k: int = 8,
        final_k: int = 3,
    ) -> dict:
        # Stage 1: broad retrieval
        candidates = self._retriever.retrieve(query, top_k=initial_k)

        # Stage 2: precise reranking
        reranked = self._reranker.rerank(query, candidates, top_k=final_k)

        return {"candidates": candidates, "reranked": reranked}


def print_reranking_comparison(query: str, result: dict):
    """Show before/after reranking for a query."""
    print(f"\n{'='*70}")
    print(f"Query: {query!r}")
    print(f"{'='*70}")

    candidates = result["candidates"]
    reranked = result["reranked"]

    print(f"\nBEFORE reranking (bi-encoder top 5):")
    for r in candidates[:5]:
        print(f"  [{r.initial_rank}] score={r.initial_score:.4f} | {r.text[:70]}")

    print(f"\nAFTER reranking (cross-encoder top 3):")
    for r in reranked:
        rank_change = r.initial_rank - r.final_rank
        change_str = f"↑{rank_change}" if rank_change > 0 else (f"↓{abs(rank_change)}" if rank_change < 0 else "=")
        print(f"  [{r.final_rank}] rerank={r.rerank_score:.4f} initial_rank={r.initial_rank} ({change_str}) | {r.text[:70]}")


def main():
    print("Initialising retriever and reranker...")
    pipeline = RerankedRAG()

    queries = [
        "What is the maximum file size I can upload?",
        "How long do I have to request a refund?",
        "What support response time do I get with Professional?",
        "Which plan has the fastest API rate limit?",
    ]

    for query in queries:
        result = pipeline.search(query, initial_k=8, final_k=3)
        print_reranking_comparison(query, result)

    # Demonstrate where reranking makes the biggest difference
    print(f"\n{'='*70}")
    print("KEY OBSERVATION")
    print("="*70)
    print("""
Reranking matters most when:
- Initial retrieval fetches semantically similar but contextually wrong chunks
- Query has specific intent that embedding similarity cannot capture
- Multiple documents are equally similar to query vector but differ in actual relevance

Example: "I want a refund for my monthly subscription"
- Bi-encoder might rank "annual plan refund" chunk highly (semantic: both about refunds)
- Cross-encoder will correctly rank "monthly plans not eligible" higher (actual relevance)

In your results above, look for cases where final_rank != initial_rank.
Those are the queries where reranking improved results.
    """)


if __name__ == "__main__":
    main()