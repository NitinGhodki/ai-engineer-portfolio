"""
Day 12 — Hybrid search: BM25 + vector similarity combined.

BM25 = keyword-based ranking algorithm (used by Elasticsearch, Lucene)
       Fast. Exact term matching. No semantic understanding.

Vector = semantic similarity via embeddings
         Understands meaning. Misses exact keyword matches.

Hybrid = run both, combine scores with weighted average.
         Best of both. Handles exact terms AND semantic meaning.

When hybrid wins over pure vector:
- Technical queries with specific terms ("rate limit", "OAuth 2.0")
- Numerical queries ("10MB limit", "30-day policy")
- Named entity queries ("Starter plan", "Enterprise tier")
- Code-related queries (function names, error codes)
"""


import os
import math
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()


# ── Document corpus
DOCUMENTS = [
    {"id": "d01", "text": "The Starter plan costs 999 rupees per month and includes 100 AI queries per day.", "category": "pricing"},
    {"id": "d02", "text": "The Professional plan costs 2999 rupees per month with unlimited queries and priority support.", "category": "pricing"},
    {"id": "d03", "text": "The Enterprise plan is custom priced with dedicated infrastructure and SLA guarantees.", "category": "pricing"},
    {"id": "d04", "text": "All plans come with a 14-day free trial. No credit card required for trial.", "category": "refund"},
    {"id": "d05", "text": "Refunds are available within 30 days of first payment for annual plans.", "category": "refund"},
    {"id": "d06", "text": "Monthly plans can be cancelled anytime but are not eligible for partial refunds.", "category": "refund"},
    {"id": "d07", "text": "Our API supports REST and WebSocket connections for all plan tiers.", "category": "technical"},
    {"id": "d08", "text": "Rate limits are 10 requests per second for Starter, 50 requests per second for Professional.", "category": "technical"},
    {"id": "d09", "text": "Maximum document size for upload is 10MB. Supported formats are PDF and TXT.", "category": "technical"},
    {"id": "d10", "text": "Response latency SLA is under 2 seconds for 95th percentile on Professional plan.", "category": "technical"},
    {"id": "d11", "text": "Starter plan support is via email with 48-hour response time.", "category": "support"},
    {"id": "d12", "text": "Professional plan includes live chat support with 4-hour response time.", "category": "support"},
    {"id": "d13", "text": "Enterprise customers get a dedicated support engineer and 1-hour response SLA.", "category": "support"},
]

# ── Result dataclass
@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float
    rank: int
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None


# ── BM25 Search
class BM25Search:
    """
    BM25 (Best Match 25) — the gold standard keyword search algorithm.
    Used internally by Elasticsearch, Solr, and many search engines.

    How it works:
    - Tokenises query and documents into words
    - Scores based on: term frequency in document + inverse document frequency
    - TF: how often query term appears in this document
    - IDF: how rare the term is across ALL documents (rare = more informative)
    - Applies length normalisation (long docs don't get unfair advantage)
    """


    def __init__(self, documents: list[dict]):
        self._docs = documents
        # Tokenise: lowercase, split on spaces
        # In production use a proper tokeniser (nltk, spacy) for better results
        tokenised = [doc["text"].lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenised)


    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True,)[:top_k]

        return [
            SearchResult(
                doc_id=self._docs[idx]["id"],
                text=self._docs[idx]["text"],
                score=round(float(score), 4),
                rank= rank + 1,
                bm25_score=round(float(score), 4),
            )
            for rank, (idx, score) in enumerate(ranked)
            if score > 0        
        ]

# ── Vector Search
class VectorSearch:
    """
    Semantic similarity search using sentence-transformers.
    Embeds all documents once at init, embeds query at search time.
    Computes cosine similarity between query vector and all doc vectors.
    """

    def __init__(self, documents: list[dict]):
        self._docs = documents
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Embedding documents for vector search...")
        self._embeddings = self._model.encode(
            [doc["text"] for doc in documents],
            normalize_embeddings=True,  # unit vectors → dot product = cosine similarity
        ) 

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vec = self._model.encode(query, normalize_embeddings=True)
        # Cosine similarity = dot product (vectors are already normalised)
        scores = np.dot(self._embeddings, query_vec)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                doc_id=self._docs[idx]["id"],
                text=self._docs[idx]["text"],
                score=round(float(score), 4),
                rank=rank + 1,
                vector_score=round(float(score), 4),
            )
            for rank, (idx, score) in enumerate(ranked)
        ]


# ── Hybrid Search 

class HybridSearch:
    """
    Combines BM25 and vector scores using Reciprocal Rank Fusion (RRF).

    Why RRF instead of weighted average?
    - BM25 scores and vector scores have completely different scales
      BM25: 0 to ~15. Vector: 0.0 to 1.0. You cannot average these directly.
    - RRF normalises by RANK not score:
      RRF_score = 1/(k + rank_in_bm25) + 1/(k + rank_in_vector)
      where k=60 is a constant that smooths the ranking
    - A document ranked #1 in both gets the highest combined score
    - A document ranked #1 in one and #5 in other still scores well
    - A document only found by one method still contributes

    Alternative: Weighted average of normalised scores (simpler but less robust)
    """

    def __init__(self, documents: list[dict], alpha: float = 0.5):
        """
        alpha: weight for vector search (0.0 = pure BM25, 1.0 = pure vector)
        0.5 = equal weight — good starting point for most domains
        """
        self._docs = documents
        self._bm25 = BM25Search(documents)
        self._vector = VectorSearch(documents)
        self._alpha = alpha

    def search(self, query: str, top_k: int = 5, k_rrf: int = 60) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion search.
        k_rrf=60 is the standard constant from the original RRF paper.
        """
        # Get results from both systems — fetch more than top_k to allow reranking
        bm25_results = self._bm25.search(query, top_k=len(self._docs))
        vector_results = self._vector.search(query, top_k=len(self._docs))

        # Build rank lookup: doc_id → rank in each system
        bm25_ranks = {r.doc_id: r.rank for r in bm25_results}
        vector_ranks = {r.doc_id: r.rank for r in vector_results}
        bm25_scores = {r.doc_id: r.bm25_score for r in bm25_results}
        vector_scores = {r.doc_id: r.vector_score for r in vector_results}

        # Compute RRF score for every document
        all_doc_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        rrf_scores = {}

        for doc_id in all_doc_ids:
            # If doc not in a system, use worst possible rank
            bm25_rank = bm25_ranks.get(doc_id, len(self._docs) + 1)
            vector_rank = vector_ranks.get(doc_id, len(self._docs) + 1)

            # RRF formula — weighted by alpha
            bm25_contribution = (1 - self._alpha) * (1 / (k_rrf + bm25_rank))
            vector_contribution = self._alpha * (1 / (k_rrf + vector_rank))
            rrf_scores[doc_id] = bm25_contribution + vector_contribution

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build results
        doc_lookup = {doc["id"]: doc["text"] for doc in self._docs}
        return [
            SearchResult(
                doc_id=doc_id,
                text=doc_lookup.get(doc_id, ""),
                score=round(score, 6),
                rank=rank + 1,
                bm25_score=bm25_scores.get(doc_id),
                vector_score=vector_scores.get(doc_id),
            )
            for rank, (doc_id, score) in enumerate(sorted_docs)
        ]


# ── Comparison runner 

def compare_search_methods(query: str, bm25: BM25Search, vector: VectorSearch, hybrid: HybridSearch, top_k: int = 3):
    """Run all three methods and print side-by-side comparison."""
    print(f"\n{'='*70}")
    print(f"Query: {query!r}")
    print(f"{'='*70}")

    bm25_results = bm25.search(query, top_k=top_k)
    vector_results = vector.search(query, top_k=top_k)
    hybrid_results = hybrid.search(query, top_k=top_k)

    methods = [
        ("BM25 (keyword)", bm25_results),
        ("Vector (semantic)", vector_results),
        ("Hybrid (RRF)", hybrid_results),
    ]

    for method_name, results in methods:
        print(f"\n  {method_name}:")
        if not results:
            print("    (no results)")
            continue
        for r in results:
            bm25_info = f" bm25={r.bm25_score}" if r.bm25_score else ""
            vec_info = f" vec={r.vector_score}" if r.vector_score else ""
            print(f"    [{r.rank}] score={r.score:.4f}{bm25_info}{vec_info}")
            print(f"        {r.text[:80]}")


def main():
    print("Building search indices...")
    bm25 = BM25Search(DOCUMENTS)
    vector = VectorSearch(DOCUMENTS)
    hybrid = HybridSearch(DOCUMENTS, alpha=0.5)

    test_queries = [
        # Keyword-heavy — BM25 should win
        "rate limit Starter plan requests per second",
        "10MB document upload PDF",
        # Semantic — vector should win
        "how much does it cost to use the service",
        "what happens if I want to cancel",
        # Mixed — hybrid should win
        "Professional plan live chat 4 hour",
        "how fast will I get support help",
    ]

    for query in test_queries:
        compare_search_methods(query, bm25, vector, hybrid)

    # Alpha sensitivity test — show how alpha changes results
    print(f"\n{'='*70}")
    print("ALPHA SENSITIVITY TEST")
    print(f"{'='*70}")
    print("Query: 'rate limit Starter plan requests per second'")
    print("(This is a keyword-heavy query — lower alpha = more BM25 = better)")

    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        h = HybridSearch(DOCUMENTS, alpha=alpha)
        results = h.search("rate limit Starter plan requests per second", top_k=1)
        top = results[0] if results else None
        label = "←BM25 dominant" if alpha < 0.4 else ("←Vector dominant" if alpha > 0.6 else "←balanced")
        print(f"  alpha={alpha}: top result = {top.text[:60]!r} {label}" if top else f"  alpha={alpha}: no results")


if __name__ == "__main__":
    main()