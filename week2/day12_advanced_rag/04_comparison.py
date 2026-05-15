"""
Day 12 — Final comparison: all 4 retrieval strategies on the same queries.

Strategies:
1. Standard vector (baseline — what you built in Week 1)
2. BM25 only (keyword)
3. Hybrid RRF (BM25 + vector)
4. Hybrid + reranking (best quality, highest cost)

This table is your resume talking point.
"I compared 4 retrieval strategies and measured quality tradeoffs."
"""

import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

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

# Ground truth: for each query, which doc_id is the correct answer
EVAL_SET = [
    {"query": "rate limit for Starter plan per second", "correct_id": "d08"},
    {"query": "how much does Professional plan cost",   "correct_id": "d02"},
    {"query": "can I get refund on monthly plan",       "correct_id": "d06"},
    {"query": "what file formats are supported",        "correct_id": "d09"},
    {"query": "Enterprise support response time",       "correct_id": "d13"},
    {"query": "free trial duration",                    "correct_id": "d04"},
    {"query": "latency guarantee Professional",         "correct_id": "d10"},
]


def evaluate_strategy(strategy_name, search_fn, eval_set, top_k=3):
    """
    Measure: does the correct document appear in top_k results?
    Returns: hit_rate (% of queries where correct doc is in top_k)
    """
    hits = 0
    latencies = []

    for sample in eval_set:
        start = time.perf_counter()
        results = search_fn(sample["query"], top_k=top_k)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        result_ids = [r["id"] for r in results]
        if sample["correct_id"] in result_ids:
            hits += 1

    hit_rate = hits / len(eval_set)
    avg_latency = sum(latencies) / len(latencies)

    return {
        "strategy": strategy_name,
        "hit_rate": round(hit_rate, 3),
        "hits": hits,
        "total": len(eval_set),
        "avg_latency_ms": round(avg_latency, 1),
    }


def main():
    print("Loading models...")
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Precompute embeddings
    doc_texts = [d["text"] for d in DOCUMENTS]
    doc_ids = [d["id"] for d in DOCUMENTS]
    embeddings = bi_encoder.encode(doc_texts, normalize_embeddings=True)

    # BM25 index
    tokenised = [t.lower().split() for t in doc_texts]
    bm25 = BM25Okapi(tokenised)

    # Strategy 1: Vector only
    def vector_search(query, top_k):
        q_vec = bi_encoder.encode(query, normalize_embeddings=True)
        scores = np.dot(embeddings, q_vec)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": doc_ids[i], "score": float(s)} for i, s in ranked]

    # Strategy 2: BM25 only
    def bm25_search(query, top_k):
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": doc_ids[i], "score": float(s)} for i, s in ranked if s > 0]

    # Strategy 3: Hybrid RRF
    def hybrid_search(query, top_k, k_rrf=60):
        bm25_scores = bm25.get_scores(query.lower().split())
        q_vec = bi_encoder.encode(query, normalize_embeddings=True)
        vec_scores = np.dot(embeddings, q_vec)

        bm25_ranks = {doc_ids[i]: r + 1 for r, (i, _) in enumerate(sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True))}
        vec_ranks = {doc_ids[i]: r + 1 for r, (i, _) in enumerate(sorted(enumerate(vec_scores), key=lambda x: x[1], reverse=True))}

        rrf = {}
        for did in doc_ids:
            rrf[did] = 0.5 * (1 / (k_rrf + bm25_ranks.get(did, 999))) + \
                       0.5 * (1 / (k_rrf + vec_ranks.get(did, 999)))

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": did, "score": score} for did, score in ranked]

    # Strategy 4: Hybrid + reranking
    def hybrid_rerank_search(query, top_k):
        candidates = hybrid_search(query, top_k=min(8, len(DOCUMENTS)))
        pairs = [(query, next(d["text"] for d in DOCUMENTS if d["id"] == c["id"])) for c in candidates]
        rerank_scores = cross_encoder.predict(pairs)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": c["id"], "score": float(s)} for c, s in reranked]

    # Run evaluation
    print("\nEvaluating all strategies...")
    strategies = [
        ("Vector only (baseline)", vector_search),
        ("BM25 only (keyword)", bm25_search),
        ("Hybrid RRF", hybrid_search),
        ("Hybrid + Reranking", hybrid_rerank_search),
    ]

    results = []
    for name, fn in strategies:
        print(f"  Testing: {name}...")
        r = evaluate_strategy(name, fn, EVAL_SET, top_k=3)
        results.append(r)

    # Print comparison table
    print(f"\n{'='*70}")
    print("RETRIEVAL STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':<28} {'Hit Rate':>10} {'Hits':>8} {'Avg Latency':>13}")
    print(f"{'-'*70}")
    for r in results:
        print(
            f"{r['strategy']:<28} "
            f"{r['hit_rate']:>9.1%} "
            f"{r['hits']:>4}/{r['total']:<4} "
            f"{r['avg_latency_ms']:>11.1f}ms"
        )
    print(f"{'='*70}")

    # Analysis
    baseline_hr = results[0]["hit_rate"]
    best = max(results, key=lambda x: x["hit_rate"])
    print(f"\nBaseline hit rate: {baseline_hr:.1%}")
    print(f"Best strategy: {best['strategy']} at {best['hit_rate']:.1%}")
    improvement = (best["hit_rate"] - baseline_hr) / baseline_hr * 100 if baseline_hr > 0 else 0
    print(f"Improvement over baseline: +{improvement:.1f}%")

    print(f"\nLatency vs quality tradeoff:")
    for r in results:
        tokens_per_query = "cheap" if r["avg_latency_ms"] < 100 else ("medium" if r["avg_latency_ms"] < 500 else "expensive")
        print(f"  {r['strategy']}: {r['avg_latency_ms']:.0f}ms ({tokens_per_query})")


if __name__ == "__main__":
    main()