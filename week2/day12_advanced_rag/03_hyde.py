"""
Day 12 — HyDE: Hypothetical Document Embedding

Problem with standard RAG:
  Query:    "what is the refund window for annual subscribers?"
  Document: "Refunds are available within 30 days of first payment for annual plans."

  The query and document don't look similar as text.
  Query has: "refund window", "annual subscribers"
  Document has: "available within 30 days", "annual plans"
  Embedding similarity is lower than it should be.

HyDE solution:
  Step 1: Ask LLM to generate a HYPOTHETICAL answer to the query
          (before looking at any documents)
  Step 2: Embed the hypothetical answer (not the query)
  Step 3: Use that embedding to search the document store

Why this works:
  Hypothetical answer looks like the actual document (both are "answer-shaped" text)
  The hypothetical answer uses similar vocabulary, sentence structure, and domain terms
  Embedding similarity between hypothetical answer and real document is HIGHER
  than between short query and long document

Tradeoff:
  + Better retrieval on question-style queries
  + Especially good for domain-specific terminology
  - Extra LLM call per query (cost + latency)
  - If LLM generates a wrong hypothetical, retrieval degrades
  - Not always better than standard — test with your eval dataset first
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from huggingface_hub import InferenceClient

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


class HyDERetriever:
    """
    Hypothetical Document Embedding retriever.
    Generates a fake answer first, then searches with it.
    """

    def __init__(self, documents: list[dict]):
        self._docs = documents
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._llm = InferenceClient(token=os.getenv("HF_API_KEY"))
        self._llm_model = token=os.getenv("Hugging_face_model")

        print("  Embedding document corpus...")
        self._embeddings = self._model.encode(
            [d["text"] for d in documents],
            normalize_embeddings=True,
        )

    def _generate_hypothetical_answer(self, query: str) -> str:
        """
        Ask LLM to generate a hypothetical answer WITHOUT seeing documents.
        The answer will be wrong (no real data), but its FORM and vocabulary
        will match real documents closely.
        """
        prompt = f"""Generate a short, factual-sounding answer to this question.
            The answer should sound like it comes from product documentation.
            Do not say you don't know. Just write a plausible-sounding answer.
            Keep it under 2 sentences.

            Question: {query}
            Answer:"""

        response = self._llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self._llm_model,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()

    def search_standard(self, query: str, top_k: int = 3) -> list[dict]:
        """Standard retrieval: embed the query directly."""
        query_vec = self._model.encode(query, normalize_embeddings=True)
        scores = np.dot(self._embeddings, query_vec)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "text": self._docs[idx]["text"],
                "score": round(float(score), 4),
                "rank": rank + 1,
                "method": "standard",
            }
            for rank, (idx, score) in enumerate(ranked)
        ]

    def search_hyde(self, query: str, top_k: int = 3) -> dict:
        """HyDE retrieval: embed the hypothetical answer instead."""
        hypothetical = self._generate_hypothetical_answer(query)
        print(f"  Hypothetical answer: {hypothetical!r}")

        # Embed the hypothetical answer (not the query)
        hyde_vec = self._model.encode(hypothetical, normalize_embeddings=True)
        scores = np.dot(self._embeddings, hyde_vec)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = [
            {
                "text": self._docs[idx]["text"],
                "score": round(float(score), 4),
                "rank": rank + 1,
                "method": "hyde",
            }
            for rank, (idx, score) in enumerate(ranked)
        ]
        return {"hypothetical_answer": hypothetical, "results": results}


def compare_hyde_vs_standard(query: str, retriever: HyDERetriever):
    """Run both methods and compare top results."""
    print(f"\n{'='*70}")
    print(f"Query: {query!r}")
    print("="*70)

    standard = retriever.search_standard(query, top_k=3)
    hyde_result = retriever.search_hyde(query, top_k=3)

    print(f"\nStandard retrieval (embed query directly):")
    for r in standard:
        print(f"  [{r['rank']}] score={r['score']:.4f} | {r['text'][:70]}")

    print(f"\nHyDE retrieval (embed hypothetical answer):")
    for r in hyde_result["results"]:
        print(f"  [{r['rank']}] score={r['score']:.4f} | {r['text'][:70]}")

    # Check if top-1 result is the same
    std_top = standard[0]["text"] if standard else ""
    hyde_top = hyde_result["results"][0]["text"] if hyde_result["results"] else ""
    same = std_top == hyde_top
    print(f"\nTop result changed: {'No (same)' if same else 'Yes (HyDE found different top result)'}")


def main():
    print("Building HyDE retriever...")
    retriever = HyDERetriever(DOCUMENTS)

    # Test queries — especially question-style queries where HyDE helps most
    queries = [
        "what is the refund window for annual subscribers?",
        "how many API calls can I make per second on the cheapest plan?",
        "what happens to my data if I cancel?",
        "does the basic tier get real-time support?",
    ]

    for query in queries:
        compare_hyde_vs_standard(query, retriever)

    print(f"\n{'='*70}")
    print("WHEN TO USE HyDE:")
    print("="*70)
    print("""
  USE HyDE when:
  - Queries are phrased as questions (not keyword searches)
  - Domain terminology differs between user queries and document language
  - Standard retrieval consistently misses the right chunks
  - You have confirmed via RAGAS that context_recall is low

  DO NOT USE HyDE when:
  - Latency is critical (adds one LLM call per query)
  - Your documents use the same language as user queries
  - Users search with keywords (BM25 handles this better)
  - Budget is constrained (every query costs an extra LLM call)

  Always verify with RAGAS: compare context_recall before/after HyDE.
  If it doesn't improve recall by at least 0.05, the cost is not justified.
    """)


if __name__ == "__main__":
    main()