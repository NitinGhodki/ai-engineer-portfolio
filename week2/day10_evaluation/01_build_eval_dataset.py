"""
Day 10 — Build a RAG evaluation dataset.

An eval dataset = list of (question, ground_truth_answer, context) triples.
You write the ground truth answers manually.
This is not optional — without ground truth you cannot measure recall.

Why manual? Because you are the subject matter expert on your documents.
The LLM cannot reliably create its own ground truth.
"""

import json
from pathlib import Path


# ── The document your RAG system is built on ─────────────────────────────────

DOCUMENT = """
TechCorp AI Assistant — Product Documentation

Pricing and Plans:
Our Starter plan costs 999 rupees per month and includes 100 AI queries per day.
The Professional plan costs 2999 rupees per month with unlimited queries and priority support.
The Enterprise plan is custom priced with dedicated infrastructure and SLA guarantees.

Refund Policy:
All plans come with a 14-day free trial. No credit card required for trial.
Refunds are available within 30 days of first payment for annual plans.
Monthly plans can be cancelled anytime but are not eligible for partial refunds.

Technical Specifications:
Our API supports REST and WebSocket connections.
Rate limits are 10 requests per second for Starter, 50 for Professional.
Maximum document size for upload is 10MB. Supported formats are PDF and TXT.
Response latency SLA is under 2 seconds for 95th percentile on Professional plan.

Support:
Starter plan support is via email with 48-hour response time.
Professional plan includes live chat support with 4-hour response time.
Enterprise customers get a dedicated support engineer and 1-hour response SLA.
"""


# ── 10 evaluation samples — written manually ──────────────────────────────────
# Structure per sample:
#   question:      what the user asks
#   ground_truth:  the correct, complete answer based on the document
#   context:       the exact chunk(s) from the document that answer this question
#   difficulty:    easy/medium/hard — for analysing where the system fails

EVAL_DATASET = [
    {
        "id": "q01",
        "question": "How much does the Professional plan cost per month?",
        "ground_truth": "The Professional plan costs 2999 rupees per month.",
        "context": "The Professional plan costs 2999 rupees per month with unlimited queries and priority support.",
        "difficulty": "easy",
        "category": "pricing",
    },
    {
        "id": "q02",
        "question": "What is included in the Starter plan?",
        "ground_truth": "The Starter plan costs 999 rupees per month and includes 100 AI queries per day.",
        "context": "Our Starter plan costs 999 rupees per month and includes 100 AI queries per day.",
        "difficulty": "easy",
        "category": "pricing",
    },
    {
        "id": "q03",
        "question": "Can I get a refund on a monthly plan?",
        "ground_truth": "Monthly plans can be cancelled anytime but are not eligible for partial refunds.",
        "context": "Monthly plans can be cancelled anytime but are not eligible for partial refunds.",
        "difficulty": "medium",
        "category": "refund",
    },
    {
        "id": "q04",
        "question": "How long is the free trial and do I need a credit card?",
        "ground_truth": "All plans come with a 14-day free trial and no credit card is required for the trial.",
        "context": "All plans come with a 14-day free trial. No credit card required for trial.",
        "difficulty": "medium",
        "category": "refund",
    },
    {
        "id": "q05",
        "question": "What is the API rate limit for the Professional plan?",
        "ground_truth": "The Professional plan has a rate limit of 50 requests per second.",
        "context": "Rate limits are 10 requests per second for Starter, 50 for Professional.",
        "difficulty": "medium",
        "category": "technical",
    },
    {
        "id": "q06",
        "question": "What file formats can I upload?",
        "ground_truth": "Supported upload formats are PDF and TXT, with a maximum size of 10MB.",
        "context": "Maximum document size for upload is 10MB. Supported formats are PDF and TXT.",
        "difficulty": "easy",
        "category": "technical",
    },
    {
        "id": "q07",
        "question": "What is the support response time for Enterprise customers?",
        "ground_truth": "Enterprise customers get a dedicated support engineer with a 1-hour response SLA.",
        "context": "Enterprise customers get a dedicated support engineer and 1-hour response SLA.",
        "difficulty": "easy",
        "category": "support",
    },
    {
        "id": "q08",
        "question": "What connection types does the API support?",
        "ground_truth": "The API supports both REST and WebSocket connections.",
        "context": "Our API supports REST and WebSocket connections.",
        "difficulty": "easy",
        "category": "technical",
    },
    {
        "id": "q09",
        "question": "What is the latency guarantee for the Professional plan?",
        "ground_truth": "The Professional plan guarantees response latency under 2 seconds for the 95th percentile.",
        "context": "Response latency SLA is under 2 seconds for 95th percentile on Professional plan.",
        "difficulty": "hard",
        "category": "technical",
    },
    {
        "id": "q10",
        "question": "If I pay annually and want a refund after 45 days, can I get one?",
        "ground_truth": "No. Refunds for annual plans are only available within 30 days of first payment. 45 days exceeds this window.",
        "context": "Refunds are available within 30 days of first payment for annual plans.",
        "difficulty": "hard",
        "category": "refund",
    },
]


def main():
    # Save dataset to JSON
    output_path = Path("./week2/day10_evaluation/eval_dataset.json")
    with open(output_path, "w") as f:
        json.dump(EVAL_DATASET, f, indent=2)

    print(f"Evaluation dataset created: {output_path}")
    print(f"Total samples: {len(EVAL_DATASET)}")

    # Analysis
    by_difficulty = {}
    by_category = {}
    for sample in EVAL_DATASET:
        d = sample["difficulty"]
        c = sample["category"]
        by_difficulty[d] = by_difficulty.get(d, 0) + 1
        by_category[c] = by_category.get(c, 0) + 1

    print(f"\nBy difficulty: {by_difficulty}")
    print(f"By category:   {by_category}")

    print("\nSample entry:")
    print(json.dumps(EVAL_DATASET[0], indent=2))

    # Also save document for RAG ingestion
    Path("sample_document.txt").write_text(DOCUMENT)
    print("\nDocument saved to sample_document.txt")


if __name__ == "__main__":
    main()