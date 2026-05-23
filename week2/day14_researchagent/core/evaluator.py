"""
RAGAS evaluation — from Day 10, adapted for on-demand API endpoint.
"""

import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from datasets import Dataset
import statistics

load_dotenv()

# Evaluation history — stores recent queries for batch evaluation
_eval_history: list[dict] = []
MAX_HISTORY = 20

def record_query(question: str, answer: str, contexts: list[str], ground_truth: str = ""):
    """Record a query for later evaluation."""
    _eval_history.append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth or answer,  # use answer as proxy if no ground truth
    })
    if len(_eval_history) > MAX_HISTORY:
        _eval_history.pop(0)


def run_evaluation(n_samples: int = 5) -> dict:
    """
    Run RAGAS on the last n_samples queries.
    Returns scores for all 4 metrics.
    """

    if not _eval_history:
        return {"error": "No queries recorded yet. Ask some questions first."}

    samples = _eval_history[-n_samples:]

    dataset = Dataset.from_dict({
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    })

    eval_llm = ChatHuggingFace (
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY_32"),
            max_new_tokens=512,
            temperature=0.0,   
        )
    )

    eval_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=LangchainLLMWrapper(eval_llm),
        embeddings=LangchainEmbeddingsWrapper(eval_embeddings),
    )

    return {
        "samples_evaluated": len(samples),
        "faithfulness": round(float(statistics.mean(result["faithfulness"])), 4),
        "answer_relevancy": round(float(statistics.mean(result["answer_relevancy"])), 4),
        "context_recall": round(float(statistics.mean(result["context_recall"])), 4),
        "context_precision": round(float(statistics.mean(result["context_precision"])), 4),
        "overall": round(
            (float(statistics.mean(result["faithfulness"])) + float(result["answer_relevancy"]) +
             float(statistics.mean(result["context_recall"])) + float(result["context_precision"])) / 4, 4
        ),
        "interpretation": {
            "faithfulness": "good" if float(statistics.mean(result["faithfulness"])) > 0.7 else "needs improvement",
            "context_recall": "good" if float(statistics.mean(result["context_recall"])) > 0.7 else "needs improvement",
        }
    }


def get_history_count() -> int:
    return len(_eval_history)
