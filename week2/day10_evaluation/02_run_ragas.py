"""
Day 10 — Run RAGAS evaluation on your RAG pipeline.

Flow:
1. Load eval dataset
2. Build RAG pipeline (same as Day 4)
3. For each question: retrieve chunks + generate answer
4. Feed (question, answer, contexts, ground_truth) to RAGAS
5. Get scores for all 4 metrics
6. Print results table
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

load_dotenv()


# ── Build RAG pipeline

def build_rag_pipeline(chunk_size: int = 300, chunk_overlap: int = 50, top_k: int = 5):
    """
    Build RAG pipeline with configurable parameters.
    Different configs let us measure how parameters affect quality.
    """
    # Load document
    doc_text = Path("sample_document.txt").read_text()
    raw_doc = Document(page_content=doc_text, metadat={"source": "sample_document.txt"})

    # split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents([raw_doc])
    print(f"  Chunks created: {len(chunks)} (size={chunk_size}, overlap={chunk_overlap})")

     # Embed and store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=f"eval_chunks_{chunk_size}",
        # persist_directory="./week2/day10_evaluation/chroma_db"
    )

    # # To load an existing DB from your hard drive:
    # vectorstore = Chroma(
    #     persist_directory="./week2/day10_evaluation/chroma_db",
    #     embedding_function=embeddings,
    #     collection_name=f"eval_chunks_{chunk_size}"
    # )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = ChatHuggingFace(
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=80,
            temperature=0.0,
        ))
    
    # Chain
    prompt = ChatPromptTemplate.from_template(
        """Answer using ONLY this context. If not found say "Not in documents."

        Context: {context}

        Question: {question}
        Answer:""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── Run RAG on eval dataset
def run_rag_on_dataset(chain, retriever, eval_dataset: list[dict]) -> dict:
    """
    Run RAG pipeline on every question in eval dataset.
    Returns dict in RAGAS format.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"\nRunning RAG on {len(eval_dataset)} eval questions...")
    for i, sample in enumerate(eval_dataset, 1):
        q = sample["question"]
        gt = sample["ground_truth"]

        # Get answer from RAG
        answer = chain.invoke(q)

        # Get retrieved chunks (for context evaluation)
        retrieved_docs = retriever.invoke(q)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        questions.append(q)
        answers.append(answer.strip())
        contexts.append(retrieved_contexts)
        ground_truths.append(gt)

        print(f"  [{i}/{len(eval_dataset)}] Q: {q[:50]!r}")
        print(f"    Answer: {answer.strip()[:80]!r}")
        print(f"    Contexts retrieved: {len(retrieved_contexts)}")

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

# ── Run RAGAS evaluation

def run_ragas_evaluation(rag_results: dict, config_name: str) -> dict:
    """
    Feed RAG results into RAGAS and get metric scores.
    Returns dict of metric_name → score.
    """
    print(f"\nRunning RAGAS evaluation for: {config_name}")

    # Convert to HuggingFace Dataset (RAGAS requirement)
    dataset = Dataset.from_dict(rag_results)

    # RAGAS needs an LLM for faithfulness and answer_relevancy
    eval_llm  = ChatHuggingFace(
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=300,
            temperature=0.0,
        ))
    
    eval_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Wrap for RAGAS
    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ], 
        llm = ragas_llm,
        embeddings=ragas_embeddings,
    )
    import statistics
    
    scores = {
        "config": config_name,
        "faithfulness": round(float(statistics.mean(result["faithfulness"])), 4),
        "answer_relevancy": round(float(statistics.mean(result["answer_relevancy"])), 4),
        "context_recall": round(float(statistics.mean(result["context_recall"])), 4),
        "context_precision": round(float(statistics.mean(result["context_precision"])), 4),
    }
    scores["overall"] = round(
        sum(v for k, v in scores.items() if k not in ["config", "overall"]) / 4, 4
    )

    return scores

# ── Print results table

def print_results_table(all_results: list[dict]):
    """Print a clean comparison table of all configs."""
    print("\n" + "="*75)
    print("RAGAS EVALUATION RESULTS")
    print("="*75)
    print(f"{'Config':<30} {'Faith':>8} {'Relev':>8} {'Recall':>8} {'Precis':>8} {'Overall':>8}")
    print("-"*75)
    for r in all_results:
        print(
            f"{r['config']:<30} "
            f"{r['faithfulness']:>8.4f} "
            f"{r['answer_relevancy']:>8.4f} "
            f"{r['context_recall']:>8.4f} "
            f"{r['context_precision']:>8.4f} "
            f"{r['overall']:>8.4f}"
        )
    print("="*75)
    print("\nScore guide: 0.0 = worst, 1.0 = best")
    print("Faithfulness <0.7: LLM is hallucinating")
    print("Context Recall <0.7: Retrieval is missing relevant chunks")
    print("Context Precision <0.7: Retrieving too much noise")

def main():
    # Load eval dataset
    eval_dataset = json.loads(Path("./week2/day10_evaluation/eval_dataset.json").read_text())
    print(f"Loaded {len(eval_dataset)} evaluation samples")

    all_results = []

    # Config 1: Good settings (baseline)
    print("\n--- CONFIG 1: Good baseline (chunk=300, overlap=50, top_k=3) ---")
    chain, retriever = build_rag_pipeline(chunk_size=300, chunk_overlap=50, top_k=3)
    rag_results = run_rag_on_dataset(chain, retriever, eval_dataset)
    scores = run_ragas_evaluation(rag_results, "Good: chunk=300 overlap=50 k=3")
    all_results.append(scores)

    print_results_table(all_results)


if __name__ == "__main__":
    main()


