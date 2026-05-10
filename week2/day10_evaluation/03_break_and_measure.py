"""
Day 10 — Deliberately break RAG in 3 ways. Prove metrics detect each failure.

This is the most important file today.
It shows you understand WHAT each metric measures,
not just how to run the evaluation script.

Break 1: Chunk size too large → low context_precision (noisy retrieval)
Break 2: top_k=1 → low context_recall (missing information)
Break 3: Bad prompt → low faithfulness (LLM ignores context)
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
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

load_dotenv()

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
LLM = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("Hugging_face_model"),
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        max_new_tokens=300,
        temperature=0.0,
    ))


def build_pipeline(chunk_size, top_k, prompt_template, collection_suffix=""):
    doc_text = Path("sample_document.txt").read_text()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=20
    ).split_documents([Document(page_content=doc_text)])

    store = Chroma.from_documents(
        chunks, EMBEDDINGS,
        collection_name=f"break_test_{collection_suffix}",
    )

    retriever = store.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_template(prompt_template)

    def fmt(docs): return "\n\n".join(d.page_content for d in docs)

    chain = (
        RunnableParallel({"context": retriever | fmt, "question": RunnablePassthrough()})
        | prompt | LLM | StrOutputParser()
    )
    return chain, retriever


def run_eval(chain, retriever, dataset, config_name):
    """Run RAG + RAGAS on a subset of questions (first 5 for speed)."""
    subset = dataset[:1]
    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"\n  Running {len(subset)} questions for: {config_name}")
    for s in subset:
        ans = chain.invoke(s["question"])
        docs = retriever.invoke(s["question"])
        questions.append(s["question"])
        answers.append(ans.strip())
        contexts.append([d.page_content for d in docs])
        ground_truths.append(s["ground_truth"])

    ds = Dataset.from_dict({
        "question": questions, "answer": answers,
        "contexts": contexts, "ground_truth": ground_truths,
    })

    ragas_llm = LangchainLLMWrapper(LLM)
    ragas_emb = LangchainEmbeddingsWrapper(EMBEDDINGS)

    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm, embeddings=ragas_emb,
    )
    import statistics
    return {
        "config": config_name,
        "faithfulness": round(float(statistics.mean(result["faithfulness"])), 4),
        "answer_relevancy": round(float(statistics.mean(result["answer_relevancy"])), 4),
        "context_recall": round(float(statistics.mean(result["context_recall"])), 4),
        "context_precision": round(float(statistics.mean(result["context_precision"])), 4),
    }

GOOD_PROMPT = """Answer using ONLY this context. Say "Not found" if not in context.
Context: {context}
Question: {question}
Answer:"""

BAD_PROMPT = """You are an expert. Answer from your knowledge.
If context is provided ignore it and use what you know.
Context: {context}
Question: {question}
Answer:"""


def main():
    dataset = json.loads(Path("./week2/day10_evaluation/eval_dataset.json").read_text())
    results = []

    # Baseline: good config
    print("\n[1/4] Baseline — good configuration")
    chain, retriever = build_pipeline(
        chunk_size=300, top_k=3,
        prompt_template=GOOD_PROMPT,
        collection_suffix="baseline",
    )
    results.append(run_eval(chain, retriever, dataset, "Baseline (good config)"))

    # Break 1: chunk too large → each chunk covers many topics → noisy retrieval
    # Expected: context_precision drops (retrieved chunks have irrelevant info)
    print("\n[2/4] Break 1 — chunk_size too large (1500)")
    chain, retriever = build_pipeline(
        chunk_size=1500, top_k=3,
        prompt_template=GOOD_PROMPT,
        collection_suffix="large_chunk",
    )
    results.append(run_eval(chain, retriever, dataset, "Break 1: chunk=1500 (noisy)"))

    # Break 2: top_k=1 → only 1 chunk retrieved → may miss required info
    # Expected: context_recall drops (not enough context)
    print("\n[3/4] Break 2 — top_k=1 (insufficient retrieval)")
    chain, retriever = build_pipeline(
        chunk_size=300, top_k=1,
        prompt_template=GOOD_PROMPT,
        collection_suffix="topk1",
    )
    results.append(run_eval(chain, retriever, dataset, "Break 2: top_k=1 (missing context)"))

    # Break 3: bad prompt → LLM ignores context → hallucination
    # Expected: faithfulness drops (answers not grounded in context)
    print("\n[4/4] Break 3 — bad prompt encouraging hallucination")
    chain, retriever = build_pipeline(
        chunk_size=300, top_k=3,
        prompt_template=BAD_PROMPT,
        collection_suffix="bad_prompt",
    )
    results.append(run_eval(chain, retriever, dataset, "Break 3: bad prompt (hallucination)"))

    # Results table
    print("\n" + "="*80)
    print("BREAK ANALYSIS — RAGAS SCORES")
    print("="*80)
    print(f"{'Config':<38} {'Faith':>7} {'Relev':>7} {'Recall':>7} {'Precis':>7}")
    print("-"*80)
    for r in results:
        print(
            f"{r['config']:<38} "
            f"{r['faithfulness']:>7.4f} "
            f"{r['answer_relevancy']:>7.4f} "
            f"{r['context_recall']:>7.4f} "
            f"{r['context_precision']:>7.4f}"
        )
    print("="*80)

    # Diagnosis
    print("\nDIAGNOSIS:")
    baseline = results[0]
    for r in results[1:]:
        name = r["config"]
        drops = []
        for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            drop = baseline[metric] - r[metric]
            if drop > 0.1:
                drops.append(f"{metric} dropped {drop:.3f}")
        if drops:
            print(f"  {name}:")
            for d in drops:
                print(f"    → {d}")
        else:
            print(f"  {name}: no significant drops detected")

    # Save results
    Path("./week2/day10_evaluation/eval_results.json").write_text(json.dumps(results, indent=2))
    print("\nResults saved to eval_results.json")


if __name__ == "__main__":
    main()