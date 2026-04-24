import os
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Two completely different documents
DOCS = {
    "python_guide": {
        "text": """Python was created by Guido van Rossum in 1991. It emphasizes
        code readability and simplicity. Python is dynamically typed and garbage
        collected. It supports object-oriented, procedural, and functional programming.
        Python's standard library is extensive. pip is Python's package manager.
        Virtual environments isolate project dependencies. Python is the dominant
        language for machine learning and data science.""",
        "category": "programming",
        "author": "tech_team",
    },
    "hr_policy": {
        "text": """Employees are entitled to 20 days of paid leave per year.
        Medical leave requires a doctor's certificate for absences exceeding 3 days.
        Work from home is allowed up to 2 days per week for eligible roles.
        Performance reviews are conducted twice a year, in June and December.
        The probation period for new employees is 3 months. Salary increments
        are processed in April each year. All employees must complete mandatory
        compliance training by end of Q1.""",
        "category": "hr",
        "author": "hr_team",
    },
}

def chunk_text(text: str, chunk_size: int = 60, overlap: int = 15) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    words = text.split()
    chunks, start = [], 0

    while start < len(words):
        chunks.append(" ".join(words[start: start+chunk_size]))
        start += chunk_size - overlap

    return chunks

class MultiDocRAG:
    def __init__(self):
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(
            "multi_doc", embedding_function=embed_fn
        )
        self._llm = InferenceClient(token=os.getenv("HUGGING_FACE_API"))
        self._model = os.getenv("Hugging_face_model")

    def ingest_all(self):
        for doc_name, doc_info in DOCS.items():
            chunks = chunk_text(doc_info["text"])
            self._collection.add(
                ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))],
                documents=chunks,
                metadatas=[
                    {
                        "doc_name": doc_name,
                        "category": doc_info["category"],
                        "author": doc_info["author"],
                        "chunk_index": i,
                    }
                    for i in range(len(chunks))
                ],
            )
            print(f"Ingested '{doc_name}': {len(chunks)} chunks")
        print(f"Total chunks: {self._collection.count()}")

    def query(
        self,
        question: str,
        category_filter: str = None,
        top_k: int = 3,
    ) -> str:
        """
        Query with optional category filter.
        category_filter="hr" → only search HR docs
        category_filter=None → search everything
        """
        where = {"category": category_filter} if category_filter else None

        results = self._collection.query(
            query_texts=[question],
            n_results=top_k,
            where=where,
        )

        context = "\n\n".join([
            f"[{meta['doc_name']}]: {doc}"
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0],
            )
        ])

        prompt = f"""Answer using ONLY this context. If not found, say "Not in provided documents."

            CONTEXT:
            {context}

            QUESTION: {question}
            ANSWER:"""

        response = self._llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            max_tokens=256,
        )
        return response.choices[0].message.content

def main():
    rag = MultiDocRAG()
    rag.ingest_all()

    test_cases = [
        ("How many leave days do employees get?", "hr"),
        ("What language was Python written in?", "programming"),
        ("When are performance reviews?", "hr"),
        ("What is pip?", "programming"),
        # Cross-category: ask HR question but filter to programming docs
        ("How many leave days do employees get?", "programming"),
    ]

    print("\n" + "="*60)
    for question, category in test_cases:
        print(f"\nQ: {question}")
        print(f"Filter: category='{category}'")
        answer = rag.query(question, category_filter=category)
        print(f"A: {answer}")
        print("-"*40)


if __name__ == "__main__":
    main()

