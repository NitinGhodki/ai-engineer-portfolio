import os
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:

    if chunk_size <= overlap*2:
        raise ValueError(f"chunk_size ({chunk_size}) must be bigger then overlap ({overlap*2}) ")

    text = text.split()
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_word = text[start:end]
        chunk_text = " ".join(chunk_word)

        chunks.append({
            "text": chunk_text,
            "chunk_index": len(chunks),
            "char_start": start,
            "word_count": len(chunk_word)
        })

        start += chunk_size - overlap

    return chunks

class VectorStore:
    def __init__(self, collection_name: str = "rag_docs"):
        self._chroma = chromadb.PersistentClient(path="./chroma_db")

        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        try:
            self._chroma.delete_collection(collection_name)
        except:
            pass

        self._collection = self._chroma.create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
        )
        print(f"  Vector store ready: collection='{collection_name}'")

    def add_chunks(self, chunks: list[dict], doc_name: str):
        self._collection.add(
            ids=[f"{doc_name}_chunk_{c['chunk_index']}" for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "doc_name": doc_name,
                "chunk_index": c["chunk_index"],
                "word_count": c["word_count"],
            } for c in chunks],
        )
        print(f"  Stored {len(chunks)} chunks from '{doc_name}'")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return hits


class MiniRAG:
    def __init__(self):
        self._client = InferenceClient(token=os.getenv("HUGGING_FACE_API"))
        self._model = os.getenv("Hugging_face_model")
        self.store = VectorStore()
        

    def ingest(self, text: str, doc_name: str):
        chunks = chunk_text(text, 200, 50)
        self.store.add_chunks(chunks=chunks, doc_name=doc_name)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        return self.store.search(query, top_k=top_k)

    def build_prompt(self, query: str, context_chunks: list[dict]) -> str:
        context_text = "\n\n---\n\n".join([
            f"[Source: {c['metadata']['doc_name']}, chunk {c['metadata']['chunk_index']}]\n{c['text']}"
            for c in context_chunks
        ])

        return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
            If the answer is not in the context, say "I don't have enough information to answer this."
            Do not use any outside knowledge.

            CONTEXT:
            {context_text}

            QUESTION: {query}

            ANSWER:"""
    
    def query(self, question, top_k: int = 3):
        print(f"\n[QUERY] {question}")

        # Retrieve relevant chunks
        chunks = self.retrieve(question, top_k=top_k)

        print(f"[RETRIEVE] Found {len(chunks)} relevant chunks:")
        for c in chunks:
            dist = c['distance']
            src = c['metadata']['doc_name']
            preview = c['text'][:80].replace('\n', ' ')
            print(f"  distance={dist:.3f} | {src} | \"{preview}...\"")

        prompt = self.build_prompt(question, chunks)

        # Generate answer
        response = self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model= self._model,
            max_tokens=512,
        )
        answer = response.choices[0].message.content

        print(f"[ANSWER] {answer}")

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "doc": c["metadata"]["doc_name"],
                    "chunk": c["metadata"]["chunk_index"],
                    "relevance_distance": round(c["distance"], 4),
                    "text_preview": c["text"][:100],
                }
                for c in chunks
            ],
        }

DOCUMENT = """
Python was created by Guido van Rossum and first released in 1991.
It was designed with code readability in mind. Python uses indentation
to define code blocks instead of curly braces. It supports multiple
programming paradigms including procedural, object-oriented, and functional.

Python has become the dominant language for data science and machine learning.
Libraries like NumPy, Pandas, and Scikit-learn make data manipulation easy.
PyTorch and TensorFlow are the main deep learning frameworks in Python.

Python's package manager is called pip. Virtual environments help isolate
project dependencies. The Python Package Index (PyPI) hosts over 400,000 packages.
Python 3 was released in 2008 and is now the standard. Python 2 reached
end of life in January 2020.
"""

def main():
    
    pipeline = MiniRAG()

    pipeline.ingest(DOCUMENT, doc_name="python")

    QUESTIONS = [
        "Who created Python?",
        "What is pip?",
        "What year did Python 2 reach end of life?",
        "Is Python good for cooking recipes?",  # not in document
    ]

    results = []
    for q in QUESTIONS:
        result = pipeline.query(q, top_k=3)
        results.append(result)
        print()

    # Summary
    print("=" * 65)
    print("QUERY SUMMARY")
    print("=" * 65)
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer'][:120]}...")
        print(f"Sources used: {[s['doc'] for s in r['sources']]}")


if __name__ == "__main__":
    main()
