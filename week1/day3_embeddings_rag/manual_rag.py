import os
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import chromadb
from chromadb.utils import embedding_functions

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping chunks.

    Why overlap?
    If a key sentence sits at the boundary between two chunks, it would be
    split and potentially missed. Overlap ensures boundary content appears
    in both adjacent chunks.

    Returns list of dicts with 'text', 'chunk_index', 'char_start'.
    """

    if chunk_size <= overlap*2:
        raise ValueError(f"chunk_size ({chunk_size}) must be bigger then overlap ({overlap*2}) ")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "chunk_index": len(chunks),
            "char_start": start,
            "word_count": len(chunk_words)
        })

        start += chunk_size - overlap

    return chunks


def print_chunking_analysis(chunks: list[dict]):
    """Show what chunking actually does to your document."""
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg words per chunk: {sum(c['word_count'] for c in chunks) / len(chunks):.0f}")
    print(f"  First chunk preview: \"{chunks[0]['text'][:100]}...\"")
    if len(chunks) > 1:
        # Show overlap between chunk 0 and chunk 1
        c0_words = set(chunks[0]['text'].split()[-20:])
        c1_words = set(chunks[1]['text'].split()[:20])
        overlap_words = c0_words & c1_words
        print(f"  Overlap between chunk 0-1: {len(overlap_words)} words")


class VectorStore:
    """
    Thin wrapper around ChromaDB.
    Handles: create collection, add documents, query.
    """

    def __init__(self, collection_name: str = "rag_docs"):
        # it will only store data at run time lavel
        # self._chroma = chromadb.Client()

        #dont ampact on re-run 
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
        """Add document chunks to the vector store."""
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
        """
        Semantic search: embed query, find top-k similar chunks.
        Returns list of results with text, score, and metadata.
        """
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
    
    def count(self) -> int:
        return self._collection.count()
    

class RAGPipeline:
    """
    Full RAG pipeline: ingest documents → answer questions.

    Ingestion:  Document text → chunk → embed → store in ChromaDB
    Retrieval:  Query → embed → vector search → top-k chunks
    Generation: chunks + query → LLM prompt → answer
    """

    def __init__(self):
        self._llm = InferenceClient(token=os.getenv("HUGGING_FACE_API"))
        self._model = os.getenv("Hugging_face_model")
        self._store = VectorStore()
        self._ingested_docs = []

    def ingest(self, text: str, doc_name: str, chunk_size: int = 2000, overlap: int = 50):
        """
        Step 1 of RAG: chunk the document and store in vector DB.
        Call this once per document before querying.
        """
        print(f"\n[INGEST] Processing '{doc_name}'...")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print_chunking_analysis(chunks)
        self._store.add_chunks(chunks, doc_name=doc_name)
        self._ingested_docs.append(doc_name)
        print(f"  Total chunks in store: {self._store.count()}")

    
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Step 2 of RAG: find the most relevant chunks for this query.
        """
        return self._store.search(query, top_k=top_k)
    
    def build_prompt(self, query: str, context_chunks: list[dict]) -> str:
        """
        Step 3 of RAG: inject retrieved context into the LLM prompt.

        This is the most important prompt in RAG systems.
        The LLM MUST be told to use ONLY the provided context.
        Without this instruction, it will use its own training knowledge
        and potentially hallucinate or ignore your documents entirely.
        """
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

    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> dict:
        """
        Full RAG query: retrieve → build prompt → generate answer.
        Returns answer + the source chunks used (for citation).
        """
        if verbose:
            print(f"\n[QUERY] {question}")

        # Retrieve relevant chunks
        chunks = self.retrieve(question, top_k=top_k)

        if verbose:
            print(f"[RETRIEVE] Found {len(chunks)} relevant chunks:")
            for c in chunks:
                dist = c['distance']
                src = c['metadata']['doc_name']
                preview = c['text'][:80].replace('\n', ' ')
                print(f"  distance={dist:.3f} | {src} | \"{preview}...\"")

        # Build prompt with context
        prompt = self.build_prompt(question, chunks)

        if verbose:
            print(f"[PROMPT] Built prompt ({len(prompt.split())} words)")

        # Generate answer
        response = self._llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model= os.getenv("Hugging_face_model"),
            max_tokens=512,
        )
        answer = response.choices[0].message.content

        if verbose:
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

# Two fake but realistic documents to test with
DOCUMENT_1 = """
Artificial Intelligence and Machine Learning Overview

Artificial intelligence (AI) refers to the simulation of human intelligence in machines
that are programmed to think and act like humans. Machine learning (ML) is a subset of AI
that provides systems the ability to automatically learn and improve from experience without
being explicitly programmed.

Deep learning is a subset of machine learning that uses neural networks with many layers.
These neural networks attempt to simulate the behavior of the human brain, allowing it to
learn from large amounts of data. Deep learning drives many AI applications like speech
recognition, image recognition, and natural language processing.

Large language models (LLMs) are a type of deep learning model trained on massive text
datasets. They learn to predict the next word in a sequence, and through this simple task,
develop a surprisingly broad understanding of language and knowledge. GPT-4, Claude, and
Gemini are examples of large language models. These models can write code, answer questions,
summarize documents, and perform many other language tasks.

Retrieval Augmented Generation (RAG) is a technique that combines LLMs with external knowledge.
Instead of relying solely on the model's training data, RAG retrieves relevant documents from
a knowledge base and provides them as context to the LLM. This allows LLMs to answer questions
about private data and recent events they were not trained on.

Vector databases are specialized databases designed to store and search high-dimensional vectors.
They use approximate nearest neighbor algorithms to efficiently find the most similar vectors
to a query vector. Pinecone, Weaviate, Chroma, and pgvector are popular vector databases
used in AI applications.
"""

DOCUMENT_2 = """
Company Refund and Support Policy

Our standard refund policy allows customers to request a full refund within 30 days of purchase.
After 30 days, refunds are evaluated on a case-by-case basis by our support team. To initiate
a refund, customers must contact support@example.com with their order ID and reason for the refund.

Premium subscribers receive priority support with a guaranteed 4-hour response time during
business hours (9 AM to 6 PM IST, Monday through Friday). Standard users can expect responses
within 24 to 48 business hours.

For technical issues, customers should first check our documentation at docs.example.com.
If the issue persists, submit a ticket through the support portal. Include your account email,
a description of the issue, and any error messages you received. Screenshots are helpful.

Account cancellations can be processed through the account settings page. Data is retained
for 90 days after cancellation, after which it is permanently deleted. Customers can request
a data export before cancellation by contacting our support team.

Our service level agreement (SLA) guarantees 99.9 percent uptime for all paid plans.
In case of downtime exceeding this threshold, affected customers receive service credits
equivalent to 10 times the downtime duration, applied to their next billing cycle.
"""

def main():
    print("=" * 65)
    print("MANUAL RAG PIPELINE — Day 3")
    print("=" * 65)

    pipeline = RAGPipeline()

    # Ingest both documents
    pipeline.ingest(DOCUMENT_1, doc_name="ai_overview")
    pipeline.ingest(DOCUMENT_2, doc_name="company_policy")

    # Test queries
    questions = [
        "What is retrieval augmented generation?",
        "How do I get a refund?",
        "What are vector databases used for?",
        "What is the response time for premium support?",
        "Who invented the internet?",  # ← not in any document
    ]

    print("\n" + "=" * 65)
    print("RUNNING QUERIES")
    print("=" * 65)

    results = []
    for q in questions:
        result = pipeline.query(q, top_k=3, verbose=True)
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