import numpy as np
import json
from sentence_transformers import SentenceTransformer

class TinyVectorDB:
    """
    A vector database that stores documents with their embeddings.
    Supports: add, search, delete, filter by metadata.
    
    Internally just 3 parallel lists:
    - _ids: unique identifiers
    - _documents: original text
    - _embeddings: numpy vectors
    - _metadatas: dict of extra info per document
    """

    def __init__(self):
        self._ids = []
        self._documents = []
        self._embeddings = []
        self._metadatas = []
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"TinyVectorDB ready. Embedding model: all-MiniLM-L6-v2 (384 dims)")

    def _embed(self, text: str) -> np.ndarray:
        """Convert text to vectors(numbers)"""
        return self._model.encode(text, normalize_embeddings=True)

    def add(self, id: str, document: str, metadata: dict = None):
        """
        Add a document to the DB.
        Embeds it immediately and stores the vector alongside the text.
        """

        if id in self._ids:
            raise ValueError(f"ID '{id}' already exists. IDs must be unique.")
        
        embedding = self._embed(document)

        self._ids.append(id)
        self._documents.append(document)
        self._embeddings.append(embedding)
        self._metadatas.append(metadata or {})


    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:

        """
        Since we use normalize_embeddings=True above,
        vectors are already unit length (magnitude=1).
        So cosine similarity = dot product. Simple.
        """

        return float(np.dot(vec_a, vec_b))
    
    def search(self, query: str, top_k: int = 5, where: dict = None) -> list[dict]:
        """
        Find the top_k most similar documents to the query.
        
        where: optional metadata filter e.g. {"category": "science"}
               ALL conditions must match (AND logic).
        """

        if not self._documents:
            return []
        
        query_vec = self._embed(query)

        results = []
        for i, (id, doc, emb, meta) in enumerate(
            zip(self._ids, self._documents, self._embeddings, self._metadatas)
        ):
            if where:
                match = all(meta.get(k) == v for k, v in where.items())
                if not match:
                    continue
                    
            score = self._cosine_similarity(query_vec, emb)

            results.append({
                "id": id,
                "document": doc,
                "score": score,
                "metadata": meta
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
        
    def delete(self, id: str):
        """Remove a document by ID."""
        if id not in self._ids:
            raise ValueError(f"ID '{id}' not found.")
        idx = self._ids.index(id)
        self._ids.pop(idx)
        self._documents.pop(idx)
        self._embeddings.pop(idx)
        self._metadatas.pop(idx)
        print(f"Deleted document: {id}")

    
    def count(self) -> int:
        return len(self._ids)

    def get(self, id: str) -> dict:
        """Retrieve a specific document by ID."""
        if id not in self._ids:
            raise ValueError(f"ID '{id}' not found.")
        idx = self._ids.index(id)
        return {
            "id": self._ids[idx],
            "document": self._documents[idx],
            "metadata": self._metadatas[idx],
            "embedding_preview": self._embeddings[idx][:5].tolist(),
        }

    def save(self, filepath: str):
        """
        Persist to disk as JSON.
        Real vector DBs use binary formats (much faster) but JSON is readable.
        """
        data = {
            "ids": self._ids,
            "documents": self._documents,
            "embeddings": [e.tolist() for e in self._embeddings],
            "metadatas": self._metadatas,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"Saved {len(self._ids)} documents to {filepath}")

    def load(self, filepath: str):
        """Load from disk."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self._ids = data["ids"]
        self._documents = data["documents"]
        self._embeddings = [np.array(e) for e in data["embeddings"]]
        self._metadatas = data["metadatas"]
        print(f"Loaded {len(self._ids)} documents from {filepath}")


def experiment_1_basic_search(db: TinyVectorDB):
    """Basic semantic search — does it find by meaning or keyword?"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Semantic search vs keyword search")
    print("="*60)

    documents = [
        ("doc1", "The cat sat on the mat.", {"type": "animal"}),
        ("doc2", "A dog is running in the park.", {"type": "animal"}),
        ("doc3", "Felines are known for being independent.", {"type": "animal"}),
        ("doc4", "Machine learning is a subset of AI.", {"type": "tech"}),
        ("doc5", "Python is used for data science.", {"type": "tech"}),
        ("doc6", "Neural networks mimic the human brain.", {"type": "tech"}),
    ]

    for id, doc, meta in documents:
        db.add(id, doc, meta)

    query = "cats and other domestic animals"
    print(f"\nQuery: '{query}'")
    print(f"(Note: query contains 'cats' — doc1 has 'cat', doc3 has no matching keyword)")
    print()

    results = db.search(query, top_k=5)
    for rank, r in enumerate(results, 1):
        print(f"  Rank {rank}: [{r['score']:.4f}] {r['document']}")

    print()
    print("KEY INSIGHT: If rank 1 is doc3 (Felines...) not doc1 (cat on mat),")
    print("it means search is by MEANING, not keyword. 'Felines' = 'cats' semantically.")

def experiment_2_metadata_filtering(db: TinyVectorDB):
    """Filter search results by metadata."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Metadata filtering")
    print("="*60)

    query = "artificial intelligence and programming"

    print(f"\nQuery: '{query}'")
    print("\nWithout filter (searches everything):")
    results = db.search(query, top_k=5)
    for r in results:
        print(f"  [{r['score']:.4f}] [{r['metadata']['type']}] {r['document']}")

    print("\nWith filter: where={'type': 'tech'} only:")
    results_filtered = db.search(query, top_k=5, where={"type": "tech"})
    for r in results_filtered:
        print(f"  [{r['score']:.4f}] [{r['metadata']['type']}] {r['document']}")

    print("\nWith filter: where={'type': 'animal'} only:")
    results_animal = db.search(query, top_k=5, where={"type": "animal"})
    for r in results_animal:
        print(f"  [{r['score']:.4f}] [{r['metadata']['type']}] {r['document']}")

    print()
    print("KEY INSIGHT: Same query, different results based on metadata filter.")
    print("This is how multi-tenant RAG works — each user only searches THEIR documents.")


def experiment_3_score_interpretation(db: TinyVectorDB):
    """What do the similarity scores actually mean?"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Understanding similarity scores")
    print("="*60)

    test_db = TinyVectorDB()

    pairs = [
        ("I love machine learning", "I enjoy ML a lot"),
        ("I love machine learning", "AI and deep learning are fascinating"),
        ("I love machine learning", "The weather is nice today"),
        ("I love machine learning", "I hate machine learning"),
        ("I love machine learning", "I love machine learning"),  # identical
    ]

    print(f"\nBase sentence: 'I love machine learning'")
    print(f"{'Comparison':<45} {'Score':>7}  {'Interpretation'}")
    print("-"*75)

    base_vec = test_db._embed("I love machine learning")

    interpretations = {
        (0.9, 1.0): "nearly identical meaning",
        (0.7, 0.9): "very similar",
        (0.5, 0.7): "related topic",
        (0.3, 0.5): "loosely related",
        (0.0, 0.3): "unrelated",
        (-1.0, 0.0): "opposite meaning",
    }

    for _, comparison in pairs:
        comp_vec = test_db._embed(comparison)
        score = float(np.dot(base_vec, comp_vec))

        interp = "unknown"
        for (low, high), label in interpretations.items():
            if low <= score < high:
                interp = label
                break

        print(f"  {comparison:<43} {score:>7.4f}  {interp}")


def experiment_4_persistence(db: TinyVectorDB):
    """Save and reload — data survives program restart."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Persistence — data surviving restart")
    print("="*60)

    db.save("week1/day3_vectordb/tiny_vectordb.json")

    fresh_db = TinyVectorDB()
    fresh_db.load("week1/day3_vectordb/tiny_vectordb.json")

    print(f"\nOriginal DB count: {db.count()}")
    print(f"Reloaded DB count: {fresh_db.count()}")

    results = fresh_db.search("cats", top_k=1)
    print(f"Search after reload still works: '{results[0]['document']}'")
    print("\nKEY INSIGHT: This is what ChromaDB PersistentClient does,")
    print("just faster and with proper indexing for large datasets.")


def experiment_5_delete_and_update(db: TinyVectorDB):
    """Delete a document — important for production systems."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Delete operations")
    print("="*60)

    print(f"Count before delete: {db.count()}")
    results_before = db.search("feline cat", top_k=1)
    print(f"Search 'feline cat' before delete: '{results_before[0]['document']}'")

    db.delete("doc3")  # delete the "Felines..." document

    print(f"Count after delete: {db.count()}")
    results_after = db.search("feline cat", top_k=1)
    print(f"Search 'feline cat' after delete: '{results_after[0]['document']}'")

    print("\nKEY INSIGHT: Why does delete matter in production?")
    print("If a user deletes a document, you must remove its chunks from the vector DB.")
    print("Otherwise deleted content still appears in search results. Privacy violation.")


if __name__ == "__main__":
    db = TinyVectorDB()
    experiment_1_basic_search(db)
    experiment_2_metadata_filtering(db)
    experiment_3_score_interpretation(db)
    experiment_4_persistence(db)
    experiment_5_delete_and_update(db)