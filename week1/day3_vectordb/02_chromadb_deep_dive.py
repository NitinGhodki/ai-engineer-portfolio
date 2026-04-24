import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# setup
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# in memory vs persistent client
def demo_client_types():
    print("="*60)
    print("PART 1: Client types")
    print("="*60)

    # In-memory: fast, dies when program exits. Use for testing.
    mem_client = chromadb.Client()
    print("In-memory client: data lives only during this run")

    # Persistent: saves to disk. Use for real applications.
    disk_client = chromadb.PersistentClient(path="./week1/day3_vectordb/chroma_storage")
    print("Persistent client: data saved to ./week1/day3_vectordb/chroma_storage/")
    print("Run this script twice — second run will find existing data")

    return disk_client

# Collection
def demo_collections(client):
    """
    A collection = a named group of documents.
    Think of it like a table in SQL.
    
    You can have multiple collections:
    - "user_123_docs" for one user's documents
    - "user_456_docs" for another user's documents
    This is how multi-user RAG systems isolate data.
    """
    print("\n" + "="*60)
    print("PART 2: Collections — organizing your data")
    print("="*60)

    # get_or_create: safe to call multiple times
    # If collection exists → returns it
    # If collection doesn't exist → creates it
    collection = client.get_or_create_collection(
        name="tech_docs",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )

    print(f"Collection: {collection.name}")
    print(f"Documents already in it: {collection.count()}")

    # List all collections
    all_collections = client.list_collections()
    print(f"All collections in this DB: {[c.name for c in all_collections]}")

    return collection

# add document
def demo_adding(collection):
    """
    Three ways to add documents:
    1. add() — fails if ID exists (safe insert)
    2. upsert() — updates if ID exists, inserts if not (safe update)
    3. update() — fails if ID doesn't exist (safe update only)
    """
    print("\n" + "="*60)
    print("PART 3: Adding documents")
    print("="*60)

    # Only add if collection is empty (handles re-runs with persistent client)
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} docs. Skipping add.")
        return

    documents = [
        "Python is a high-level programming language created by Guido van Rossum.",
        "Machine learning uses algorithms to learn patterns from data.",
        "Docker containers package applications with their dependencies.",
        "REST APIs use HTTP methods like GET, POST, PUT, DELETE.",
        "PostgreSQL is an open-source relational database.",
        "Redis is an in-memory data store used for caching.",
        "Kubernetes orchestrates containerized applications at scale.",
        "FastAPI is a modern Python framework for building APIs.",
        "LangChain is a framework for building LLM applications.",
        "Vector databases store embeddings for semantic search.",
    ]

    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [
        {"category": "language",   "difficulty": "beginner"},
        {"category": "ml",         "difficulty": "intermediate"},
        {"category": "devops",     "difficulty": "intermediate"},
        {"category": "backend",    "difficulty": "beginner"},
        {"category": "database",   "difficulty": "beginner"},
        {"category": "database",   "difficulty": "beginner"},
        {"category": "devops",     "difficulty": "advanced"},
        {"category": "backend",    "difficulty": "beginner"},
        {"category": "ml",         "difficulty": "intermediate"},
        {"category": "ml",         "difficulty": "intermediate"},
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"Added {collection.count()} documents")
    print("Sample metadata structure:")
    print(f"  doc_0: {metadatas[0]}")
    print(f"  doc_6: {metadatas[6]}")

#Querying 
def demo_querying(collection):
    """
    Three types of queries you must know:
    1. Pure semantic search
    2. Semantic + metadata filter
    3. Get by ID (exact lookup)
    """
    print("\n" + "="*60)
    print("PART 4: Query types")
    print("="*60)

    # Query 1: Pure semantic search
    print("\n--- Query 1: Pure semantic search ---")
    results = collection.query(
        query_texts=["how to deploy software applications"],
        n_results=3,
    )
    print("Query: 'how to deploy software applications'")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  [{dist:.4f}] [{meta['category']}] {doc[:60]}")

    # Query 2: Semantic + metadata filter (AND logic)
    print("\n--- Query 2: Semantic search + metadata filter ---")
    results_filtered = collection.query(
        query_texts=["how to deploy software applications"],
        n_results=3,
        where={"category": "devops"},
    )
    print("Same query but ONLY from category='devops':")
    for doc, meta, dist in zip(
        results_filtered["documents"][0],
        results_filtered["metadatas"][0],
        results_filtered["distances"][0],
    ):
        print(f"  [{dist:.4f}] [{meta['category']}] {doc[:60]}")

    # Query 3: Filter by multiple metadata conditions
    print("\n--- Query 3: Multiple metadata filters ($and) ---")
    try:
        results_multi = collection.query(
            query_texts=["database and storage"],
            n_results=3,
            where={
                "$and": [
                    {"category": {"$eq": "database"}},
                    {"difficulty": {"$eq": "beginner"}},
                ]
            },
        )
        print("Query: 'database and storage' where category=database AND difficulty=beginner")
        for doc, meta in zip(
            results_multi["documents"][0],
            results_multi["metadatas"][0],
        ):
            print(f"  [{meta['category']}|{meta['difficulty']}] {doc[:60]}")
    except Exception as e:
        print(f"Multi-filter error: {e}")

    # Query 4: Get by exact ID
    print("\n--- Query 4: Get by exact ID ---")
    result = collection.get(ids=["doc_0", "doc_4"])
    for id, doc, meta in zip(
        result["ids"],
        result["documents"],
        result["metadatas"],
    ):
        print(f"  ID={id} | [{meta['category']}] {doc[:60]}")

#Upsert and delete
def demo_upsert_delete(collection):
    """
    Update and delete — critical for production systems.
    
    When do you need these?
    - User edits a document → upsert with new content
    - User deletes a document → delete all its chunks
    - Document content changes → upsert updates the embedding too
    """
    print("\n" + "="*60)
    print("PART 5: Upsert and delete")
    print("="*60)

    # Upsert: update doc_0 with new content
    print("\nBefore upsert:")
    before = collection.get(ids=["doc_0"])
    print(f"  doc_0: {before['documents'][0]}")

    collection.upsert(
        ids=["doc_0"],
        documents=["Python 3.12 is the latest stable version, featuring improved performance."],
        metadatas=[{"category": "language", "difficulty": "beginner", "updated": "true"}],
    )

    print("After upsert:")
    after = collection.get(ids=["doc_0"])
    print(f"  doc_0: {after['documents'][0]}")
    print(f"  new metadata: {after['metadatas'][0]}")

    # Delete
    print(f"\nCount before delete: {collection.count()}")
    collection.delete(ids=["doc_9"])
    print(f"Count after deleting doc_9: {collection.count()}")

    # Verify it's gone
    result = collection.get(ids=["doc_9"])
    print(f"doc_9 after delete: {result['documents']}")  # should be empty list

#  Distance metrics explained
def demo_distance_metrics():
    """
    ChromaDB supports 3 distance metrics. Know the difference.
    """
    print("\n" + "="*60)
    print("PART 6: Distance metrics")
    print("="*60)

    metrics = {
        "cosine": {
            "formula": "1 - cosine_similarity",
            "range": "0 (identical) to 2 (opposite)",
            "use_when": "comparing meaning regardless of text length",
            "default": True,
        },
        "l2": {
            "formula": "euclidean distance (straight line between vectors)",
            "range": "0 (identical) to infinity",
            "use_when": "when absolute vector magnitude matters",
            "default": False,
        },
        "ip": {
            "formula": "negative dot product",
            "range": "depends on vector magnitude",
            "use_when": "when vectors are pre-normalized (same as cosine then)",
            "default": False,
        },
    }

    for metric, info in metrics.items():
        default_marker = " ← USE THIS (default)" if info["default"] else ""
        print(f"\n  {metric.upper()}{default_marker}")
        print(f"    Formula:   {info['formula']}")
        print(f"    Range:     {info['range']}")
        print(f"    Use when:  {info['use_when']}")

    print("\nFor RAG systems: always use cosine. Period.")
    print("Reason: you care about semantic direction, not vector magnitude.")


if __name__ == "__main__":
    client = demo_client_types()
    collection = demo_collections(client)
    demo_adding(collection)
    demo_querying(collection)
    demo_upsert_delete(collection)
    demo_distance_metrics()

    print("\n" + "="*60)
    print("Run this script a SECOND TIME.")
    print("Observe: persistent client remembers your data.")
    print("Collection count will be non-zero on the second run.")
    print("="*60)                  