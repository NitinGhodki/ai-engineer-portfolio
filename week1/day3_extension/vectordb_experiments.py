"""
3 experiments to understand what ChromaDB is actually doing.
Run each one, read the output, write a one-line comment explaining what you observed.
"""
import chromadb
from chromadb.utils import embedding_functions

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()
collection = client.create_collection("experiments", embedding_function=embed_fn)

collection.add(
    ids=["1", "2", "3", "4", "5"],
    documents=[
        "The cat sat on the mat.",
        "A dog is running in the park.",
        "Felines are known for being independent.",
        "Machine learning is a subset of AI.",
        "Python is used for data science.",
    ]
)

results = collection.query(query_texts=["cats and other domestic animals"], n_results=2)
print("EXPERIMENT 1 — Semantic search:")
print(f"  Query: 'cats and other domestic animals'")
print(f"  Top result: {results['documents'][0][0]}")
print(f"  2nd result: {results['documents'][0][1]}")
client2 = chromadb.Client()
col2 = client2.create_collection("meta_test", embedding_function=embed_fn)

col2.add(
    ids=["a", "b", "c", "d"],
    documents=[
        "Python is a great language for AI.",
        "Java is used in enterprise systems.",
        "Python has excellent ML libraries.",
        "JavaScript runs in the browser.",
    ],
    metadatas=[
        {"language": "python"},
        {"language": "java"},
        {"language": "python"},
        {"language": "javascript"},
    ]
)

# Search only within Python documents
results2 = col2.query(
    query_texts=["programming language for data science"],
    n_results=2,
    where={"language": "python"}  # ← metadata filter
)
print("\nEXPERIMENT 2 — Metadata filtering:")
print(f"  Results filtered to python only: {results2['documents'][0]}")


# EXPERIMENT 3: What happens when you query more results than documents?
client3 = chromadb.Client()
col3 = client3.create_collection("small", embedding_function=embed_fn)
col3.add(ids=["x"], documents=["Only one document exists."])

try:
    results3 = col3.query(query_texts=["anything"], n_results=5)
    print("\nEXPERIMENT 3 — Over-querying:")
    print(f"  Asked for 5, got: {len(results3['documents'][0])} results")
except Exception as e:
    print(f"\nEXPERIMENT 3 — Error: {e}")
