import os 
import json
import numpy as np 
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

def get_embedding(text: str, client: InferenceClient) -> np.ndarray:

    response = client.feature_extraction(
        text,
        # model=os.getenv("Hugging_face_model")
         model="sentence-transformers/all-MiniLM-L6-v2",
    ) 

    vector = np.array(response)
    if vector.ndim > 1:
        vector = vector[0]
    return vector

def get_embeddings_batch(texts: list[str], client: InferenceClient) -> list[np.ndarray]:
    return [get_embedding(t, client) for t in texts]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:

    dot_production = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return float(dot_production / ( magnitude_a * magnitude_b))


def find_top_k(
        query_vec: np.ndarray,
        corpus_vecs: list[np.ndarray],
        corpus_texts: list[np.ndarray],
        k: int = 3
) -> list[dict] :
    
    scopes = [
        {"text": text, "score": cosine_similarity(query_vec, vec), "index": i}
        for i, (text, vec) in enumerate(zip(corpus_texts, corpus_vecs))
    ]

    scopes.sort(key=lambda x:x["score"], reverse=True)
    return scopes[:k]


def demo_similarity_intuition(client: InferenceClient):
    """
    Test your intuition: which pairs should be most similar?
    Run this and see if the numbers match what you expect.
    """
    pairs = [
        ("I love cats", "I adore kittens"),             # very similar meaning
        ("I love cats", "Cats are my favorite animals"), # similar meaning
        ("I love cats", "The stock market crashed"),     # unrelated
        ("Machine learning is hard", "ML is difficult"), # paraphrase
        ("Python is a programming language", "Python is a snake"), # same word, different meaning
    ]

    print("=" * 65)
    print("SIMILARITY INTUITION TEST")
    print("=" * 65)
    print(f"{'Pair':<55} {'Score':>6}")
    print("-" * 65)

    for text_a, text_b in pairs:
        vec_a = get_embedding(text_a, client)
        vec_b = get_embedding(text_b, client)
        score = cosine_similarity(vec_a, vec_b)
        pair_display = f'"{text_a[:20]}..." vs "{text_b[:20]}..."'
        print(f"{pair_display:<55} {score:>6.3f}")

    print()
    print("Notice the last pair — 'Python the language' vs 'Python the snake'.")
    print("This is a known limitation of sentence embeddings. Keep it in mind.")


def demo_semantic_search(client: InferenceClient):
    """
    Build a tiny in-memory semantic search engine.
    20 sentences. Query it. See what comes back.
    """
    corpus = [
        "Python is a popular programming language used in data science.",
        "Java is widely used for building enterprise applications.",
        "Machine learning models require large amounts of training data.",
        "Neural networks are inspired by the structure of the human brain.",
        "The stock market saw significant volatility this quarter.",
        "Apple released a new iPhone model with improved camera features.",
        "Climate change is causing more frequent extreme weather events.",
        "Docker containers make it easier to deploy applications consistently.",
        "The Indian Premier League cricket season starts in April.",
        "React is a JavaScript library for building user interfaces.",
        "Large language models can generate human-like text.",
        "Kubernetes helps manage containerized applications at scale.",
        "Inflation rates have been rising across major economies.",
        "Transfer learning allows models to reuse learned knowledge.",
        "The Himalayas are the highest mountain range in the world.",
        "FastAPI is a modern Python web framework for building APIs.",
        "Vector databases store and search high-dimensional embeddings.",
        "GPT models are trained using reinforcement learning from human feedback.",
        "SQL is used to query relational databases.",
        "Embeddings represent text as dense numerical vectors.",
    ]

    queries = [
        "How do I deploy software using containers?",
        "What are language models trained on?",
        "Tell me about sports in India.",
        "How do I store and search vectors?",
    ]

    print("\n" + "=" * 65)
    print("IN-MEMORY SEMANTIC SEARCH")
    print("=" * 65)

    print("Generating embeddings for corpus...")
    corpus_vecs = get_embeddings_batch(corpus, client)
    print(f"Done. {len(corpus_vecs)} vectors, each {len(corpus_vecs[0])} dimensions.\n")

    for query in queries:
        query_vec = get_embedding(query, client)
        results = find_top_k(query_vec, corpus_vecs, corpus, k=3)

        print(f"Query: \"{query}\"")
        for rank, r in enumerate(results, 1):
            print(f"  {rank}. [{r['score']:.3f}] {r['text']}")
        print()

def demo_vector_arithmetic(client: InferenceClient):
    """
    Famous example: King - Man + Woman ≈ Queen
    This shows embeddings capture semantic relationships, not just similarity.
    Works best with word embeddings — sentence embeddings are less precise here,
    but the concept is what matters for interviews.
    """
    print("=" * 65)
    print("VECTOR ARITHMETIC (concept demo)")
    print("=" * 65)

    words = ["king", "man", "woman", "queen", "prince", "princess"]
    vecs = {w: get_embedding(w, client) for w in words}

    # king - man + woman should be close to queen
    result_vec = vecs["king"] - vecs["man"] + vecs["woman"]

    candidates = ["queen", "prince", "princess"]
    print("king - man + woman = ?")
    for c in candidates:
        score = cosine_similarity(result_vec, vecs[c])
        print(f"  Similarity to '{c}': {score:.3f}")

    print()
    print("The highest score should be 'queen'. If it's not, it's because")
    print("sentence-transformers optimize for sentences, not single words.")
    print("With word2vec or GloVe embeddings this works almost perfectly.")

def main():
    client = InferenceClient(token=os.getenv("HUGGING_FACE_API"))

    # Run all three demos
    demo_similarity_intuition(client)
    demo_semantic_search(client)
    demo_vector_arithmetic(client)

    # Print a raw vector so you see what it actually looks like
    print("\n" + "=" * 65)
    print("RAW VECTOR INSPECTION")
    print("=" * 65)
    vec = get_embedding("Hello world", client)
    print(f"Text: 'Hello world'")
    print(f"Vector dimensions: {len(vec)}")
    print(f"First 10 values: {vec[:10].round(4)}")
    print(f"Min: {vec.min():.4f} | Max: {vec.max():.4f} | Mean: {vec.mean():.4f}")


if __name__ == "__main__":
    main()