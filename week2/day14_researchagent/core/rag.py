"""
Hybrid search RAG — BM25 + vector with RRF combination.
Pulled from Day 12 and adapted for production use.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()


class HybridRAG:
    """
    Production RAG with hybrid search.
    BM25 handles exact terms. Vector handles semantics. RRF combines both.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._persist_dir = persist_dir
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=30,
        )
        self._chunks: list[str] = []
        self._bm25 = None
        self._vectorstore = None
        self._doc_vectors = None

    def ingest(self, text: str, doc_name: str = "document") -> int:
        """Ingest text into both BM25 and vector indices."""
        raw = Document(page_content=text, metadata={"source": doc_name})
        chunks = self._splitter.split_documents([raw])

        # Store chunk texts for BM25
        new_texts = [c.page_content for c in chunks]
        self._chunks.extend(new_texts)

        # Rebuild BM25 index
        tokenised = [t.lower().split() for t in self._chunks]
        self._bm25 = BM25Okapi(tokenised)

        # Add to vector store
        if self._vectorstore is None:
            self._vectorstore = Chroma.from_documents(
                chunks, self._embeddings,
                persist_directory=self._persist_dir,
                collection_name="researchagent",
            )
        else:
            self._vectorstore.add_documents(chunks)

        # Recompute all vectors for RRF
        self._doc_vectors = self._bi_encoder.encode(
            self._chunks, normalize_embeddings=True
        )

        print(f"[RAG] Ingested '{doc_name}': {len(chunks)} chunks. Total: {len(self._chunks)}")
        return len(chunks)

    def ingest_file(self, file_path: str) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return self.ingest(path.read_text(encoding="utf-8"), doc_name=path.name)

    def search(self, query: str, top_k: int = 3, alpha: float = 0.4) -> list[dict]:
        """
        Hybrid search using RRF.
        alpha=0.4: slightly BM25-dominant for product docs with exact terms.
        Returns top_k most relevant chunks with scores.
        """
        if not self._chunks:
            return []

        # BM25 scores
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ranks = {
            i: rank + 1
            for rank, (i, _) in enumerate(
                sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
            )
        }

        # Vector scores
        query_vec = self._bi_encoder.encode(query, normalize_embeddings=True)
        vec_scores = np.dot(self._doc_vectors, query_vec)
        vec_ranks = {
            i: rank + 1
            for rank, (i, _) in enumerate(
                sorted(enumerate(vec_scores), key=lambda x: x[1], reverse=True)
            )
        }

        # RRF combination
        k = 60
        n = len(self._chunks)
        rrf_scores = {}
        for i in range(n):
            bm25_r = bm25_ranks.get(i, n + 1)
            vec_r = vec_ranks.get(i, n + 1)
            rrf_scores[i] = (
                (1 - alpha) * (1 / (k + bm25_r)) +
                alpha * (1 / (k + vec_r))
            )

        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "text": self._chunks[idx],
                "score": round(score, 6),
                "rank": rank + 1,
                "bm25_rank": bm25_ranks.get(idx, n),
                "vec_rank": vec_ranks.get(idx, n),
            }
            for rank, (idx, score) in enumerate(top)
        ]

    def chunk_count(self) -> int:
        return len(self._chunks)