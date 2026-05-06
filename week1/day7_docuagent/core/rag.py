"""
RAG pipeline — ingestion and retrieval.
Combines everything from Day 3 and Day 4.
"""

import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class RAGPipeline:
    """
    Handles document ingestion and semantic retrieval.
    Persistent ChromaDB — survives server restarts.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._persist_dir = persist_dir
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._vectorstore = self._load_or_create_store()

    def _load_or_create_store(self) -> Chroma:
        """Load existing store from disk or create a new one."""

        store = Chroma(
            collection_name="docuagent",
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
        )
        count = store._collection.count()
        print(f"[RAG] Vector store loaded. Existing chunks: {count}")
        return store
    
    def ingest_text(self, text: str, doc_name: str) -> int:
        """
        Chunk and embed a text document.
        Returns number of chunks created.
        """
        raw_doc = Document(
            page_content=text,
            metadata={"source": doc_name},
        )
        chunks = self._splitter.split_documents([raw_doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        self._vectorstore.add_documents(chunks)
        print(f"[RAG] Ingested '{doc_name}': {len(chunks)} chunks")
        return len(chunks)
    
    def ingest_file(self, file_path: str) -> int:
        """Ingest a text file from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        text = path.read_text(encoding="utf-8")
        return self.ingest_text(text, doc_name=path.name)


    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Semantic search — returns top_k most relevant chunks.
        Each result has: text, source, relevance score.
        """
        results = self._vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "score": round(float(1 - score), 4),
            }
            for doc, score in results
        ]
    
    def get_retriever(self, top_k: int = 3):
        """Return a LangChain retriever for use in chains."""
        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
    
    def chunk_count(self) -> int:
        return self._vectorstore._collection.count()

    def clear(self):
        """Clear all documents from the store."""
        self._vectorstore.delete_collection()
        self._vectorstore = self._load_or_create_store()
        print("[RAG] Vector store cleared.")