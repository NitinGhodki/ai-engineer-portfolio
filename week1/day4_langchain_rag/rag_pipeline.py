import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

#documents
import tempfile, pathlib

DOC_AI = """
Artificial intelligence refers to the simulation of human intelligence in machines.
Machine learning is a subset of AI that learns from data without explicit programming.
Deep learning uses neural networks with many layers to process complex patterns.
Large language models like GPT-4 and Claude are trained on massive text datasets.
They learn to predict the next token and develop broad language understanding.
Retrieval Augmented Generation combines LLMs with external knowledge bases.
RAG retrieves relevant documents and injects them as context into the LLM prompt.
Vector databases store embeddings and support semantic similarity search.
Embeddings are dense numerical representations of text meaning.
Cosine similarity measures how close two embedding vectors are in meaning.
"""

DOC_POLICY = """
Our refund policy allows full refunds within 30 days of purchase.
After 30 days refunds are evaluated case by case by our support team.
Contact support@example.com with your order ID to initiate a refund.
Premium subscribers get priority support with 4 hour response time.
Standard users receive responses within 24 to 48 business hours.
Account cancellations can be done through account settings.
Data is retained for 90 days after cancellation then permanently deleted.
Our SLA guarantees 99.9 percent uptime for all paid plans.
Downtime beyond SLA threshold results in service credits of 10x the downtime.
"""

def write_temp_docs() -> list[str]:
    paths = []
    for name, content in [("ai_overview.txt", DOC_AI), ("policy.txt", DOC_POLICY)]:
        path = pathlib.Path(f"tmp/{name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        paths.append(str(path))
    return paths

# load document
def load_documents(file_paths: list[str]):
    """
    TextLoader reads a file and returns a LangChain Document object.
    Document = {page_content: str, metadata: {source: filepath}}
    LangChain automatically adds 'source' to metadata.
    """
    docs = []
    for path in file_paths:
        loader = TextLoader(path)
        loaded = loader.load()
        docs.extend(loaded)
        print(f"Loaded: {path} → {len(loaded[0].page_content)} chars")
    return docs

# slpit into chuncks
def split_documents(documents):
    """
    RecursiveCharacterTextSplitter tries to split on paragraphs first,
    then sentences, then words — in that order. Smarter than a fixed word split.

    chunk_size=300: characters (not words — different from your Day 3 version)
    chunk_overlap=50: overlap in characters
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 50,
        length_function = len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Inspect first chunk — notice metadata is preserved automatically
    print(f"First chunk preview: {chunks[0].page_content[:80]!r}")
    print(f"First chunk metadata: {chunks[0].metadata}")
    return chunks

def build_vectorstore(chunks):
    """
    HuggingFaceEmbeddings: same sentence-transformers model you used in Day 3.
    Chroma.from_documents: embeds all chunks and stores them in one call.
    
    What LangChain saves you: 15 lines of embedding + storage code.
    What you lose: direct control over batch size, error handling per chunk.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./day4_chroma_db",
        collection_name="day4_rag",
    )
    print(f"Vectorstore built: {vectorstore._collection.count()} chunks stored")
    return vectorstore

def build_retriever(vectorstore):
    """
    A retriever wraps the vectorstore with a search interface.
    search_type="similarity": cosine similarity (what you implemented manually)
    k=3: return top 3 chunks

    Other search types you should know exist:
    - "mmr": Maximal Marginal Relevance — diverse results (avoids redundant chunks)
    - "similarity_score_threshold": only return chunks above a score threshold
    """

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    print("Retriever ready")
    return retriever

def build_llm():
    """
    HuggingFaceEndpoint wraps the HF Inference API.
    Same model, same API key — just LangChain's interface.
    """
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("Hugging_face_model"),
        huggingfacehub_api_token=os.getenv("HUGGING_FACE_API"),
        max_new_tokens=512,
        temperature=0.1,
    )
    return ChatHuggingFace(llm=llm)

# build prompt
def build_prompt():
    """
    ChatPromptTemplate defines the prompt structure.
    {context} and {question} are filled in at runtime by the chain.
    """
    template = """You are a helpful assistant. Answer using ONLY the context below.
        If the answer is not in the context, say exactly: "I don't have enough information."
        Do not use outside knowledge.

        CONTEXT:
        {context}

        QUESTION: {question}

        ANSWER:"""

    return ChatPromptTemplate.from_template(template)

#Build the LCEL chain
def build_chain(retriever, llm, prompt):
    """
    LCEL (LangChain Expression Language) pipes components together.

    RunnableParallel runs retriever and passthrough at the same time:
    - "context": retriever.invoke(question) → fetches relevant chunks
    - "question": RunnablePassthrough() → passes the original question unchanged

    Then feeds both into prompt → llm → StrOutputParser (extracts text from response)

    Reading this left to right:
    question → {retrieve context, keep question} → fill prompt → LLM → parse text
    """

    def format_docs(docs) -> str:
        """Convert list of Document objects to a single context string."""
        return "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ])

    chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── Step 9: Run everything ────────────────────────────────────────────────────

def build_rag_pipeline():
    """
    Full pipeline: load → split → embed → store → retrieve → answer.
    Returns a callable chain.
    """
    file_paths = write_temp_docs()
    documents = load_documents(file_paths)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)
    retriever = build_retriever(vectorstore)
    llm = build_llm()
    prompt = build_prompt()
    chain = build_chain(retriever, llm, prompt)
    return chain, vectorstore


if __name__ == "__main__":
    print("Building RAG pipeline...")
    chain, vectorstore = build_rag_pipeline()

    questions = [
        "What is retrieval augmented generation?",
        "How do I request a refund?",
        "What are vector databases used for?",
        "What is the SLA uptime guarantee?",
        "Who invented electricity?",  # not in docs
    ]

    print("\n" + "="*60)
    print("QUERYING")
    print("="*60)

    for q in questions:
        print(f"\nQ: {q}")
        answer = chain.invoke(q)
        print(f"A: {answer.strip()}")

        # Also show which chunks were retrieved
        retrieved = vectorstore.similarity_search(q, k=3)
        sources = list(set([
            doc.metadata.get("source", "unknown").split("/")[-1]
            for doc in retrieved
        ]))
        print(f"Sources: {sources}")