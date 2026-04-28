"""
LCEL data flow — inspect what each step produces.
This makes the chain transparent. No more black box.

Run this file directly: python 02_lcel_data_flow.py
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# setup with 5 documents
def setup_minimal_vectorstore():
    docs = [
        Document(page_content="Python is a programming language created in 1991.", metadata={"source": "python"}),
        Document(page_content="Machine learning uses data to train predictive models.", metadata={"source": "ml"}),
        Document(page_content="FastAPI is a Python web framework for building APIs.", metadata={"source": "fastapi"}),
        Document(page_content="Vector databases store embeddings for semantic search.", metadata={"source": "vectordb"}),
        Document(page_content="LangChain is a framework for building LLM applications.", metadata={"source": "langchain"}),
    ]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

# Build each piece separately
def demo_each_component_separately():
    """
    Test each chain component in isolation.
    This is how you debug when the chain breaks.
    """
    print("="*60)
    print("TESTING EACH COMPONENT SEPARATELY")
    print("="*60)

    vectorstore = setup_minimal_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k":2})

    # ── Component 1: Retriever 
    print("\n--- Component 1: Retriever ---")
    print("Input:  str (question)")
    print("Output: list[Document]")

    question = "What is LangChain used for?"
    retrieved_docs = retriever.invoke(question)
    print(f"\nInput:  {question!r}")
    print(f"Output type: {type(retrieved_docs)}")
    print(f"Output length: {len(retrieved_docs)} documents")
    for i, doc in enumerate(retrieved_docs):
        print(f"    Doc {i}: {doc.page_content!r}")
        print(f"        meradata: {doc.metadata}")

# ── Component 2: format_docs 
    print("\n--- Component 2: format_docs ---")
    print("Input:  list[Document]")
    print("Output: str (joined context)")

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ])

    context_str = format_docs(retrieved_docs)
    print(f"\nOutput:\n{context_str}")

    # ── Component 3: ChatPromptTemplate 
    print("\n--- Component 3: ChatPromptTemplate ---")
    print("Input:  dict with 'context' and 'question' keys")
    print("Output: ChatPromptValue (formatted prompt object)")

    template = """Answer using ONLY this context:

        {context}

        Question: {question}
        Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    prompt_input = {"context": context_str, "question": question}
    prompt_output = prompt.invoke(prompt_input)

    print(f"\nInput dict: {{'context': '...{len(context_str)} chars...', 'question': {question!r}}}")
    print(f"Output type: {type(prompt_output)}")
    print(f"Output messages: {prompt_output.messages}")
    print(f"\nFull formatted prompt:\n{prompt_output.messages[0].content}")

    # ── Component 4: LLM 
    print("\n--- Component 4: LLM ---")
    print("Input:  ChatPromptValue")
    print("Output: AIMessage object (not a plain string yet)")

    llm_main = HuggingFaceEndpoint(
        repo_id=os.getenv("Hugging_face_model"),
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        max_new_tokens=150,
        temperature=0.1,
    )
    
    llm = ChatHuggingFace(llm=llm_main)
    llm_output = llm.invoke(prompt_output)
    print(f"\nOutput type: {type(llm_output)}")
    print(f"Output value: {llm_output!r}")

    # ── Component 5: StrOutputParser 
    print("\n--- Component 5: StrOutputParser ---")
    print("Input:  str or AIMessage")
    print("Output: plain str")

    parser = StrOutputParser()
    final = parser.invoke(llm_output)
    print(f"\nOutput type: {type(final)}")
    print(f"Output value: {final!r}")

    return vectorstore

# ── Now build the chain and trace data through it 

def demo_chain_with_tracing(vectorstore):
    """
    Build the full LCEL chain with a tracing wrapper
    so you can see exactly what flows between each step.
    """
    print("\n" + "="*60)
    print("FULL CHAIN WITH DATA FLOW TRACING")
    print("="*60)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def trace(label: str):
        """
        Wraps any step with a print statement showing input/output.
        RunnableLambda converts a plain function into a LangChain Runnable.
        """
        def _trace(x):
            print(f"\n[TRACE] {label}")
            print(f"  Input type:  {type(x).__name__}")
            if isinstance(x, str):
                print(f"  Input value: {x[:80]!r}")
            elif isinstance(x, dict):
                for k, v in x.items():
                    val = str(v)[:60]
                    print(f"  Input[{k!r}]: {val!r}")
            elif isinstance(x, list):
                print(f"  Input: list of {len(x)} items")
            return x  # pass through unchanged
        return RunnableLambda(_trace)

    prompt = ChatPromptTemplate.from_template(
        """Answer using ONLY this context:
        {context}

        Question: {question}
        Answer:"""
    )

    llm_main = HuggingFaceEndpoint(
        repo_id=os.getenv("Hugging_face_model"),
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        max_new_tokens=150,
        temperature=0.1,
    )
    llm = ChatHuggingFace(llm = llm_main)
    # Chain with tracing at every step
    chain = (
        trace("1. Input question")
        | RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        | trace("2. After RunnableParallel")
        | prompt
        | trace("3. After PromptTemplate")
        | llm
        | trace("4. After LLM")
        | StrOutputParser()
        | trace("5. After StrOutputParser")
    )

    question = "What is a vector database?"
    print(f"\nRunning chain with question: {question!r}")
    result = chain.invoke(question)
    print(f"\n{'='*60}")
    print(f"FINAL ANSWER: {result}")


# ── RunnableParallel explained in isolation
def demo_runnable_parallel():
    """
    RunnableParallel is the most confusing part of LCEL.
    Test it in isolation with simple functions.
    """
    print("\n" + "="*60)
    print("RUNNABLEPARALLEL EXPLAINED")
    print("="*60)

    # Simple example: RunnableParallel with two lambda functions
    parallel = RunnableParallel({
        "upper": RunnableLambda(lambda x: x.upper()),
        "length": RunnableLambda(lambda x: len(x)),
        "original": RunnablePassthrough(),
    })

    result = parallel.invoke("hello world")
    print(f"\nInput: 'hello world'")
    print(f"Output: {result}")
    print()
    print("KEY INSIGHT: RunnableParallel takes ONE input and runs")
    print("ALL branches with that SAME input simultaneously.")
    print("Output is a dict with results from each branch.")
    print()
    print("In the RAG chain:")
    print("  input = the user's question (str)")
    print("  'context' branch: question → retriever → format_docs → str")
    print("  'question' branch: question → passthrough → same str")
    print("  output = {'context': '...chunks...', 'question': 'What is...'}")
    print("  This dict then fills the prompt template's {context} and {question}")


if __name__ == "__main__":
    vectorstore = demo_each_component_separately()
    demo_chain_with_tracing(vectorstore)
    demo_runnable_parallel()