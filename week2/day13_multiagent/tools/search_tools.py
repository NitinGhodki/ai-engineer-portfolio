"""
Shared tools available to agents.
Each agent only gets the tools relevant to its job.
"""

import math
import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Build a small in-memory knowledge base
KNOWLEDGE_BASE = """
TechCorp AI Assistant — Product Documentation

Pricing and Plans:
Our Starter plan costs 999 rupees per month and includes 100 AI queries per day.
The Professional plan costs 2999 rupees per month with unlimited queries and priority support.
The Enterprise plan is custom priced with dedicated infrastructure and SLA guarantees.
Annual plans offer 20 percent discount compared to monthly billing.

Refund Policy:
All plans come with a 14-day free trial. No credit card required for trial.
Refunds are available within 30 days of first payment for annual plans.
Monthly plans can be cancelled anytime but are not eligible for partial refunds.

Technical Specifications:
Our API supports REST and WebSocket connections.
Rate limits are 10 requests per second for Starter, 50 for Professional.
Maximum document size for upload is 10MB. Supported formats are PDF and TXT.
Response latency SLA is under 2 seconds for 95th percentile on Professional plan.

Support:
Starter plan support is via email with 48-hour response time.
Professional plan includes live chat support with 4-hour response time.
Enterprise customers get a dedicated support engineer and 1-hour response SLA.

Competitors:
CompetitorA charges 1200 rupees per month for basic tier with 50 queries per day.
CompetitorB charges 2500 rupees per month for professional tier with unlimited queries.
CompetitorB does not offer a free trial. CompetitorA offers 7-day trial only.
"""

def build_knowledge_store():
    """Build ChromaDB from knowledge base. Returns retriever."""
    docs = [Document(page_content=KNOWLEDGE_BASE)]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=30
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    store = Chroma.from_documents(
        chunks, embeddings,
        collection_name="multiagent_kb",
    )
    return store

# Build store once at module level — shared across all agents
_store = None

def get_store():
    global _store
    if _store is None:
        print("  [Tools] Building knowledge store...")
        _store = build_knowledge_store()
    return _store


# ── Tool definitions

def _clean(text: str) -> str:
    return text.split("\n")[0].strip()


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the product knowledge base for relevant information.
    Use for: pricing, features, policies, specifications, support details.
    Input: specific search query string.
    """

    query = _clean(query)
    store = get_store()
    results = store.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in knowledge base."
    
    context = "\n\n---\n\n".join(r.page_content for r in results)
    return f"Knowledge base results:\n{context}"

@tool
def search_competitor_info(company_name: str) -> str:
    """
    Search for competitor pricing and feature information.
    Use for: comparing our plans with competitors.
    Input: competitor name or 'all competitors'.
    """
    company_name = _clean(company_name)
    store = get_store()
    query = f"competitor {company_name} pricing features"
    results = store.similarity_search(query, k=2)
    if not results:
        return f"No competitor information found for: {company_name}"
    return "\n".join(r.page_content for r in results)

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use for: price calculations, percentage discounts, comparisons.
    Input: math expression like '2999 * 0.8' or '(2999 - 2500) / 2999 * 100'.
    """
    expression = _clean(expression)
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(float(result), 4))
    except Exception as e:
        return f"Calculator error: {e}"

@tool
def get_current_date(query: str = "") -> str:
    """Get the current date. Use when date context is needed."""
    return datetime.datetime.now().strftime("%Y-%m-%d, %A")


@tool
def format_as_table(data: str) -> str:
    """
    Format pipe-separated data as a readable comparison table.
    Input format: 'header1|header2|header3\nval1|val2|val3\nval4|val5|val6'
    Use for: side-by-side comparisons of plans or features.
    """
    data = data.strip()
    lines = [line.strip() for line in data.split("\n") if line.strip()]
    if not lines:
        return "No data to format."

    rows = [line.split("|") for line in lines]
    if not rows:
        return data

    col_widths = [max(len(str(row[i])) for row in rows if i < len(row)) for i in range(len(rows[0]))]

    table_lines = []
    for i, row in enumerate(rows):
        line = " | ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row))
        table_lines.append(line)
        if i == 0:
            table_lines.append("-" * len(line))

    return "\n".join(table_lines)