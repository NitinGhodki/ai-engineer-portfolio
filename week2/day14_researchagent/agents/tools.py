"""Shared tools for all agents. Uses HybridRAG from core."""

import math
import datetime
from langchain_core.tools import tool

def _clean(text: str) -> str:
    return text.split("\n")[0].strip()

def make_search_tool(rag):
    """Factory: creates search tool bound to a specific RAG instance."""

    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Search the knowledge base for relevant information.
        Use for pricing, features, policies, specifications, support details.
        Input: specific search query string.
        """
        query = _clean(query)
        results = rag.search(query, top_k=3)
        if not results:
            return "No relevant information found."
        # print("search_knowledge_base", results)
        return "\n\n---\n\n".join([
            f"[Relevance: {r['score']:.4f}]\n{r['text']}"
            for r in results
        ])
    
    @tool
    def search_competitors(company: str) -> str:
        """
        Search for competitor pricing and feature information.
        Use when comparing our products with competitors.
        Input: competitor name or 'all'.
        """
        company = _clean(company)
        results = rag.search(f"competitor {company} pricing", top_k=2)
        if not results:
            return f"No competitor info found for: {company}"
        # print("search_competitors", results)
        return "\n".join(r["text"] for r in results)

    return search_knowledge_base, search_competitors


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a math expression.
    Use for price calculations and percentages.
    Input: expression like '2999 * 0.8'.
    """
    expression = _clean(expression)
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        return str(round(float(eval(expression, {"__builtins__": {}}, allowed)), 4))
    except Exception as e:
        return f"Calculator error: {e}"


@tool
def get_date(query: str = "") -> str:
    """Get today's date. Use when date context is needed."""
    return datetime.datetime.now().strftime("%Y-%m-%d, %A")


@tool
def format_table(data: str) -> str:
    """
    Format pipe-separated data as a readable comparison table.
    Input format: 'header1|header2|header3\nval1|val2|val3\nval4|val5|val6'
    Use for: side-by-side comparisons of plans or features.
    """
    lines = [l.strip() for l in data.strip().split("\n") if l.strip()]
    if not lines:
        return "No data to format."
    rows = [l.split("|") for l in lines]
    widths = [max(len(str(r[i])) for r in rows if i < len(r)) for i in range(len(rows[0]))]
    result = []
    for i, row in enumerate(rows):
        line = " | ".join(str(c).ljust(widths[j]) for j, c in enumerate(row))
        result.append(line)
        if i == 0:
            result.append("-" * len(line))
    return "\n".join(result)