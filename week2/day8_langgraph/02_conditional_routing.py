"""
Conditional routing using a real LLM to classify intent.
This is the production version of Graph 2 — LLM decides the route.
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("Hugging_face_model"),
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        max_new_tokens=20,
        temperature=0.0,
    ))


class RouterState(TypedDict):
    user_input: str
    intent: str           # classified intent
    rag_answer: str       # populated if intent=document_query
    tool_answer: str      # populated if intent=calculation
    direct_answer: str    # populated if intent=general_chat
    final_response: str   # always populated at the end


def llm_classifier(state: RouterState) -> dict:
    """
    Use LLM to classify intent.
    Strict prompt = one-word response = easy to parse.
    """
    prompt = f"""Classify this user input into exactly one category.
        Return ONLY the category name, nothing else.

        Categories:
        - document_query: asking about specific facts, policies, or documents
        - calculation: asking to compute or calculate something numeric
        - general_chat: greetings, opinions, general knowledge questions

        User input: "{state['user_input']}"
        Category:"""

    response = llm.invoke(prompt).content.strip()
    # response = response.lo

    # Clean and validate
    valid_intents = ["document_query", "calculation", "general_chat"]
    intent = "general_chat"  # default
    for v in valid_intents:
        if v in response:
            intent = v
            break

    print(f"  [LLM Classifier] '{state['user_input'][:40]}' → {intent!r}")
    return {"intent": intent}


def route_by_intent(state: RouterState) -> Literal["rag_node", "calc_node", "chat_node"]:
    """Conditional edge: maps intent to node name."""
    mapping = {
        "document_query": "rag_node",
        "calculation": "calc_node",
        "general_chat": "chat_node",
    }
    return mapping.get(state["intent"], "chat_node")

def rag_node(state: RouterState) -> dict:
    """Would call your RAG pipeline. Simulated here."""
    print(f"  [RAG Node] searching documents for: {state['user_input']!r}")
    answer = f"[From documents] Information about: {state['user_input']}"
    return {"rag_answer": answer, "final_response": answer}

def calc_node(state: RouterState) -> dict:
    """Extracts and evaluates math expression."""
    print(f"  [Calc Node] calculating: {state['user_input']!r}")
    try:
        # Extract digits and operators
        import re
        user_input = state["user_input"].lower()
        
        expr = user_input.replace('of', '*').replace('x', '*')
        expr = re.sub(r'(\d+(\.\d+)?)%', r'(\1/100)', expr)
        expr = re.sub(r'[^0-9+\-*/().\s]', '', expr).strip()
        expr = re.sub(r'(\))\s*(\d)', r'\1*\2', expr)
        expr = expr.strip('+-*/. ')
        print(f"calculation for {expr}")
        if expr:
            result = eval(expr, {"__builtins__": {}})
            answer = f"The result is: {result}"
        else:
            answer = f"Could not extract a calculable expression from: {state['user_input']}"
    except Exception as e:
        answer = f"Calculation error: {e}"
    return {"tool_answer": answer, "final_response": answer}

def chat_node(state: RouterState) -> dict:
    """Handles general conversation."""
    print(f"  [Chat Node] responding to: {state['user_input']!r}")
    prompt = f"Answer this conversationally in one sentence: {state['user_input']}"
    answer = llm.invoke(prompt).content.strip()
    return {"direct_answer": answer, "final_response": answer}


def build_llm_router():
    graph = StateGraph(RouterState)

    graph.add_node("classifier", llm_classifier)
    graph.add_node("rag_node", rag_node)
    graph.add_node("calc_node", calc_node)
    graph.add_node("chat_node", chat_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier",
        route_by_intent,
        {
            "rag_node": "rag_node",
            "calc_node": "calc_node",
            "chat_node": "chat_node",
        },
    )
    graph.add_edge("rag_node", END)
    graph.add_edge("calc_node", END)
    graph.add_edge("chat_node", END)

    return graph.compile()


def main():
    print("="*60)
    print("LLM-POWERED CONDITIONAL ROUTER")
    print("="*60)

    router = build_llm_router()

    test_cases = [
        "What is the refund policy?",       # → rag_node
        "What is 18% of 45000?",            # → calc_node
        "Hello, how are you today?",        # → chat_node
        "Calculate 250 * 12",               # → calc_node
        "Who created Python language?",     # → general_chat or rag
    ]

    for inp in test_cases:
        print(f"\nInput: {inp!r}")
        result = router.invoke({
            "user_input": inp,
            "intent": "",
            "rag_answer": "",
            "tool_answer": "",
            "direct_answer": "",
            "final_response": "",
        })
        print(f"Route: {result['intent']}")
        print(f"Answer: {result['final_response'][:100]}")


if __name__ == "__main__":
    main()