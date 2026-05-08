"""
ReAct agent rebuilt as an explicit LangGraph.

In Day 6 you built this as a manual loop and as AgentExecutor.
Today you build it as a LangGraph graph — every step is a visible node.

Graph structure:
START → llm_node → tool_node → llm_node (loop) → END
                 ↓
           (if final answer)
                 ↓
               END
"""

import os
import math
import json
import datetime
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool

load_dotenv()

# ── Tools
def _clean(text: str) -> str:
    return text.split("\n")[0].strip()


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input: expression like '2 + 2'."""
    expression = _clean(expression)
    try:
        raise ValueError("calulator fail")
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        return str(round(eval(expression, {"__builtins__": {}}, allowed), 4))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_date(query: str = "") -> str:
    """Get the current date. Use when date or month or year is needed."""
    try:
        raise ValueError("get_date fail")
        return datetime.datetime.now().strftime("%Y-%m-%d, %A, %B %d %Y")
    except Exception as e:
            return f"Error: {e}"

TOOLS = [calculator, get_date]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

class AgentState(TypedDict):
    """
    Annotated[list, add_messages] is a LangGraph reducer.
    Instead of replacing the list on each update,
    add_messages APPENDS new messages to the existing list.
    This is how the conversation history accumulates automatically.
    """
    messages: Annotated[list, add_messages]
    iterations: int


# ── Nodes
def llm_node(state: AgentState) -> dict:
    """
    Core LLM node — calls the model with current message history.
    The LLM sees all previous messages including tool results.
    Returns either a tool call or a final answer.
    """

    llm = ChatHuggingFace(
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=80,
            temperature=0.0,
        ))
    
    # Build prompt from message history
    history_text = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history_text += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"
        elif isinstance(msg, ToolMessage):
            history_text += f"Tool result: {msg.content}\n"

    tools_desc = "\n".join([
        f"- {t.name}: {t.description}" for t in TOOLS
    ])

    prompt = f"""You are a helpful assistant with tools.
        You must respond EXCLUSIVELY with a single JSON object. 
        DO NOT include any introductory text, markdown code blocks (like ```json), or footers.

        Available tools:
        {tools_desc}
        
        To use a tool: {{"action": "tool_name", "input": "value"}}
        To give final answer: {{"action": "final_answer", "input": "your answer"}}

        {history_text}JSON:"""

    response = llm.invoke(prompt).content.strip()
    print(f"  [LLM Node] iteration {state['iterations'] + 1}: {response[:80]!r}")

    return {
        "messages": [AIMessage(content=response)],
        "iterations": state["iterations"] + 1,
    }

def tool_node(state: AgentState) -> dict:
    """
    Tool execution node.
    Reads the last AIMessage, parses the tool call, executes it,
    appends the result as a ToolMessage.
    """
    last_message = state["messages"][-1]
    content = last_message.content.strip()

    # Clean and parse JSON
    if "```" in content:
        content = content.split("```")[1].strip()
        if content.startswith("json"):
            content = content[4:].strip()

    start = content.find("{")
    end = content.rfind("}") + 1

    try:
        data = json.loads(content[start:end])
        tool_name = data.get("action", "")
        tool_input = str(data.get("input", ""))
    except Exception:
        return {"messages": [ToolMessage(content="Error: could not parse tool call", tool_call_id="error")]}

    if tool_name not in TOOLS_BY_NAME:
        result = f"Unknown tool: {tool_name}"
    else:
        result = TOOLS_BY_NAME[tool_name].invoke(tool_input)

    print(f"  [Tool Node] {tool_name}({tool_input!r}) → {result!r}")
    return {"messages": [ToolMessage(content=str(result), tool_call_id=tool_name)]}


def should_continue(state: AgentState) -> str:
    """
    Conditional edge: reads last LLM message.
    If it contains 'final_answer' → END
    If max iterations → END
    Otherwise → tool_node (continue loop)
    """
    if state["iterations"] >= 6:
        print(f"  [Router] max iterations reached → END")
        return "end"

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        content = last_message.content
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end])
            if data.get("action") == "final_answer":
                print(f"  [Router] final answer detected → END")
                return "end"
            else:
                print(f"  [Router] tool call detected → tool_node")
                return "tool"
        except Exception:
            print(f"  [Router] parse error, treating as final answer → END")
            return "end"

    return "end"

def extract_final_answer(state: AgentState) -> str:
    """Extract the final answer from the last AIMessage."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            try:
                content = msg.content
                start = content.find("{")
                end = content.rfind("}") + 1
                data = json.loads(content[start:end])
                if data.get("action") == "final_answer":
                    return data["input"]
            except Exception:
                return msg.content
    return "No answer produced"

# ── Build graph


def build_react_graph():
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {"tool": "tools", "end": END},
    )
    graph.add_edge("tools", "llm")   # after tool → back to LLM

    return graph.compile()

def main():
    graph = build_react_graph()

    questions = [
        "What is 23% of 18500?",
        "What year is it and how many years since 2000?",
        "What is the capital of India?",  # no tool needed
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("="*60)

        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "iterations": 0,
        })

        final = extract_final_answer(result)
        print(f"\nFinal Answer: {final}")
        print(f"Total messages: {len(result['messages'])}")
        print(f"Iterations: {result['iterations']}")


if __name__ == "__main__":
    main()