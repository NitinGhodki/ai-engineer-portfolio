"""
Day 9 — Human-in-the-loop using LangGraph interrupts.

How it works:
1. Graph runs normally until it hits an interrupt point
2. Graph PAUSES — execution stops completely
3. State is saved exactly as-is
4. Human reviews what's about to happen
5. Human approves or rejects
6. Graph RESUMES from the exact pause point

This is NOT the same as asking the LLM to "check with the user."
This is the graph literally stopping execution and waiting.

Real world use cases:
- AI agent about to execute code → pause, show code, get approval
- AI agent about to send email → pause, show draft, get approval
- AI agent about to call paid API → pause, show cost, get approval
"""

import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("Hugging_face_model"),
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        max_new_tokens=80,
        temperature=0.0,
    ))

# DEMO 1: Simple interrupt — pause before a dangerous action

class ActionState(TypedDict):
    user_request: str
    planned_action: str
    human_approved: bool
    execution_result: str
    status: str


def plan_action(state: ActionState):
    """Agent plans what it wants to do."""
    request = state["user_request"]
    print(f"  [Plan] analysing request: {request!r}")

    # Simulate LLM planning
    prompt = f"""Given this request, describe in one sentence what action you would take:
        Request: {request}
        Action:"""
    
    planned = llm.invoke(prompt).content.strip()
    print(f"  [Plan] planned action: {planned!r}")
    return {"planned_action": planned, "status": "planned"}

def human_review(state: ActionState) -> dict:
    """
    THIS IS THE KEY NODE.
    
    interrupt() pauses execution here.
    The graph state is saved to the checkpointer.
    Control returns to the caller.
    
    When the caller resumes (with Command(resume=...)),
    execution continues from this exact point.
    interrupt() returns whatever value the human provided.
    """

    print(f"\n  [Human Review] Pausing for approval...")
    print(f"  Planned action: {state['planned_action']}")

    # This line PAUSES the graph
    # The string argument is shown to whoever is resuming the graph
    human_decision = interrupt(
        f"Agent want to: {state['planned_action']}\nApprove? (yes/no)"
    )

    # Code below runs AFTER the human resumes the graph
    approved = str(human_decision).lower().strip() in ["yes", "y", "approve", "ok"]
    print(f"  [Human Review] Decision: {'APPROVED' if approved else 'REJECTED'}")
    return {"human_approved": approved, "status": "reviewed"}

def execute_action(state: ActionState) -> dict:
    """Executes only if approved."""
    if not state["human_approved"]:
        print(f"  [Execute] Action rejected by human — skipping")
        return {"execution_result": "Action cancelled by human", "status": "cancelled"}

    print(f"  [Execute] Executing approved action...")
    result = f"Successfully executed: {state['planned_action']}"
    print(f"  [Execute] {result}")
    return {"execution_result": result, "status": "completed"}

def build_hitl_graph():
    """
    CRITICAL: human-in-the-loop requires a checkpointer.
    The graph cannot pause without somewhere to save state.
    MemorySaver = in-memory checkpointer (fine for development).
    In production: use SqliteSaver or PostgresSaver.
    """
    graph = StateGraph(ActionState)

    graph.add_node("plan", plan_action)
    graph.add_node("human_review", human_review)
    graph.add_node("execute", execute_action)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "human_review")
    graph.add_edge("human_review", "execute")
    graph.add_edge("execute", END)

    checkpointer = MemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],  # pause BEFORE this node runs
    )


def demo_hitl_approved():
    """
    Run graph with human approval.
    Shows: initial run → interrupt → resume with 'yes'.
    """
    print("\n" + "="*60)
    print("DEMO 1A: Human-in-the-loop — APPROVED")
    print("="*60)

    graph = build_hitl_graph()

    # thread_id identifies this specific run
    # Same thread_id = same checkpoint = resume from same state
    config = {"configurable": {"thread_id": "demo_approved_1"}}

    # First run — will pause at human_review
    print("\n[PHASE 1] Starting graph...")
    result = graph.invoke(
        {
            "user_request": "Delete all records older than 30 days from the database",
            "planned_action": "",
            "human_approved": False,
            "execution_result": "",
            "status": "pending",
        },
        config=config,
    )
    print(f"\n[PHASE 1 COMPLETE] Graph paused.")
    print(f"Current status: {result.get('status', 'unknown')}")

    # Inspect current state — this is what you'd show the human
    current_state = graph.get_state(config)
    print(f"\nPaused at nodes: {current_state.next}")
    print(f"Planned action: {current_state.values.get('planned_action', 'N/A')}")

    # Human makes decision — simulate "yes"
    human_input = "yes"
    print(f"\n[HUMAN DECISION] '{human_input}'")

    # Resume graph with human's decision
    print("\n[PHASE 2] Resuming graph with human approval...")
    final_result = graph.invoke(
        Command(resume=human_input),  # resume with human's answer
        config=config,                # same thread_id = same checkpoint
    )

    print(f"\n[FINAL] Status: {final_result.get('status')}")
    print(f"[FINAL] Result: {final_result.get('execution_result')}")


def demo_hitl_rejected():
    """
    Same flow but human rejects the action.
    """
    print("\n" + "="*60)
    print("DEMO 1B: Human-in-the-loop — REJECTED")
    print("="*60)

    graph = build_hitl_graph()
    config = {"configurable": {"thread_id": "demo_rejected_1"}}

    print("\n[PHASE 1] Starting graph...")
    graph.invoke(
        {
            "user_request": "Send promotional email to all 50,000 users",
            "planned_action": "",
            "human_approved": False,
            "execution_result": "",
            "status": "pending",
        },
        config=config,
    )

    current_state = graph.get_state(config)
    print(f"\nPlanned action: {current_state.values.get('planned_action')}")
    print(f"\n[HUMAN DECISION] 'no' — rejecting dangerous action")

    final_result = graph.invoke(
        Command(resume="no"),
        config=config,
    )

    print(f"\n[FINAL] Status: {final_result.get('status')}")
    print(f"[FINAL] Result: {final_result.get('execution_result')}")


# DEMO 2: Multi-step graph with interrupt in the middle
# Shows: nodes before interrupt run, nodes after run only on resume

class PipelineState(TypedDict):
    raw_data: str
    cleaned_data: str
    analysis: str
    human_feedback: str
    final_report: str
    nodes_executed: list[str]


def data_cleaning(state: PipelineState) -> dict:
    """Runs before interrupt — always executes."""
    print(f"  [Data Cleaning] cleaning input data...")
    cleaned = state["raw_data"].strip().lower().replace("  ", " ")
    return {
        "cleaned_data": cleaned,
        "nodes_executed": state.get("nodes_executed", []) + ["data_cleaning"],
    }


def data_analysis(state: PipelineState) -> dict:
    """Runs before interrupt — always executes."""
    print(f"  [Analysis] analysing cleaned data...")
    analysis = f"Analysis of '{state['cleaned_data'][:50]}': contains {len(state['cleaned_data'].split())} words"
    return {
        "analysis": analysis,
        "nodes_executed": state.get("nodes_executed", []) + ["data_analysis"],
    }


def human_feedback_node(state: PipelineState) -> dict:
    """
    Interrupt point in the middle of the pipeline.
    Nodes before this already ran.
    Nodes after this run only after human responds.
    """
    print(f"\n  [Human Feedback] Pausing for review...")
    print(f"  Analysis so far: {state['analysis']}")

    feedback = interrupt(
        f"Please review this analysis and provide feedback:\n{state['analysis']}"
    )

    print(f"  [Human Feedback] Received: {feedback!r}")
    return {
        "human_feedback": str(feedback),
        "nodes_executed": state.get("nodes_executed", []) + ["human_feedback"],
    }


def report_generation(state: PipelineState) -> dict:
    """Runs after interrupt — only executes after human responds."""
    print(f"  [Report] generating final report with feedback...")
    report = (
        f"FINAL REPORT\n"
        f"Data: {state['cleaned_data'][:80]}\n"
        f"Analysis: {state['analysis']}\n"
        f"Human feedback incorporated: {state['human_feedback']}\n"
        f"Nodes executed: {' → '.join(state['nodes_executed'])}"
    )
    return {
        "final_report": report,
        "nodes_executed": state.get("nodes_executed", []) + ["report_generation"],
    }


def build_pipeline_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("clean", data_cleaning)
    graph.add_node("analyse", data_analysis)
    graph.add_node("feedback", human_feedback_node)
    graph.add_node("report", report_generation)

    graph.add_edge(START, "clean")
    graph.add_edge("clean", "analyse")
    graph.add_edge("analyse", "feedback")
    graph.add_edge("feedback", "report")
    graph.add_edge("report", END)

    return graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["feedback"],
    )


def demo_pipeline_with_feedback():
    print("\n" + "="*60)
    print("DEMO 2: Pipeline with mid-execution human feedback")
    print("="*60)

    graph = build_pipeline_graph()
    config = {"configurable": {"thread_id": "pipeline_demo_1"}}

    print("\n[PHASE 1] Running pipeline until feedback node...")
    state = graph.invoke(
        {
            "raw_data": "  The Quick Brown Fox   Jumped Over The Lazy Dog  ",
            "cleaned_data": "",
            "analysis": "",
            "human_feedback": "",
            "final_report": "",
            "nodes_executed": [],
        },
        config=config,
    )

    current = graph.get_state(config)
    print(f"\nPaused. Nodes executed so far: {current.values.get('nodes_executed')}")
    print(f"Next nodes: {current.next}")
    print(f"Analysis ready for review: {current.values.get('analysis')}")

    # Simulate human providing feedback
    feedback = "Looks good. Please emphasise the word count in the final report."
    print(f"\n[HUMAN] Providing feedback: {feedback!r}")

    print("\n[PHASE 2] Resuming with feedback...")
    final = graph.invoke(Command(resume=feedback), config=config)

    print(f"\n[FINAL REPORT]\n{final['final_report']}")


if __name__ == "__main__":
    demo_hitl_approved()
    demo_hitl_rejected()
    demo_pipeline_with_feedback()