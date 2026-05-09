"""
Combined: human-in-the-loop + checkpointing in one graph.
This is the production pattern.

Scenario: AI code reviewer
1. Analyse submitted code
2. Generate suggested changes
3. PAUSE → show human the suggestions
4. Human approves, rejects, or requests modifications
5. Apply changes
6. Generate final review report
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


class CodeReviewState(TypedDict):
    code_snippet: str
    issues_found: list[str]
    suggested_changes: str
    human_decision: str       # "approve", "reject", "modify"
    human_notes: str
    final_review: str
    status: str


def analyse_code(state: CodeReviewState) -> dict:
    print(f"  [Analyse] Scanning code for issues...")
    prompt = f"""Review this code and list all issues found. Be specific.
        Code:
        {state['code_snippet']}

        Issues (list format):"""
    
    response = llm.invoke(prompt).content.strip()
    issues = [line.strip() for line in response.split("\n") if line.strip()][:3]
    print(f"  [Analyse] Found {len(issues)} issues ✓")
    return {"issues_found": issues, "status": "analysed"}


def generate_suggestions(state: CodeReviewState) -> dict:
    print(f"  [Suggest] Generating improvement suggestions...")
    issues_text = "\n".join(state["issues_found"])

    prompt = f"""Based on these code issues, write specific improvement suggestions:
        Issues:
        {issues_text}

        Suggestions:"""
    suggestions = llm.invoke(prompt).content.strip()
    print(f"  [Suggest] Suggestions ready ✓")
    return {"suggested_changes": suggestions, "status": "suggestions_ready"}


def human_approval(state: CodeReviewState) -> dict:
    """Interrupt point — human reviews before changes are applied."""
    print(f"\n  [Human Approval] Pausing for review...")

    display = (
        f"CODE REVIEW READY FOR APPROVAL\n"
        f"{'='*40}\n"
        f"Issues found:\n" +
        "\n".join(f"  - {i}" for i in state["issues_found"]) +
        f"\n\nSuggested changes:\n{state['suggested_changes']}\n"
        f"{'='*40}\n"
        f"Enter: 'approve', 'reject', or 'modify: <your notes>'"
    )

    human_input = interrupt(display)

    # Parse human decision
    human_str = str(human_input).strip().lower()
    if human_str.startswith("modify:"):
        decision = "modify"
        notes = human_str[7:].split()
    elif "approve" in human_str:
        decision = "approve"
        notes = ""
    else:
        decision = "reject"
        notes = human_str
    
    print(f"  [Human Approval] Decision: {decision.upper()}")
    return {
        "human_decision": decision,
        "human_notes": notes,
        "status": "reviewed",
    }

def route_after_review(state: CodeReviewState) -> str:
    """Route based on human decision."""
    routes = {
        "approve": "apply_changes",
        "reject": "end_rejected",
        "modify": "generate_suggestions",  # loop back with human notes
    }
    route = routes.get(state["human_decision"], "end_rejected")
    print(f"  [Router] routing to: {route}")
    return route

def apply_changes(state: CodeReviewState) -> dict:
    print(f"  [Apply] Applying approved changes...")
    report = (
        f"CODE REVIEW COMPLETE — APPROVED\n"
        f"Original issues: {len(state['issues_found'])}\n"
        f"Changes applied: {state['suggested_changes'][:100]}...\n"
        f"Status: Changes merged successfully"
    )
    return {"final_review": report, "status": "approved_and_applied"}


def end_rejected(state: CodeReviewState) -> dict:
    print(f"  [Rejected] Review rejected by human.")
    report = (
        f"CODE REVIEW REJECTED\n"
        f"Reason: {state['human_notes'] or 'No reason provided'}\n"
        f"Issues identified: {len(state['issues_found'])}\n"
        f"Action required: Developer must address issues before resubmission."
    )
    return {"final_review": report, "status": "rejected"}


def build_code_review_graph():
    graph = StateGraph(CodeReviewState)

    graph.add_node("analyse", analyse_code)
    graph.add_node("suggest", generate_suggestions)
    graph.add_node("review", human_approval)
    graph.add_node("apply", apply_changes)
    graph.add_node("reject_end", end_rejected)

    graph.add_edge(START, "analyse")
    graph.add_edge("analyse", "suggest")
    graph.add_edge("suggest", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "apply_changes": "apply",
            "end_rejected": "reject_end",
            "generate_suggestions": "suggest",  # loop back
        },
    )
    graph.add_edge("apply", END)
    graph.add_edge("reject_end", END)

    return graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["review"],
    )


SAMPLE_CODE = """
def get_user(id):
    db = connect_database()
    query = "SELECT * FROM users WHERE id = " + id
    result = db.execute(query)
    return result
"""

def run_code_review(human_decision: str):
    """Run full code review with given human decision."""
    print(f"\n{'='*60}")
    print(f"CODE REVIEW — Human will: {human_decision.upper()}")
    print("="*60)

    graph = build_code_review_graph()
    config = {"configurable": {"thread_id": f"review_{human_decision}"}}

    print("\n[Phase 1] Running analysis and suggestions...")
    graph.invoke(
        {
            "code_snippet": SAMPLE_CODE,
            "issues_found": [],
            "suggested_changes": "",
            "human_decision": "",
            "human_notes": "",
            "final_review": "",
            "status": "pending",
        },
        config=config,
    )

    saved = graph.get_state(config)
    print(f"\n[Paused] Suggestions ready:")
    print(f"  Issues: {saved.values.get('issues_found', [])}")

    print(f"\n[Human] Entering decision: {human_decision!r}")
    final = graph.invoke(Command(resume=human_decision), config=config)

    print(f"\n[Final Review]\n{final['final_review']}")
    return final


if __name__ == "__main__":
    # Test all three decision paths
    run_code_review("approve")
    run_code_review("reject: SQL injection risk is too severe")
