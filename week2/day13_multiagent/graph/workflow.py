"""
LangGraph workflow — connects all four agents.

State flows:
START → supervisor → researcher → supervisor → writer → supervisor → critic → supervisor → END

The supervisor node runs after EVERY agent call.
It reads the updated state and decides who goes next.
This is the supervisor pattern in multi-agent systems.
"""

import operator
from typing import TypedDict, Optional, Annotated, List
from langgraph.graph import StateGraph, START, END
from agents.researcher import build_researcher
from agents.writer import build_writer
from agents.critic import build_critic
from agents.supervisor import supervisor_route, build_supervisor_llm


# ── Shared state 

class WorkflowState(TypedDict):
    """
    State shared across all agents.
    Each agent reads what it needs and writes its output.
    The supervisor reads everything to make routing decisions.
    """
    user_request: str           # original user query
    output_format: str          # "paragraph", "bullets", "table"
    research_findings: str      # output from Researcher
    written_draft: str          # output from Writer
    critique_result: str        # output from Critic
    revision_count: int         # how many times Writer has revised
    next_agent: str
    final_output: str           # final answer to return to user
    execution_log: Annotated[List[str], operator.add]   # trace of which agents ran and when
    

# ── Node functions 

def researcher_node(state: WorkflowState) -> dict:
    """Run the Researcher agent."""
    print(f"\n{'─'*50}")
    print(f"[RESEARCHER] Starting research...")

    researcher = build_researcher()
    research_task = (
        f"Research this topic thoroughly: {state['user_request']}\n"
        f"Search for all relevant facts, numbers, and comparisons."
    )

    result = researcher.invoke({"input": research_task})
    findings = result.get("output", "No findings returned.")

    print(f"[RESEARCHER] Complete. Findings: {findings[:100]}...")

    return {
        "research_findings": findings,
        "execution_log": [f"Researcher completed: {len(findings)} chars"],
    }


def writer_node(state: WorkflowState) -> dict:
    """Run the Writer agent."""
    print(f"\n{'─'*50}")
    revision = state.get("revision_count", 0)
    print(f"[WRITER] {'Writing draft' if revision == 0 else f'Revising draft (revision {revision})'}...")

    writer = build_writer()

    # Include critique issues if this is a revision
    critique = state.get("critique_result", "")
    revision_instruction = ""
    if critique and "NEEDS_REVISION" in critique.upper():
        revision_instruction = f"\n\nIMPORTANT: Previous draft had these issues:\n{critique}\nFix all issues."

    writing_task = (
        f"Format the following research findings as {state['output_format']}.\n\n"
        f"Research findings:\n{state['research_findings']}"
        f"{revision_instruction}"
    )

    result = writer.invoke({"input": writing_task})
    draft = result.get("output", "No draft produced.")

    print(f"[WRITER] Complete. Draft: {draft[:100]}...")

    return {
        "written_draft": draft,
        "critique_result": "",  # clear previous critique when new draft exists
        "revision_count": revision + 1,
        "execution_log": [f"Writer completed revision {revision + 1}"],
    }


def critic_node(state: WorkflowState) -> dict:
    """Run the Critic agent."""
    print(f"\n{'─'*50}")
    print(f"[CRITIC] Reviewing draft for accuracy...")

    critic = build_critic()

    review_task = (
        f"Review this output for factual accuracy:\n\n"
        f"ORIGINAL REQUEST: {state['user_request']}\n\n"
        f"RESEARCH FINDINGS (ground truth):\n{state['research_findings']}\n\n"
        f"WRITTEN OUTPUT (to review):\n{state['written_draft']}\n\n"
        f"Check: do all facts in the output match the research findings? "
        f"Are all numbers correct? Is anything missing or wrong?"
    )

    result = critic.invoke({"input": review_task})
    critique = result.get("output", "APPROVED")

    approved = "APPROVED" in critique.upper()
    print(f"[CRITIC] Verdict: {'✓ APPROVED' if approved else '✗ NEEDS_REVISION'}")
    if not approved:
        print(f"[CRITIC] Issues: {critique[:150]}...")

    return {
        "critique_result": critique,
        "final_output": state["written_draft"] if approved else "",
        "execution_log": [f"Critic: {'APPROVED' if approved else 'NEEDS_REVISION'}"],
    }


def supervisor_node(state: WorkflowState) -> dict:
    """Supervisor decides next step. Updates state with routing decision."""
    llm = build_supervisor_llm()
    next_agent = supervisor_route(state, llm)
    return {
        "next_agent": next_agent,
        "execution_log": [f"Supervisor → {next_agent}"],
    }


def route_from_supervisor(state: WorkflowState) -> str:
    """Conditional edge: read supervisor's routing decision from state."""
    return state.get("next_agent", "end")


def finalise_node(state: WorkflowState) -> dict:
    """Produce final output when workflow ends."""
    final = state.get("final_output") or state.get("written_draft", "No output produced.")
    print(f"\n{'='*50}")
    print(f"[FINAL] Workflow complete.")
    print(f"[FINAL] Agents used: {state.get('execution_log', [])}")
    return {
        "final_output": final,
        "execution_log": ["Workflow finalised"],
    }


# ── Build graph 

def build_workflow() -> StateGraph:
    """
    Graph structure:
    START → supervisor → researcher ↘
                      ↗              supervisor → writer → supervisor → critic → supervisor → END
    """
    graph = StateGraph(WorkflowState)

    # Add all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalise", finalise_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor routes to any agent or end
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            "end": "finalise",
        },
    )

    # Every agent returns to supervisor after completing
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("finalise", END)

    return graph.compile()