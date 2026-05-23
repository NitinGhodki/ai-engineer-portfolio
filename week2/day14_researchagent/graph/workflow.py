import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from agents.supervisor import supervisor_route


class WorkflowState(TypedDict):
    user_request: str
    output_format: str
    research_findings: str
    written_draft: str
    critique_result: str
    revision_count: int
    next_agent: str
    final_output: str
    session_id: str
    execution_log: Annotated[List[str], operator.add]
    cost_summary: dict


def build_workflow(rag) -> StateGraph:
    """Build graph with RAG instance injected into agents."""

    from agents.researcher import build_researcher
    from agents.writer import build_writer
    from agents.critic import build_critic

    def researcher_node(state: WorkflowState) -> dict:
        print(f"\n[RESEARCHER] Starting...")
        executor, tracer = build_researcher(rag, session_id=state["session_id"])
        result = executor.invoke({
            "input": f"Research: {state['user_request']}. Find all relevant facts and numbers."
        },
        config={"stream_runnable": False}
        )
        findings = result.get("output", "No findings.")
        print(f"[RESEARCHER] Done: {findings[:80]}...")
        return {
            "research_findings": findings,
            "execution_log": [f"Researcher: {len(findings)} chars, cost=${tracer.summary()['total_cost_usd']:.6f}"],
        }

    def writer_node(state: WorkflowState) -> dict:
        rev = state.get("revision_count", 0)
        print(f"\n[WRITER] {'Draft' if rev == 0 else f'Revision {rev}'}...")
        executor, tracer = build_writer(session_id=state["session_id"])

        critique = state.get("critique_result", "")
        revision_note = ""
        if "NEEDS_REVISION" in critique.upper():
            revision_note = f"\n\nFix these issues:\n{critique}"

        task = (
            f"Format as {state['output_format']}:\n\n"
            f"Research:\n{state['research_findings']}"
            f"{revision_note}"
        )
        result = executor.invoke({"input": task}, config={"stream_runnable": False})
        draft = result.get("output", "No draft.")
        print(f"[WRITER] Done: {draft[:80]}...")
        return {
            "written_draft": draft,
            "critique_result": "",
            "revision_count": rev + 1,
            "execution_log": [f"Writer rev{rev + 1}: cost=${tracer.summary()['total_cost_usd']:.6f}"],
        }

    def critic_node(state: WorkflowState) -> dict:
        print(f"\n[CRITIC] Reviewing...")
        executor, tracer = build_critic(rag, session_id=state["session_id"])
        task = (
            f"Review for accuracy:\n"
            f"REQUEST: {state['user_request']}\n"
            f"RESEARCH: {state['research_findings']}\n"
            f"DRAFT: {state['written_draft']}"
        )
        result = executor.invoke({"input": task}, config={"stream_runnable": False})
        critique = result.get("output", "APPROVED")
        approved = "APPROVED" in critique.upper()
        print(f"[CRITIC] {'✓ APPROVED' if approved else '✗ NEEDS_REVISION'}")
        return {
            "critique_result": critique,
            "final_output": state["written_draft"] if approved else "",
            "execution_log": [f"Critic: {'APPROVED' if approved else 'NEEDS_REVISION'}, cost=${tracer.summary()['total_cost_usd']:.6f}"],
        }

    def supervisor_node(state: WorkflowState) -> dict:
        next_ag = supervisor_route(state)
        return {
            "next_agent": next_ag,
            "execution_log": [f"Supervisor → {next_ag}"],
        }

    def route_from_supervisor(state: WorkflowState) -> str:
        return state.get("next_agent", "end")

    def finalise_node(state: WorkflowState) -> dict:
        final = state.get("final_output") or state.get("written_draft", "No output produced.")
        print(f"\n[FINAL] Complete. Steps: {len(state.get('execution_log', []))}")
        return {
            "final_output": final,
            "execution_log": ["Workflow complete"],
        }

    graph = StateGraph(WorkflowState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalise", finalise_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor", 
        route_from_supervisor,
        {
            "researcher": "researcher", 
            "writer": "writer",
            "critic": "critic", 
            "end": "finalise"
        },
    )

    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("finalise", END)

    return graph.compile()