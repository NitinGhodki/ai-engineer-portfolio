"""
Day 9 — Checkpointing in depth.

Three things you learn here:
1. How to save and inspect graph state at any point
2. How to resume a graph after it "crashed"
3. How to travel back to a previous checkpoint (time travel)

MemorySaver = in-memory (development only, dies on restart)
SqliteSaver = SQLite file (survives restarts, use for local production)
"""

import os
import time
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# PART 1: Understanding checkpoints
# Every node execution creates a checkpoint automatically

class ResearchState(TypedDict):
    topic: str
    sources: list[str]
    summaries: list[str]
    draft: str
    final: str
    step: int


def step_1_gather_sources(state: ResearchState) -> dict:
    print(f"  [Step 1] Gathering sources for: {state['topic']!r}")
    time.sleep(0.1)  # simulate work
    sources = [
        f"Source A: Introduction to {state['topic']}",
        f"Source B: Advanced {state['topic']} techniques",
        f"Source C: {state['topic']} case studies",
    ]
    print(f"  [Step 1] Found {len(sources)} sources ✓")
    return {"sources": sources, "step": 1}


def step_2_summarise(state: ResearchState) -> dict:
    print(f"  [Step 2] Summarising {len(state['sources'])} sources...")
    time.sleep(0.1)
    summaries = [f"Summary of: {s[:40]}" for s in state["sources"]]
    print(f"  [Step 2] Summaries created ✓")
    return {"summaries": summaries, "step": 2}


def step_3_draft(state: ResearchState) -> dict:
    print(f"  [Step 3] Writing draft...")
    time.sleep(0.1)
    draft = f"DRAFT on {state['topic']}:\n" + "\n".join(state["summaries"])
    print(f"  [Step 3] Draft written ✓")
    return {"draft": draft, "step": 3}


def step_4_finalise(state: ResearchState) -> dict:
    print(f"  [Step 4] Finalising report...")
    time.sleep(0.1)
    final = f"FINAL REPORT\n{'='*40}\n{state['draft']}\n\nReport complete."
    print(f"  [Step 4] Report finalised ✓")
    return {"final": final, "step": 4}


def build_research_graph(checkpointer):
    graph = StateGraph(ResearchState)
    graph.add_node("gather", step_1_gather_sources)
    graph.add_node("summarise", step_2_summarise)
    graph.add_node("draft", step_3_draft)
    graph.add_node("finalise", step_4_finalise)

    graph.add_edge(START, "gather")
    graph.add_edge("gather", "summarise")
    graph.add_edge("summarise", "draft")
    graph.add_edge("draft", "finalise")
    graph.add_edge("finalise", END)

    return graph.compile(checkpointer=checkpointer)


def demo_checkpoint_inspection():
    """
    Run the graph and inspect every checkpoint created.
    Shows: what is saved, when, and how to read it.
    """
    print("\n" + "="*60)
    print("PART 1: Checkpoint inspection")
    print("="*60)

    checkpointer = MemorySaver()
    graph = build_research_graph(checkpointer)
    config = {"configurable": {"thread_id": "research_001"}}

    print("\nRunning full graph...")
    result = graph.invoke(
        {
            "topic": "LangGraph",
            "sources": [],
            "summaries": [],
            "draft": "",
            "final": "",
            "step": 0,
        },
        config=config,
    )

    print(f"\nGraph complete. Final step: {result['step']}")

    # Get all checkpoints for this thread
    print("\n--- All checkpoints saved ---")
    checkpoints = list(graph.get_state_history(config))
    print(f"Total checkpoints: {len(checkpoints)}")

    for i, checkpoint in enumerate(reversed(checkpoints)):
        step_val = checkpoint.values.get("step", "?")
        next_nodes = checkpoint.next
        created = checkpoint.metadata.get("step", i)
        print(f"\nCheckpoint {i + 1}:")
        print(f"  Step value in state: {step_val}")
        print(f"  Next nodes: {next_nodes if next_nodes else '(END)'}")
        print(f"  Checkpoint ID: {checkpoint.config['configurable'].get('checkpoint_id', 'N/A')[:16]}...")


def demo_crash_and_resume():
    """
    Simulate a graph "crashing" midway.
    Show how checkpointing allows resuming from the crash point.

    In real production: the crash would be a server restart, OOM error, etc.
    We simulate by only running part of the graph using interrupt_before.
    """
    print("\n" + "="*60)
    print("PART 2: Crash simulation and resume")
    print("="*60)

    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "crash_demo_001"}}

    # Build graph that stops at "draft" node (simulating crash after step 2)
    graph_partial = StateGraph(ResearchState)

    graph_partial.add_node("gather", step_1_gather_sources)
    graph_partial.add_node("summarise", step_2_summarise)
    graph_partial.add_node("draft", step_3_draft)
    graph_partial.add_node("finalise", step_4_finalise)

    graph_partial.add_edge(START, "gather")
    graph_partial.add_edge("gather", "summarise")
    graph_partial.add_edge("summarise", "draft")
    graph_partial.add_edge("draft", "finalise")
    graph_partial.add_edge("finalise", END)

    crashed_graph = graph_partial.compile(
        checkpointer=checkpointer,
        interrupt_before=["draft"],  # "crash" before step 3
    )

    print("\n[PHASE 1] Running until simulated crash point (before draft node)...")
    crashed_graph.invoke(
        {
            "topic": "Checkpointing",
            "sources": [],
            "summaries": [],
            "draft": "",
            "final": "",
            "step": 0,
        },
        config=config,
    )

    # Check what we have
    saved_state = crashed_graph.get_state(config)
    print(f"\n[CRASH] Graph stopped at: {saved_state.next}")
    print(f"[CRASH] Work preserved — step: {saved_state.values.get('step')}")
    print(f"[CRASH] Sources saved: {len(saved_state.values.get('sources', []))} ✓")
    print(f"[CRASH] Summaries saved: {len(saved_state.values.get('summaries', []))} ✓")
    print(f"\n[CRASH] In production: server restarts here.")
    print(f"[CRASH] Without checkpointing: start from step 0 again.")
    print(f"[CRASH] With checkpointing: resume from step 2.")

    # Build a full graph (server "restarted") with same checkpointer
    # Same checkpointer = same saved state
    full_graph = graph_partial.compile(checkpointer=checkpointer)

    print("\n[RESUME] Server restarted. Resuming from checkpoint...")
    from langgraph.types import Command
    final = full_graph.invoke(Command(resume=""), config=config)

    print(f"\n[RESUME] Complete! Final step: {final['step']}")
    print(f"[RESUME] Final report preview: {final['final'][:100]}...")


def demo_time_travel():
    """
    LangGraph allows you to go back to any previous checkpoint.
    Useful for: debugging, replaying with different inputs, A/B testing.
    """
    print("\n" + "="*60)
    print("PART 3: Time travel — replay from any checkpoint")
    print("="*60)

    checkpointer = MemorySaver()
    graph = build_research_graph(checkpointer)
    config = {"configurable": {"thread_id": "timetravel_001"}}

    # Run the full graph
    print("\nRunning full graph first...")
    graph.invoke(
        {
            "topic": "Time Travel",
            "sources": [],
            "summaries": [],
            "draft": "",
            "final": "",
            "step": 0,
        },
        config=config,
    )

    # Get all checkpoints
    all_checkpoints = list(graph.get_state_history(config))
    print(f"\nTotal checkpoints available: {len(all_checkpoints)}")

    # Go back to checkpoint after step 1 (after "gather" node)
    # Checkpoints are ordered newest-first — so last index is earliest
    if len(all_checkpoints) >= 3:
        early_checkpoint = all_checkpoints[-4]  # after gather node
        early_config = early_checkpoint.config

        print(f"\nTravelling back to checkpoint at step: {early_checkpoint.values.get('step')}")
        print(f"State at that point:")
        print(f"  sources: {len(early_checkpoint.values.get('sources', []))} items")
        print(f"  summaries: {len(early_checkpoint.values.get('summaries', []))} items (empty at this point)")

        # Update state at that checkpoint and replay from there
        graph.update_state(
            early_config,
            {"topic": "Time Travel (MODIFIED TOPIC)"},  # change the topic
        )

        print("\nReplaying from modified checkpoint with new topic...")
        replay_result = graph.invoke(None, config=early_config)
        print(f"New final report preview:\n{replay_result['final'][:150]}...")


if __name__ == "__main__":
    demo_checkpoint_inspection()
    demo_crash_and_resume()
    demo_time_travel()