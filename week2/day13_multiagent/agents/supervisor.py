"""
Supervisor Agent — orchestrates the other three agents.

Single responsibility: given a user request,
decide which agent to call next and in what order.
The Supervisor does NOT do any research or writing itself.
It reads the current state and routes to the right specialist.

Routing logic:
- New request → Researcher first
- Research done, no draft yet → Writer
- Draft exists → Critic
- Critic says APPROVED → END
- Critic says NEEDS_REVISION → Writer (with issues listed)
- Too many revisions → END (with best available output)
"""



import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def build_supervisor_llm():
    return ChatHuggingFace (
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY_32"),
            max_new_tokens=50,
            temperature=0.0,   # critic needs strict accuracy
        )
    )

def supervisor_route(state: dict, llm) -> str:
    """
    Decide which agent runs next based on current state.
    Returns: "researcher", "writer", "critic", or "end"

    This is the brain of the multi-agent system.
    In production this would be an LLM call.
    Here we use deterministic logic for reliability.
    """
    research = state.get("research_findings", "")
    draft = state.get("written_draft", "")
    critique = state.get("critique_result", "")
    revision_count = state.get("revision_count", 0)

    # Safety valve — prevent infinite revision loops
    if revision_count >= 2:
        print(f"  [Supervisor] Max revisions reached → END")
        return "end"

    # No research yet → start with researcher
    if not research:
        print(f"  [Supervisor] No research found → routing to Researcher")
        return "researcher"

    # Research done but no draft → write it
    if research and not draft:
        print(f"  [Supervisor] Research complete, no draft → routing to Writer")
        return "writer"

    # Draft exists but not yet critiqued → review it
    if draft and not critique:
        print(f"  [Supervisor] Draft ready, no critique → routing to Critic")
        return "critic"

    # Critique exists — check verdict
    if critique:
        if "APPROVED" in critique.upper():
            print(f"  [Supervisor] Critique: APPROVED → END")
            return "end"
        elif "NEEDS_REVISION" in critique.upper():
            print(f"  [Supervisor] Critique: NEEDS_REVISION → routing to Writer (revision {revision_count + 1})")
            return "writer"
        else:
            print(f"  [Supervisor] Critique verdict unclear → END")
            return "end"

    print(f"  [Supervisor] Unexpected state → END")
    return "end"