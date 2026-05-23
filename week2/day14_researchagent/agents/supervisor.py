
VALID_ROUTES = {"researcher", "writer", "critic", "end"}


def supervisor_route(state: dict) -> str:
    """
    Deterministic routing — no LLM needed for simple state machine.
    Falls back to 'end' for any unexpected state.
    """
    research = state.get("research_findings", "")
    draft = state.get("written_draft", "")
    critique = state.get("critique_result", "")
    revisions = state.get("revision_count", 0)

    if revisions >= 2:
        print("  [Supervisor] Max revisions → end")
        return "end"

    if not research:
        print("  [Supervisor] No research → researcher")
        return "researcher"

    if not draft:
        print("  [Supervisor] No draft → writer")
        return "writer"

    if not critique:
        print("  [Supervisor] No critique → critic")
        return "critic"

    if "APPROVED" in critique.upper():
        print("  [Supervisor] Approved → end")
        return "end"

    if "NEEDS_REVISION" in critique.upper():
        print(f"  [Supervisor] Needs revision → writer (rev {revisions + 1})")
        return "writer"

    print("  [Supervisor] Unknown state → end")
    return "end"