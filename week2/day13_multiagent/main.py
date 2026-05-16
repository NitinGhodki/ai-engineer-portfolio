"""
Day 13 — Multi-agent system entrypoint.
Run different types of requests to see all four agents collaborate.
"""

from graph.workflow import build_workflow, WorkflowState


def run_request(user_request: str, output_format: str = "paragraph"):
    """Run a request through the full 4-agent pipeline."""
    print(f"\n{'='*60}")
    print(f"REQUEST: {user_request}")
    print(f"FORMAT:  {output_format}")
    print("="*60)

    workflow = build_workflow()

    initial_state: WorkflowState = {
        "user_request": user_request,
        "output_format": output_format,
        "research_findings": "",
        "written_draft": "",
        "critique_result": "",
        "revision_count": 0,
        "next_agent": "",
        "final_output": "",
        "execution_log": [],
    }

    result = workflow.invoke(initial_state)

    print(f"\n{'='*60}")
    print("FINAL OUTPUT:")
    print("="*60)
    print(result["final_output"])

    print(f"\nEXECUTION LOG:")
    for i, step in enumerate(result["execution_log"], 1):
        print(f"  {i}. {step}")

    return result


def main():
    # Request 1: Simple factual — Researcher + Writer + Critic
    # run_request(
    #     user_request="What are the pricing plans and what do they include?",
    #     output_format="bullets",
    # )

    # Request 2: Comparison — tests competitor search + calculator + table format
    run_request(
        user_request="Compare our Professional plan with CompetitorB's pricing. Include price difference percentage.",
        output_format="table",
    )

    # Request 3: Multi-step — needs calculation + research + structured output
    # run_request(
    #     user_request="If a customer pays annually for the Professional plan with the 20 percent discount, what is their monthly effective cost?",
    #     output_format="paragraph",
    # )


if __name__ == "__main__":
    main()