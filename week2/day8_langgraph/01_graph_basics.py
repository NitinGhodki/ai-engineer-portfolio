"""
Day 8 — LangGraph basics.
Build three progressively complex graphs.
Understand state flow before adding LLMs.

Graph 1: linear — A → B → C
Graph 2: branching — A → B or C based on state value
Graph 3: joining — two parallel paths merge into one node
"""


from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# GRAPH 1: Linear graph — understand state flow
#
# State flows: START → node_a → node_b → node_c → END
# Each node reads state, modifies it, passes it forward

class LinearState(TypedDict):
    """
    The state dict shared across all nodes in this graph.
    Every node receives this, every node returns a partial update.
    LangGraph merges the update into the full state automatically.
    """
    input: str
    step_a_result: str
    step_b_result: str
    step_c_result: str
    history: list[str]

def node_a(state: LinearState) -> dict:
    """
    First node: receives raw input, does initial processing.
    Returns only the keys it wants to update — not the full state.
    LangGraph merges this partial return into the full state.
    """

    print(f"  [Node A] received input: {state['input']!r}")
    result = f"A processed: {state['input'].upper()}"
    return {
        "step_a_result": result,
        "history": state.get("history", []) + ["Node A completed"],
    }

def node_b(state: LinearState) -> dict:
    """Second node: receives state with step_a_result populated."""

    print(f"  [Node B] received from A: {state['step_a_result']!r}")
    result = f"B enriched: {state['step_a_result']} | length={len(state['step_a_result'])}"
    return {
        "step_b_result": result,
        "history": state.get("history", []) + ["Node B completed"],
    }


def node_c(state: LinearState) -> dict:
    """Third node: final processing."""
    print(f"  [Node C] received from B: {state['step_b_result']!r}")
    result = f"C final: {state['step_b_result']} ✓"
    return {
        "step_c_result": result,
        "history": state.get("history", []) + ["Node C completed"],
    }

def build_linear_graph():
    """
    Build the graph by:
    1. Creating a StateGraph with the state schema
    2. Adding nodes (name → function)
    3. Adding edges (source → destination)
    4. Compiling — validates the graph structure
    """

    graph = StateGraph(LinearState)

    #add nodes
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_node("node_c", node_c)

    # Add edges — defines execution order
    graph.add_edge(START, "node_a")   # START is always the entry point
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", "node_c")
    graph.add_edge("node_c", END)     # END is always the exit point

    return graph.compile()


# GRAPH 2: Conditional routing — the core LangGraph pattern
#
# State flows: START → classifier → router decides → handler_a OR handler_b → END
# This is how real agents decide what to do next

class RoutingState(TypedDict):
    input: str
    input_type: str        # "question", "calculation", or "command"
    handler_result: str
    routing_reason: str

def classifier_node(state: RoutingState) -> dict:
    """
    Classifies the input. In production this would call an LLM.
    For now: simple keyword classification.
    """
    text = state["input"].lower()
    if any(w in text for w in ["what", "who", "why", "how", "explain", "?"]):
        input_type = "question"
        reason = "contains question words"
    elif any(w in text for w in ["+", "-", "*", "/", "calculate", "sum", "multiply", "%"]):
        input_type = "calculation"
        reason = "contains math operators or keywords"
    else:
        input_type = "command"
        reason = "no question words or math operators found"

    print(f"  [Classifier] '{state['input'][:40]}' → type={input_type!r} ({reason})")
    return {"input_type": input_type, "routing_reason": reason}


def route_by_type(state: RoutingState) -> Literal["question_handler", "calc_handler", "command_handler"] :
    """
    Conditional edge function.
    Returns the NAME of the next node to visit.
    LangGraph calls this after classifier_node to decide where to go.

    This is just a Python function that returns a string.
    That string must match a node name in the graph.
    """

    routing_map = {
        "question": "question_handler",
        "calculation": "calc_handler",
        "command": "command_handler",
    }
    next_node = routing_map.get(state["input_type"], "command_handler")
    print(f"  [Router] routing to: {next_node}")
    return next_node

def question_handler(state: RoutingState) -> dict:
    print(f"  [Question Handler] processing question")
    return {"handler_result": f"ANSWER: I'll help you understand: {state['input']}"}


def calc_handler(state: RoutingState) -> dict:
    print(f"  [Calc Handler] processing calculation")
    # Simple eval for demo — in production use your calculator tool
    try:
        expr = state["input"].replace("calculate", "").replace("what is", "").strip()
        result = eval(expr, {"__builtins__": {}})
        return {"handler_result": f"RESULT: {state['input']} = {result}"}
    except Exception:
        return {"handler_result": f"RESULT: Could not evaluate '{state['input']}'"}


def command_handler(state: RoutingState) -> dict:
    print(f"  [Command Handler] processing command")
    return {"handler_result": f"EXECUTED: Command '{state['input']}' acknowledged"}


def build_routing_graph():
    graph = StateGraph(RoutingState)

    graph.add_node("classifier", classifier_node)
    graph.add_node("question_handler", question_handler)
    graph.add_node("calc_handler", calc_handler)
    graph.add_node("command_handler", command_handler)

    graph.add_edge(START, "classifier")

    # Conditional edge: after classifier, call route_by_type() to decide next node
    graph.add_conditional_edges(
        "classifier",
        route_by_type,
        {                       # mapping: return value → node name
            "question_handler": "question_handler",
            "calc_handler": "calc_handler",
            "command_handler": "command_handler",
        },
    )

    # All handlers go to END
    graph.add_edge("question_handler", END)
    graph.add_edge("calc_handler", END)
    graph.add_edge("command_handler", END)

    return graph.compile()

# GRAPH 3: Cycle with validation — graph that loops back on itself
#
# State flows: START → generate → validate → (if failed) → generate (loop)
#                                           → (if passed) → END
# This is how you build retry logic INSIDE the graph


class ValidationState(TypedDict):
    target_length: int      # we want output of exactly this word count
    generated_text: str
    word_count: int
    attempts: int
    passed: bool
    validation_errors: list[str]

def generate_node(state: ValidationState) -> dict:
    """Generates text. Simulates LLM output that may not meet requirements."""
    import random
    attempts = state.get("attempts", 0) + 1
    target = state["target_length"]

    # Simulate: first attempt is always off, later attempts improve
    if attempts == 1:
        words = ["word"] * (target + random.randint(5, 15))  # too long
    elif attempts == 2:
        words = ["word"] * (target - random.randint(2, 8))   # too short
    else:
        words = ["word"] * target                             # correct

    text = " ".join(words)
    print(f"  [Generate] attempt {attempts}: generated {len(text.split())} words (target: {target})")
    return {
        "generated_text": text,
        "word_count": len(text.split()),
        "attempts": attempts,
    }

def validate_node(state: ValidationState) -> dict:
    """Validates the generated text against requirements."""
    errors = []
    target = state["target_length"]
    actual = state["word_count"]
    tolerance = 2  # allow ±2 words

    if abs(actual - target) > tolerance:
        errors.append(f"Word count {actual} is outside target range {target-tolerance}-{target+tolerance}")

    passed = len(errors) == 0
    print(f"  [Validate] {'PASSED' if passed else 'FAILED'}: {errors if errors else 'all checks passed'}")
    return {"passed": passed, "validation_errors": errors}

def should_retry(state: ValidationState) -> Literal["generate", "end"]:
    """Conditional edge: retry if failed and under max attempts, else end."""
    if not state["passed"] and state.get("attempts", 0) < 4:
        print(f"  [Router] validation failed, retrying (attempt {state['attempts']})")
        return "generate"   # loop back
    elif not state["passed"]:
        print(f"  [Router] max attempts reached, ending with failure")
        return "end"
    else:
        print(f"  [Router] validation passed, ending successfully")
        return "end"

def build_validation_graph():
    graph = StateGraph(ValidationState)

    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {"generate": "generate", "end": END},
    )

    return graph.compile()


# RUN ALL THREE GRAPHS

def main():
    print("="*60)
    print("GRAPH 1: Linear execution")
    print("="*60)
    linear = build_linear_graph()
    result = linear.invoke({"input": "hello langgraph", "history": []})
    print(f"\nFinal state:")
    print(f"  step_a: {result['step_a_result']}")
    print(f"  step_b: {result['step_b_result']}")
    print(f"  step_c: {result['step_c_result']}")
    print(f"  history: {result['history']}")

    print("\n" + "="*60)
    print("GRAPH 2: Conditional routing")
    print("="*60)
    router = build_routing_graph()
    test_inputs = [
        "What is machine learning?",
        "calculate 150 * 0.18",
        "save this file",
    ]
    for inp in test_inputs:
        print(f"\nInput: {inp!r}")
        result = router.invoke({"input": inp})
        print(f"Result: {result['handler_result']}")

    print("\n" + "="*60)
    print("GRAPH 3: Validation cycle with retry")
    print("="*60)
    validator = build_validation_graph()
    result = validator.invoke({
        "target_length": 10,
        "generated_text": "",
        "word_count": 0,
        "attempts": 0,
        "passed": False,
        "validation_errors": [],
    })
    print(f"\nFinal result:")
    print(f"  Passed: {result['passed']}")
    print(f"  Attempts taken: {result['attempts']}")
    print(f"  Final word count: {result['word_count']}")
    print(f"  Target: {result['target_length']}")


if __name__ == "__main__":
    main()