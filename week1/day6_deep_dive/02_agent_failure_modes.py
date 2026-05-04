"""
Three agent failure modes every AI engineer must know.
Build, trigger, and fix each one.

Failure 1: Parsing failure — LLM doesn't follow ReAct format
Failure 2: Tool error — tool throws exception, agent gets stuck
Failure 3: Infinite loop — agent keeps calling tools without progress
"""

import os
import math
import datetime
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

def _clean_input(text: str) -> str:
    text = text.split("\n")[0].strip()
    for keyword in ["Observ", "Observation", "Thought", "Action"]:
        if text.endswith(keyword):
            text = text[:-len(keyword)].strip()
    return text


# ── Tools for failure mode demos
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use for arithmetic and numeric calculations.
    Input: a math expression like '2 + 2' or '100 * 0.15'.
    """
    expression = _clean_input(expression)
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(float(result), 4))
    except Exception as e:
        return f"Calculator error for '{expression}': {e}. Check your expression syntax."

@tool
def get_current_date(query: str = "") -> str:
    """Get today's date. Use when current date or year is needed."""
    _ = _clean_input(query)
    return datetime.datetime.now().strftime("%Y-%m-%d, %A, %B %d %Y")

@tool
def flaky_tool(query: str) -> str:
    """
    Search for information about a topic.
    Use when you need facts about a subject.
    Input: the topic to search for.
    """
    query = _clean_input(query)
    # Simulate a tool that sometimes fails
    import random
    if random.random() < 0.6:
        raise ConnectionError(f"Search service temporarily unavailable for query: {query}")
    return f"Result for '{query}': This is simulated search data about {query}."

@tool
def always_vague_tool(query: str) -> str:
    """
    Get information. Use for any question.
    Input: your question.
    """
    query = _clean_input(query)
    # Returns unhelpfully vague results — triggers infinite loop
    return "I found some information but it's unclear. You may want to search again."


TOOLS_BASIC = [calculator, get_current_date]
TOOLS_FLAKY = [calculator, get_current_date, flaky_tool]
TOOLS_VAGUE = [calculator, always_vague_tool]


# ── Callback that counts iterations clearly

class SimpleTracer(BaseCallbackHandler):
    def __init__(self):
        self.iteration = 0
        self.tool_calls = []
        self.errors = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.iteration += 1
        tool_name = serialized.get("name", "unknown")
        cleaned = _clean_input(input_str)
        print(f"  [{self.iteration}] {tool_name}({cleaned!r})", end="")
        self.tool_calls.append(tool_name)

    def on_tool_end(self, output, **kwargs):
        print(f" → {str(output)[:60]!r}")

    def on_tool_error(self, error, **kwargs):
        print(f" → ERROR: {str(error)[:60]}")
        self.errors.append(str(error))

    def on_agent_finish(self, finish, **kwargs):
        pass


def build_executor(tools, max_iterations=5, handle_errors=True):
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("Hugging_face_model"),
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        max_new_tokens=400,
        temperature=0.1,
    )
    llm = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate.from_template("""You are a helpful assistant with tools.

        Tools available:
        {tools}

        Format EXACTLY:
        Question: the question
        Thought: what to do
        Action: tool name from [{tool_names}]
        Action Input: value only, one line
        Observation: (provided)
        Thought: next step
        Final Answer: answer

        Question: {input}
        Thought:{agent_scratchpad}""")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=max_iterations,
        verbose=False,
        handle_parsing_errors=handle_errors,
        return_intermediate_steps=True,
    )


# ── FAILURE MODE 1: handle_parsing_errors
def demo_failure_mode_1():
    """
    Failure: LLM returns text that doesn't match ReAct format.
    
    handle_parsing_errors=True: sends error back to LLM, asks it to try again
    handle_parsing_errors=False: raises exception immediately
    
    Production decision: always True, but log parsing failures.
    High parsing failure rate = prompt needs improvement.
    """
    print("\n" + "="*60)
    print("FAILURE MODE 1: Parsing errors")
    print("="*60)

    print("\nWith handle_parsing_errors=True (production setting):")
    executor = build_executor(TOOLS_BASIC, handle_errors=True)
    tracer = SimpleTracer()
    try:
        result = executor.invoke(
            {"input": "What is 15% of 8000?"},
            config={"callbacks": [tracer]},
        )
        print(f"\nResult: {result['output']}")
        print(f"Tool calls made: {tracer.tool_calls}")
    except Exception as e:
        print(f"Exception: {e}")

    print("\nKEY INSIGHT: handle_parsing_errors feeds the error back")
    print("to the LLM with 'Your response was invalid, try again.'")
    print("This is the same retry-with-error pattern from Day 2.")


# ── FAILURE MODE 2: Tool exceptions — graceful vs crash ──────────────────────

def demo_failure_mode_2():
    """
    Failure: tool raises an exception mid-execution.
    
    Without error handling: agent crashes entirely
    With error handling in tool: agent sees the error, can try differently
    
    Rule: tools should NEVER raise exceptions to the agent.
          They should return error strings.
          Let the agent decide what to do with the error.
    """
    print("\n" + "="*60)
    print("FAILURE MODE 2: Tool exceptions")
    print("="*60)

    executor = build_executor(TOOLS_FLAKY, handle_errors=True)

    print("\nRunning agent with a flaky tool (60% failure rate)...")
    print("Watch: does the agent recover when the tool fails?\n")

    for attempt in range(3):
        print(f"Attempt {attempt + 1}:")
        tracer = SimpleTracer()
        try:
            result = executor.invoke(
                {"input": "Search for information about Python programming"},
                config={"callbacks": [tracer]},
            )
            print(f"Output: {result['output'][:100]}")
        except Exception as e:
            print(f"Agent failed completely: {e}")
        print()

    print("KEY INSIGHT: Your calculator returns error strings, not exceptions.")
    print("That is correct. The agent reads 'Calculator error: ...' and retries.")
    print("A tool that raises an exception gives the agent no information to work with.")


# ── FAILURE MODE 3: Infinite loop — max_iterations as safety valve ───────────

def demo_failure_mode_3():
    """
    Failure: agent keeps calling tools but never reaches Final Answer.
    Happens when: tool returns vague results, agent keeps trying.
    
    max_iterations is your circuit breaker.
    Without it: infinite loop, infinite cost, never returns.
    
    Production rule: always set max_iterations.
    Log when it's hit — means either tool is bad or question is unanswerable.
    """
    print("\n" + "="*60)
    print("FAILURE MODE 3: Infinite loop / max_iterations")
    print("="*60)

    # max_iterations=3 — will hit the limit
    executor = build_executor(TOOLS_VAGUE, max_iterations=3, handle_errors=True)

    print("\nRunning agent that will hit max_iterations...")
    print("The 'always_vague_tool' returns unclear results — agent keeps retrying\n")

    tracer = SimpleTracer()
    result = executor.invoke(
        {"input": "Find specific facts about machine learning"},
        config={"callbacks": [tracer]},
    )

    print(f"\nFinal output: {result['output']}")
    print(f"\nTool calls made: {len(tracer.tool_calls)}")
    print(f"Hit max_iterations: {len(tracer.tool_calls) >= 3}")
    print("\nKEY INSIGHT: When max_iterations is hit, AgentExecutor returns")
    print("whatever partial answer it has. It does NOT raise an exception.")
    print("In production: check if output looks like a real answer or a bailout.")

    print("\n\nPRODUCTION PATTERN — detect max_iterations bailout:")
    bailout_phrases = [
        "i was unable",
        "i could not",
        "agent stopped",
        "max iterations",
        "i don't have enough",
    ]
    output_lower = result['output'].lower()
    is_bailout = any(phrase in output_lower for phrase in bailout_phrases)
    print(f"Output: {result['output'][:100]!r}")
    print(f"Detected as bailout: {is_bailout}")
    print("If bailout detected → log it, return fallback message to user")


if __name__ == "__main__":
    demo_failure_mode_1()
    demo_failure_mode_2()
    demo_failure_mode_3()