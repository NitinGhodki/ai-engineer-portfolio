"""
Day 6 — Manual agent loop. Zero LangChain.

Build the ReAct loop to understand exactly
what LangChain's AgentExecutor does under the hood.

The loop:
1. LLM receives question + available tools
2. LLM returns either: tool_call OR final_answer
3. If tool_call: execute the tool, feed result back
4. If final_answer: stop the loop
5. Max iterations guard — prevents infinite loops
"""


import os
import json
import math
import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from dataclasses import dataclass, field
import ollama

load_dotenv()

client = InferenceClient(token=os.getenv("HF_API_KEY"))
MODEL = os.getenv("Hugging_face_model")

# STEP 1: Define tools
# A tool = a Python function + a description the LLM reads
# The LLM never calls the function directly — it tells YOU to call it

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict

def tool_calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Example input: "2.5 * 40 + 15"
    """
    try:
        # Safe eval — only math operations allowed
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("_")
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{result}"
    except Exception as e:
        return f"Calculator error: {e}"


def tool_get_current_date(_: str = "") -> str:
    """Return today's date."""
    return datetime.datetime.now().strftime("%Y-%m-%d (%A, %B %d %Y)")


def tool_word_counter(text: str) -> str:
    """Count words and characters in text."""
    words = len(text.split())
    chars = len(text)
    return f"{words} words, {chars} characters"


def tool_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


# Tool registry — maps tool name to (Tool definition, function)
TOOLS = {
    "calculator": (
        Tool(
            name="calculator",
            description="Evaluate mathematical expressions. Use for any arithmetic, percentage, or numeric calculation.",
            parameters={"expression": "string — a valid math expression like '2.5 * 40' or 'sqrt(144)'"},
        ),
        tool_calculator,
    ),
    "get_current_date": (
        Tool(
            name="get_current_date",
            description="Get today's date. Use when the question involves today's date or current time.",
            parameters={},
        ),
        tool_get_current_date,
    ),
    "word_counter": (
        Tool(
            name="word_counter",
            description="Count words and characters in a piece of text.",
            parameters={"text": "string — the text to count"},
        ),
        tool_word_counter,
    ),
    "uppercase": (
        Tool(
            name="uppercase",
            description="Convert text to uppercase letters.",
            parameters={"text": "string — the text to convert"},
        ),
        tool_uppercase,
    ),
}


# STEP 2: Build the prompt that describes tools to the LLM
# The LLM learns what tools exist and how to call them from this prompt

def build_system_prompt() -> str:
    """
    Tell the LLM about available tools and the exact response format.
    This is the most important part of agent design.
    If this prompt is unclear, the agent breaks.
    """
    tool_descriptions = []
    for tool_def, _ in TOOLS.values():
        params = json.dumps(tool_def.parameters)
        tool_descriptions.append(
            f"- {tool_def.name}: {tool_def.description}\n"
            f"  Parameters: {params}"
        )
    tools_text = "\n".join(tool_descriptions)

    return f"""You are a helpful assistant with access to tools.

        Available tools:
        {tools_text}

        To use a tool, respond with EXACTLY this JSON format (nothing else):
        {{"action": "tool_name", "input": "tool_input_value"}}

        When you have the final answer and no more tools are needed, respond with:
        {{"action": "final_answer", "input": "your complete answer here"}}

        Rules:
        - Use tools when they help answer the question more accurately
        - After getting a tool result, decide if you need another tool or have enough to answer
        - Always end with a final_answer action
        - Respond with ONLY the JSON object, no extra text"""
 

# STEP 3: Parse LLM response into action
# LLM returns JSON string → parse it → decide what to do

@dataclass
class AgentAction:
    action: str
    input: str

def parse_llm_response(response: str) -> AgentAction:
    """
    Parse the LLM's JSON response into an AgentAction.
    Handles common formatting mistakes LLMs make.
    """

    # Clean markdown formatting if present
    cleaned = response.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            if "{"in part:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                break
    
    # Extract JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end <= start:
        # LLM didn't follow format — treat as final answer
        return AgentAction(action="final_answer", input = cleaned)
    
    try:
        data = json.loads(cleaned[start:end])
        return AgentAction(
            action = data.get("action", "final_answer"),
            input=str(data.get("input", "")),
        )
    except json.JSONDecodeError as e:
        print(f"[ERROR]: {e}")
        return AgentAction(action="final_answer", input=cleaned)
    

# STEP 4: The agent execution trace
# Records every step for debugging and observability

@dataclass
class AgentStep:
    step_number: int
    llm_response: str
    action: AgentAction
    tool_result: str | None = None

@dataclass
class AgentTrace:
    question: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_steps: int = 0

    def print_trace(self):
        print(f"\n{'='*60}")
        print(f"AGENT TRACE")
        print(f"Question: {self.question}")
        print(f"{'='*60}")
        for step in self.steps:
            print(f"\n[Step {step.step_number}]")
            print(f"  LLM decided: action={step.action.action!r}, input={step.action.input!r}")
            if step.tool_result is not None:
                print(f"  Tool result: {step.tool_result!r}")
        print(f"\n[FINAL ANSWER] {self.final_answer}")
        print(f"Total steps: {self.total_steps}")


# STEP 5: The agent loop
# This is the core. Run it, read every line, understand each decision.

def run_agent(question: str, max_iterations: int = 6) -> AgentTrace:
    """
    The ReAct agent loop.

    Conversation history grows with each iteration:
    - system prompt (always first)
    - user question (always second)
    - LLM response (added each iteration)
    - tool result (added when tool is called)

    The LLM sees the full history every iteration —
    this is how it "remembers" what tools it already called.
    """

    trace = AgentTrace(question=question)

    # Build conversation history

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": question},
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"    [Iteration {iteration}] calling LLM.....")

        #Call LLM
        response = client.chat_completion(
            messages=messages,
            model=MODEL,
            max_tokens=300,
        )
        llm_response = response.choices[0].message.content

        # response = ollama.chat(
        #     model="llama3.2",
        #     messages=messages,
        #     options={"temperature": 0.3}
        # )
        # llm_response = response["message"]["content"].strip()
        llm_response = llm_response.strip()

        action = parse_llm_response(llm_response)

        # Record step
        step = AgentStep(
            step_number=iteration,
            llm_response=llm_response,
            action=action,
        )

        if action.action == "final_answer":
            trace.final_answer = action.input
            trace.total_steps = iteration
            trace.steps.append(step)
            break

        # Execute tool
        if action.action in TOOLS:
            _, tool_fn = TOOLS[action.action]
            tool_result = tool_fn(action.input)
            step.tool_result = tool_result

            # Add to conversation history so LLM sees the result
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role": "user",
                "content": f"Tool result for {action.action}: {tool_result} \n Continue."
            })

        else: 
            # Unknown tool - tell LLM
            step.tool_result = f"Error: tool '{action.action}' not found"
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role": "user",
                "content": f"Error: tool '{action.action}' does not exist. Use only available tools if needed."
            })

        trace.steps.append(step)

        # Safety guard
        if iteration == max_iterations:
            trace.final_answer = "Max iterations reached without final answer."
            trace.total_steps = iteration

    return trace

# STEP 6: Test with real questions
def main():
    test_questions = [
        # Requires calculator
        "What is 15% of 8500?",

        # Requires date tool
        "What day of the week is today?",

        # Requires multiple tools: calculator + date
        "Today is my birthday. If I was born in 1995, how old am I today?",

        # No tool needed — LLM should answer directly
        "What is the capital of France?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Running agent for: {question!r}")
        trace = run_agent(question)
        trace.print_trace()


if __name__ == "__main__":
    main()