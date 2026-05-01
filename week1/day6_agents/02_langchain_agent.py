"""
Day 6 — LangChain agent with 4 tools and execution tracing.

After building the manual agent, LangChain's agent is transparent.
AgentExecutor IS the loop you wrote in 01_manual_agent.py.
"""

import os
import math
import datetime
import json
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

# PART 1: Define tools using @tool decorator
# LangChain reads the docstring as the tool description
# The docstring IS the prompt for the tool — write it clearly

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use for arithmetic, percentages, and numeric calculations.
    Input must be a valid math expression string like '2.5 * 40' or '15 / 100 * 8500'.
    """
    expression = _clean_input(expression)
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(float(result),4))
    except Exception as e:
        return f"Error: {e}"
    

@tool
def get_current_date(query: str = "") -> str:
    """
    Get the current date and day of week.
    Use when the question involves today's date, current year, or day of the week.
    Input can be empty or any string — it is ignored.
    """
    query = _clean_input(query)
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d, %A, %B %d %Y")

@tool
def word_counter(text: str) -> str:
    """
    Count the number of words and characters in a text.
    Use when asked to count words, characters, or text length.
    Input: the text to count.
    """
    text = _clean_input(text)
    return f"{len(text.split())} words, {len(text)} characters"


@tool
def text_transformer(instruction_and_text: str) -> str:
    """
    Transform text according to an instruction.
    Format input as: 'instruction|text'
    Supported instructions: uppercase, lowercase, reverse, title_case
    Example: 'uppercase|hello world' returns 'HELLO WORLD'
    """
    instruction_and_text = _clean_input(instruction_and_text)
    try:
        if "|" not in instruction_and_text:
            return "Error: format must be 'instruction|text'"
        instruction, text = instruction_and_text.split("|", 1)
        instruction = instruction.strip().lower()
        transformers = {
            "uppercase": str.upper,
            "lowercase": str.lower,
            "reverse": lambda x: x[::-1],
            "title_case": str.title,
        }
        if instruction not in transformers:
            return f"Unknown instruction. Use one of: {list(transformers.keys())}"
        return transformers[instruction](text.strip())
    except Exception as e:
        return f"Error: {e}"


TOOLS = [calculator, get_current_date, word_counter, text_transformer]

def _clean_input(text: str) -> str:
    """
    Strip ReAct format bleed from tool inputs.
    Mistral sometimes includes '\nObservation:' as part of the input.
    """
    # Cut off at newline — everything after is ReAct format, not actual input
    text = text.split("\n")[0].strip()
    # Remove common ReAct keywords that bleed in
    for keyword in ["Observ", "Observation", "Thought", "Action"]:
        if text.endswith(keyword):
            text = text[:-len(keyword)].strip()
    return text


# PART 2: Custom callback handler for tracing
# This logs every step the agent takes — tool calls, LLM thoughts, results
# This is what LangSmith does professionally — you're building it manually


class AgentTrace(BaseCallbackHandler):
    """
    Hooks into LangChain's execution to log every agent step.
    Called automatically by AgentExecutor at each step.
    """

    def __init__(self):
        self.steps = []
        self.step_count = 0


    def on_tool_start(self, serialized, input_str, **kwargs):
        self.step_count += 1
        tool_name = serialized.get("name", "unknown")
        # Show if input needed cleaning
        cleaned = input_str.split("\n")[0].strip()
        was_dirty = cleaned != input_str.strip()
        status = " [input cleaned]" if was_dirty else ""
        print(f"\n  [Step {self.step_count}] Tool: {tool_name!r}{status}")
        print(f"  Raw input:     {input_str!r}")
        print(f"  Cleaned input: {cleaned!r}")
        self.steps.append({
            "step": self.step_count,
            "tool": tool_name,
            "input": cleaned,
        })

    def on_tool_end(self, output, **kwargs):
        print(f"    Result: {str(output)[:100]!r}")
        if self.steps:
            self.steps[-1]["result"] = str(output)
    
    def on_tool_error(self, error, **kwargs):
        print(f"  Tool error: {error}")

    def on_agent_finish(self, finish, **kwargs):
        print(f"\n  [DONE] Final answer reached after {self.step_count} tool calls")

    def print_summary(self):
        if not self.steps:
            print("  No tools were called (LLM answered directly)")
            return
        print(f"\n  Execution summary ({len(self.steps)} tool calls):")
        for s in self.steps:
            print(f"    Step {s['step']}: {s['tool']}({s['input']!r}) → {s.get('result', 'N/A')!r}")


# PART 3: Build the agent
def build_agent():
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("Hugging_face_model"),
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        max_new_tokens=512,
        temperature=0.1,
    )
    llm = ChatHuggingFace(llm=llm)

    # ReAct prompt template — LangChain requires specific placeholder names
    # {tools}, {tool_names}, {input}, {agent_scratchpad} are required
    prompt = PromptTemplate.from_template("""You are a helpful assistant with access to tools.

        Available tools:
        {tools}

        Use this EXACT format. Do not add anything after Action Input:

        Question: the input question you must answer
        Thought: reason about what to do next
        Action: the tool to use, must be one of [{tool_names}]
        Action Input: the exact input to the tool, nothing else, stop here
        Observation: the result of the tool (this will be filled in for you)
        Thought: reason about the observation
        Final Answer: your final answer to the original question

        Important rules:
        - Action Input must be on ONE line only
        - Do not write Observation yourself — it will be provided
        - Stop writing immediately after Action Input value

        Begin:

        Question: {input}
        Thought:{agent_scratchpad}""")
    
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        max_iterations=6,
        verbose=False,      # we use our own tracer
        handle_parsing_error=True,
    )

    return executor

# PART 4: Run with full tracing

def run_with_trace(executor, question: str):
    tracer = AgentTrace()
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print("="*60)

    try:
        result = executor.invoke(
            {"input": question},
            config={"callbacks": [tracer]},
        )
        tracer.print_summary()
        print(f"\nFinal Answer: {result.get('output', 'No output')}")
        return result
    except Exception as e:
        print(f"Agent error: {e}")
        return None
    
def main():
    print("Building LangChain agent...")
    executor = build_agent()

    questions = [
        "What is 23% of 15000?",
        "What day is today and what year is it?",
        "If someone was born in 1998 and today is the current date, how old are they?",
        "How many words are in the sentence: 'The quick brown fox jumps over the lazy dog'?",
        "What is the capital of Japan?",  # no tool needed
    ]

    for question in questions:
        run_with_trace(executor, question)

if __name__ == "__main__":
    main()