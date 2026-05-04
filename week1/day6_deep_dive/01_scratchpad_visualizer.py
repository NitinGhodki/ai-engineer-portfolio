"""
Visualize agent_scratchpad at every step.
This makes AgentExecutor completely transparent.

The scratchpad is what separates iteration 1 from iteration 3 —
it is the agent's memory of what it already tried.
"""

import os
import math
import datetime
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


def _clean_input(text: str) -> str:
    text = text.split("\n")[0].strip()
    for keyword in ["Observ", "Observation", "Thought", "Action"]:
        if text.endswith(keyword):
            text = text[:-len(keyword)].strip()
    return text


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use for arithmetic, percentages, and numeric calculations.
    Input must be a valid math expression like '2.5 * 40'.
    """
    expression = _clean_input(expression)
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(float(result), 4))
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_current_date(query: str = "") -> str:
    """
    Get today's date and day of week.
    Use when the question involves current date or year.
    """
    _ = _clean_input(query)
    return datetime.datetime.now().strftime("%Y-%m-%d, %A, %B %d %Y")


TOOLS = [calculator, get_current_date]


class ScratchpadVisualizer(BaseCallbackHandler):
    """
    Shows the agent_scratchpad at every LLM call.
    This is the single most useful thing for understanding agent behaviour.
    """

    def __init__(self):
        self.iteration = 0
        self.tool_calls = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called every time the LLM is about to be invoked."""
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"LLM CALL #{self.iteration}")
        print("="*60)

        # Extract and display the full prompt being sent to LLM
        if prompts:
            full_prompt = prompts[0]
            # Find and display only the scratchpad portion
            if "Thought:" in full_prompt:
                # Get everything after the first "Thought:"
                scratchpad_start = full_prompt.rfind("Question:")
                if scratchpad_start != -1:
                    relevant = full_prompt[scratchpad_start:]
                    print(f"Prompt sent to LLM (relevant portion):\n{relevant}")
            else:
                print(f"Prompt length: {len(full_prompt)} chars")
                print(f"Prompt preview:\n{full_prompt[-500:]}")

    def on_llm_end(self, response, **kwargs):
        """Called after LLM returns a response."""
        if response.generations:
            llm_output = response.generations[0][0].text
            print(f"\nLLM raw response:\n{llm_output}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        cleaned = _clean_input(input_str)
        print(f"\n→ Tool called: {tool_name}")
        print(f"  Raw input:     {input_str!r}")
        print(f"  Cleaned input: {cleaned!r}")
        self.tool_calls.append({"tool": tool_name, "input": cleaned})

    def on_tool_end(self, output, **kwargs):
        print(f"  Tool result:   {output!r}")
        if self.tool_calls:
            self.tool_calls[-1]["result"] = str(output)

    def on_agent_finish(self, finish, **kwargs):
        print(f"\n{'='*60}")
        print(f"AGENT FINISHED after {self.iteration} LLM calls, {len(self.tool_calls)} tool calls")
        print(f"Final answer: {finish.return_values.get('output')}")

    def print_summary(self):
        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print("="*60)
        print(f"Total LLM calls:  {self.iteration}")
        print(f"Total tool calls: {len(self.tool_calls)}")
        for i, tc in enumerate(self.tool_calls, 1):
            print(f"  {i}. {tc['tool']}({tc['input']!r}) → {tc.get('result', 'N/A')!r}")


def build_agent():
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=512,
            temperature=0.1,
        )
    )

    prompt = PromptTemplate.from_template(""" you are a helpfull assistant with access to tools.
            Available tools:
            {tools}

            Use this EXACT format. Do not add anything after Action Input:

            Question: the input question you must answer
            Thought: reason about what to do next
            Action: the tool to use, must be one of [{tool_names}]
            Action Input: the exact input to the tool, nothing else, stop here
            Observation: the result of the tool (this will be filled in for you)
            Thought: reason about result observation
            Final Answer: your final answer to the original question

            Important rules:
            - Action Input must be on ONE line only
            - Do not write Observation yourself — it will be provided
            - Stop writing immediately after Action Input value

            Question: {input}
            Thought:{agent_scratchpad}""")
    
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        max_iterations=6,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def main():
    executor = build_agent()

    # Use a multi-step question so you see the scratchpad grow
    question = "If someone was born in 1995, how old are they today?"

    print(f"\nQuestion: {question}")
    print("Watch how the scratchpad grows with each LLM call...\n")

    visualizer = ScratchpadVisualizer()
    result = executor.invoke(
        {"input": question},
        config={"callbacks": [visualizer]},
    )

    # Show intermediate steps — this is what return_intermediate_steps=True gives you
    print(f"\n{'='*60}")
    print("INTERMEDIATE STEPS (what return_intermediate_steps captures):")
    print("="*60)
    for i, (agent_action, tool_output) in enumerate(result.get("intermediate_steps", []), 1):
        print(f"\nStep {i}:")
        print(f"  Tool:   {agent_action.tool}")
        print(f"  Input:  {agent_action.tool_input!r}")
        print(f"  Output: {tool_output!r}")
        print(f"  Log:    {agent_action.log[:100]!r}")

    visualizer.print_summary()


if __name__ == "__main__":
    main()