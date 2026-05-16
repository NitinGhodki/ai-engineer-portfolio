"""
Writer Agent — structures and formats research into readable output.

Single responsibility: given raw research findings,
produce a well-structured, readable response in the requested format.
Tools: format_as_table (formatting only — no search tools)
The Writer does NOT search for new information.
If it needs more facts, it says so and the Supervisor routes back to Researcher.
"""

import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools.search_tools import format_as_table

load_dotenv()

WRITER_TOOLS = [format_as_table]

WRITER_PROMPT = PromptTemplate.from_template("""You are a Writer Agent. Your ONLY job is to structure and format information clearly.

Rules:
- You receive raw research findings and a format request
- Transform the findings into the requested format
- Do NOT search for new information — use only what is provided
- If the research findings are insufficient, say "INSUFFICIENT_DATA: <what is missing>"
- Keep factual accuracy — do not add or invent information
- Use format_as_table tool when a comparison table is needed

Available tools:
{tools}

Format EXACTLY:
Question: the writing task
Thought: how to structure the output
Action: tool name from [{tool_names}]
Action Input: data to format, one line only
Observation: (provided)
Final Answer: the formatted output

Writing task: {input}
Thought:{agent_scratchpad}""")


def build_writer() -> AgentExecutor:
    llm = ChatHuggingFace (
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY_32"),
            max_new_tokens=600,
            temperature=0.3,   # slight creativity for writing quality
        )
    )
    # from langchain_ollama import ChatOllama
    # llm = ChatOllama(
    #     model="llama3.2",
    #     temperature=0.3,       # Kept at 0.3 for factual accuracy and deterministic choices
    #     num_predict=400,       # This replaces 'max_new_tokens'
    #     # Default local URL is http://localhost:11434, no API key needed
    # )

    agent = create_react_agent(
        llm=llm,
        tools=WRITER_TOOLS,
        prompt=WRITER_PROMPT,
    )

    return AgentExecutor(
        agent=agent,
        tools=WRITER_TOOLS,
        max_iterations=4,
        verbose=False,
        handle_parsing_errors=True,
    )
