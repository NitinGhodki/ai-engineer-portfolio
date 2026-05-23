import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from agents.tools import format_table
from core.tracer import AgentTracer

load_dotenv()

PROMPT = PromptTemplate.from_template("""You are a Writer Agent. Structure and format information only.

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


def build_writer(session_id: str = None):
    tracer = AgentTracer(agent_name="writer", session_id=session_id)

    llm = ChatHuggingFace (
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY_32"),
            max_new_tokens=400,
            temperature=0.2, 
            callbacks=[tracer],
        ) 
    )

    tools = [format_table]
    agent = create_react_agent(llm=llm, tools=tools, prompt=PROMPT)
    # agent =create_tool_calling_agent(llm=llm, tools=tools, prompt=PROMPT)
    return AgentExecutor(
        agent=agent, tools=tools,
        max_iterations=5, verbose=False,
        handle_parsing_errors=True,
    ), tracer