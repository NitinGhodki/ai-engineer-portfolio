import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from agents.tools import calculator, get_date

load_dotenv()

PROMPT = PromptTemplate.from_template("""You are a Research Agent. Find facts and data only.

Rules:
- Search thoroughly before concluding
- Always cite which source you found information from
- If asked for numbers, verify with calculator
- Do NOT write summaries or format nicely — just report raw findings
- Be specific: include exact numbers, dates, and terms

Available tools:
{tools}

Format EXACTLY:
Question: the research task
Thought: what to search for
Action: tool name from [{tool_names}]
Action Input: search query, one line only
Observation: (provided)
Thought: what else to search or if done
Final Answer: raw research findings with sources

Research task: {input}
Thought:{agent_scratchpad}""")


def build_researcher(rag, session_id: str = None):
    from agents.tools import make_search_tool
    from core.tracer import AgentTracer

    search_kb, search_comp = make_search_tool(rag)
    tools = [search_kb, search_comp, calculator, get_date]

    tracer = AgentTracer(agent_name="researcher", session_id=session_id)

    llm = ChatHuggingFace (
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY_32"),
            max_new_tokens=400,
            temperature=0.0, 
            callbacks=[tracer],
        )
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=PROMPT)
    # agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=PROMPT)
    return AgentExecutor(
        agent=agent, 
        tools=tools,
        max_iterations=6, 
        verbose=False,
        handle_parsing_errors=True,
    ), tracer

