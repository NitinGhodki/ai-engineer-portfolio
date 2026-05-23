import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from agents.tools import calculator
from core.tracer import AgentTracer

load_dotenv()

PROMPT = PromptTemplate.from_template("""You are a Critic Agent. Your ONLY job is to verify accuracy and quality.

Rules:
- Check every factual claim against the knowledge base
- Verify all numbers with the calculator
- Be specific about issues — not "it's wrong" but "the price stated is X but actual price is Y"
- Give a verdict: APPROVED or NEEDS_REVISION
- If NEEDS_REVISION, list each issue clearly numbered

Available tools:
{tools}

Format EXACTLY:
Question: the review task
Thought: what facts to verify first
Action: tool name from [{tool_names}]
Action Input: what to verify, one line only
Observation: (provided)
Thought: what else to verify
Final Answer: APPROVED or NEEDS_REVISION\n<issues if any>

Review task: {input}
Thought:{agent_scratchpad}""")


def build_critic(rag, session_id: str = None):
    from agents.tools import make_search_tool
    search_kb, _ = make_search_tool(rag)
    tools = [search_kb, calculator]
    tracer = AgentTracer(agent_name="critic", session_id=session_id)

    llm = ChatHuggingFace (
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=400,
            temperature=0.0, 
            callbacks=[tracer],
        )
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=PROMPT)
    # agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=PROMPT)
    return AgentExecutor(
        agent=agent, tools=tools,
        max_iterations=5, verbose=False,
        handle_parsing_errors=True,
    ), tracer