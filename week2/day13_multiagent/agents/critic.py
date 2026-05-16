"""
Critic Agent — reviews output for accuracy and quality.

Single responsibility: given a written output and the original research,
check for factual errors, missing information, and quality issues.
Tools: search_knowledge_base (to verify facts), calculator (to verify numbers)
Returns: APPROVED or NEEDS_REVISION with specific issues listed.
"""

import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools.search_tools import search_knowledge_base, calculator
from langchain_ollama import ChatOllama

load_dotenv()

CRITIC_TOOLS = [search_knowledge_base, calculator]

CRITIC_PROMPT = PromptTemplate.from_template("""You are a Critic Agent. Your ONLY job is to verify accuracy and quality.

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


def build_critic() -> AgentExecutor:
    llm = ChatHuggingFace (
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY_32"),
            max_new_tokens=400,
            temperature=0.0,   # critic needs strict accuracy
        )
    )
    # llm = ChatOllama(
    #     model="llama3.2",
    #     temperature=0.0,       # Kept at 0.0 for factual accuracy and deterministic choices
    #     num_predict=400,       # This replaces 'max_new_tokens'
    #     # Default local URL is http://localhost:11434, no API key needed
    # )
    agent = create_react_agent(
        llm=llm,
        tools=CRITIC_TOOLS,
        prompt=CRITIC_PROMPT,
    )

    return AgentExecutor(
        agent=agent,
        tools=CRITIC_TOOLS,
        max_iterations=5,
        verbose=False,
        handle_parsing_errors=True,
    )

