"""
Researcher Agent — finds and gathers information.

Single responsibility: given a topic or question,
search the knowledge base and return structured research findings.
Tools: search_knowledge_base, search_competitor_info, calculator, get_current_date.
Does NOT write or format — that is the Writer's job.
"""

import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from tools.search_tools import (
    search_knowledge_base,
    search_competitor_info,
    calculator,
    get_current_date,
)

load_dotenv()

RESEARCHER_TOOLS = [
    search_knowledge_base,
    search_competitor_info,
    calculator,
    get_current_date,
]

RESEARCHER_PROMPT = PromptTemplate.from_template("""You are a Research Agent. Your ONLY job is to find facts and data.

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


def build_researcher() -> AgentExecutor:
    llm = ChatHuggingFace (
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY_32"),
            max_new_tokens=512,
            temperature=0.0,   # researcher needs factual accuracy, not creativity
        )
    )

    # llm = ChatOllama(
    #     model="llama3.2",
    #     temperature=0.0,       # Kept at 0.0 for factual accuracy and deterministic choices
    #     num_predict=512,       # This replaces 'max_new_tokens'
    #     # Default local URL is http://localhost:11434, no API key needed
    # )
    
    agent = create_react_agent(
        llm=llm,
        tools=RESEARCHER_TOOLS,
        prompt=RESEARCHER_PROMPT,
    )

    return AgentExecutor(
        agent=agent,
        tools=RESEARCHER_TOOLS,
        max_iterations=6,
        verbose=False,
        handle_parsing_errors=True,
    )