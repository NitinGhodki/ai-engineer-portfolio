"""
DocuAgent — LangChain agent with RAG tool + calculator + date tool.
Combines Day 5 prompt patterns + Day 6 agent patterns.
"""

import os
import math
import datetime
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from core.rag import RAGPipeline
from core.safety import check_input

load_dotenv()

def _clean_input(text: str) -> str:
    """Clean ReAct format bleed from Mistral outputs."""
    text = text.split("\n")[0].strip()
    for keyword in ["Observ", "Observation", "Thought", "Action"]:
        if text.endswith(keyword):
            text = text[:-len(keyword)].strip()
    return text


# ── Agent execution tracer
class ExecutionTracer(BaseCallbackHandler):
    """Records every step for API response and debugging."""

    def __init__(self):
        self.steps = []
        self.tool_call_count = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_call_count += 1
        tool_name = serialized.get("name", "unknown")
        cleaned = _clean_input(input_str)
        self.steps.append({
            "type": "tool_call",
            "tool": tool_name,
            "input": cleaned,
            "result": None,
        })

    def on_tool_end(self, output, **kwargs):
        if self.steps and self.steps[-1]["type"] == "tool_call":
            self.steps[-1]["result"] = str(output)[:200]

    def on_agent_finish(self, finish, **kwargs):
        self.steps.append({
            "type": "final_answer",
            "output": finish.return_values.get('output'),
        })


# ── DocuAgent class

class DocuAgent:
    """
    Main agent class.
    Three tools:
    1. search_documents — RAG retrieval from ChromaDB
    2. calculator       — math operations
    3. get_date         — current date
    """

    def __init__(self, rag: RAGPipeline):
        self._rag = rag
        self._history: list[dict] = []
        self._executor = self._build_executor()

    def _build_tools(self):
        rag = self._rag

        @tool
        def search_documents(query: str) -> str:
            """
            Search the uploaded documents for relevant information.
            Use this tool when the question is about document contents,
            policies, pricing, specifications, or any specific facts.
            Input: a search query string describing what you are looking for.
            """
            query = _clean_input(query)
            results = rag.retrieve(query, top_k=3)

            if not results:
                return "No relevant documents found for this query."

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"[Result {i} | Source: {r['source']} | "
                    f"Relevance: {r['score']}]\n{r['text']}"
                )
            return "\n\n".join(formatted)

        @tool
        def calculator(expression: str) -> str:
            """
            Evaluate a mathematical expression.
            Use for any arithmetic, percentage calculations, or numeric operations.
            Input: a valid math expression like '2999 * 12' or '100 * 0.15'.
            """
            expression = _clean_input(expression)
            try:
                allowed = {
                    k: v for k, v in math.__dict__.items()
                    if not k.startswith("_")
                }
                result = eval(expression, {"__builtins__": {}}, allowed)
                return str(round(float(result), 4))
            except Exception as e:
                return f"Calculator error for '{expression}': {e}"

        @tool
        def get_date(query: str = "") -> str:
            """
            Get today's date and day of week.
            Use when the question involves current date or year.
            """
            date = datetime.datetime.now().strftime("%Y-%m-%d, %A, %B %d %Y")
            print(f"date: {date}")
            return date

        return [search_documents, calculator, get_date]

    def _build_executor(self) -> AgentExecutor:
        tools = self._build_tools()

        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=512,
            temperature=0.1,
        )

        llm = ChatHuggingFace(llm=llm)
        prompt = PromptTemplate.from_template("""You are DocuAgent, a helpful AI assistant.
                You have access to uploaded documents and tools.

                Previous conversation:
                {chat_history}

                Available tools:
                {tools}

                Use this EXACT format:
                Question: the question
                Thought: what to do
                Action: tool name from [{tool_names}]
                Action Input: the value only, on one single line, stop writing here
                Observation: (provided automatically)
                Thought: next step or final reasoning
                Final Answer: complete answer with source references if from documents

                Rules:
                - Action Input must be ONE line only
                - Do not write Observation yourself
                - Always cite which document information came from
                - If no documents are relevant, say so clearly

                Question: {input}
                Thought:{agent_scratchpad}""")

        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=6,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def _format_history(self) -> str:
        if not self._history:
            return "No previous conversation."
        lines = []
        for turn in self._history[-5:]:  # last 5 turns only
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)

    def query(self, question: str) -> dict:
        """
        Main entry point.
        Returns: answer, tool_steps, sources, is_blocked.
        """
        # Safety check first
        safety = check_input(question)
        if not safety["is_safe"]:
            return {
                "answer": f"Request blocked: {safety['reason']}",
                "tool_steps": [],
                "sources": [],
                "is_blocked": True,
                "flagged_pattern": safety["flagged_pattern"],
            }

        tracer = ExecutionTracer()

        result = self._executor.invoke(
            {
                "input": question,
                "chat_history": self._format_history(),
            },
            config={"callbacks": [tracer]},
        )

        answer = result["output"]

        # Extract sources from intermediate steps
        sources = []
        for step in tracer.steps:
            if step["type"] == "tool_call" and step["tool"] == "search_documents":
                if step["result"]:
                    sources.append({
                        "tool": "search_documents",
                        "query": step["input"],
                        "result_preview": step["result"][:150],
                    })

        # Store in history
        self._history.append({"user": question, "assistant": answer})
        if len(self._history) > 5:
            self._history = self._history[-5:]

        return {
            "answer": answer,
            "tool_steps": [s for s in tracer.steps if s["type"] == "tool_call"],
            "sources": sources,
            "is_blocked": False,
            "tool_call_count": tracer.tool_call_count,
        }