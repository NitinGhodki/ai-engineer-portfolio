"""
Day 11 — Build a custom LLM call tracer from scratch.

Tracks every LLM call:
- timestamp
- prompt (truncated)
- response (truncated)
- prompt tokens (approximate)
- completion tokens (approximate)
- latency in ms
- estimated cost in USD
- model used
- call type (rag, agent, eval, direct)

Stores everything as structured JSON logs.
This is what LangSmith does professionally.
You build it manually first to understand what observability means.
"""

import os 
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

load_dotenv()

LOGS_DIR = Path("./week2/day11_observability/llm_logs")
LOGS_DIR.mkdir(exist_ok=True)

# Cost table — approximate pricing per 1K tokens
# Update these when model pricing changes


COST_PER_1K_TOKENS = {
    # HuggingFace Inference API (approximate serverless pricing)
    "mistralai/Mistral-7B-Instruct-v0.3": {"input": 0.0001, "output": 0.0001},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {"input": 0.0006, "output": 0.0006},
    "Qwen/Qwen2.5-7B-Instruct": {"input": 0.0001, "output": 0.0001},
    # OpenAI (for reference comparison)
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Default fallback
    "default": {"input": 0.0001, "output": 0.0001},
}

def approx_tokens(text: str) -> int:
    """Approximate token count. 1 token ≈ 0.75 words."""
    return max(1, int(len(text.split()) / 0.75))


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate estimated cost in USD."""
    pricing = COST_PER_1K_TOKENS.get(model, COST_PER_1K_TOKENS["default"])
    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 8)


# LLM Call Record — one record per LLM invocation
@dataclass
class LLMCallRecord:
    call_id: str
    timestamp: str
    model: str
    call_type: str           # "rag", "agent", "eval", "direct"
    prompt_preview: str      # first 200 chars of prompt
    response_preview: str    # first 200 chars of response
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    session_id: str
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
    

# Tracer — plugs into LangChain as a callback
# LangChain calls these methods automatically at each step

class LLMTracer(BaseCallbackHandler):
    """
    LangChain callback handler that records every LLM call.

    How LangChain callbacks work:
    - on_llm_start: called before LLM receives prompt
    - on_llm_end: called after LLM returns response
    - on_llm_error: called if LLM throws an exception

    LangChain passes these automatically when you add the handler
    to the config: config={"callbacks": [tracer]}
    """

    def __init__(
        self,
        session_id: str = None,
        call_type: str = "direct",
        log_to_file: bool = True,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.call_type = call_type
        self.log_to_file = log_to_file
        self._call_start_times: dict[str, float] = {}
        self._call_prompts: dict[str, str] = {}
        self.records: list[LLMCallRecord] = []

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        """
        Called before LLM processes the prompt.
        We record the start time and prompt here.
        run_id uniquely identifies this specific LLM call.
        """
        run_id = str(kwargs.get("run_id", uuid.uuid4()))
        self._call_start_times[run_id] = time.perf_counter()
        # Join all prompts (usually just one)
        full_prompt = "\n".join(prompts) if prompts else ""
        self._call_prompts[run_id] = full_prompt


    def on_llm_end(self, response: LLMResult, **kwargs):
        """
        Called after LLM returns.
        We compute latency, tokens, cost, and save the record.
        """
        run_id = str(kwargs.get("run_id", ""))
        end_time = time.perf_counter()
        start_time = self._call_start_times.pop(run_id, end_time)
        latency_ms = (end_time - start_time) * 1000

        prompt = self._call_prompts.pop(run_id, "")

        # Extract response text
        response_text = ""
        if response.generations:
            gen = response.generations[0][0]
            response_text = getattr(gen, "text", str(gen))

        # Get model name
        model = "unknown"
        if response.llm_output:
            model = response.llm_output.get("model_name", "unknown")
        if model == "unknown" and hasattr(response, "llm_output") and response.llm_output:
            model = response.llm_output.get("model", "unknown")

        # Token counts — use actual if available, approximate otherwise
        prompt_tokens = approx_tokens(prompt)
        completion_tokens = approx_tokens(response_text)

        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

        cost = calculate_cost(prompt_tokens, completion_tokens, model)

        record = LLMCallRecord(
            call_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            model=model,
            call_type=self.call_type,
            prompt_preview=prompt[:200].replace("\n", " "),
            response_preview=response_text[:200].replace("\n", " "),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=cost,
            session_id=self.session_id,
        )

        self.records.append(record)

        # Print to console
        print(
            f"  [TRACE] {record.call_type} | "
            f"tokens={record.total_tokens} | "
            f"latency={record.latency_ms:.0f}ms | "
            f"cost=${record.cost_usd:.6f}"
        )

        # Save to file
        if self.log_to_file:
            self._append_to_log(record)

    def on_llm_error(self, error: Exception, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        self._call_start_times.pop(run_id, None)
        prompt = self._call_prompts.pop(run_id, "")
        print(f"  [TRACE ERROR] {type(error).__name__}: {error}")

    def _append_to_log(self, record: LLMCallRecord):
        """Append record to daily log file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = LOGS_DIR / f"llm_calls_{date_str}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def session_summary(self) -> dict:
        """Summary of all calls in this session."""
        if not self.records:
            return {"message": "No calls recorded"}
        return {
            "session_id": self.session_id,
            "total_calls": len(self.records),
            "total_tokens": sum(r.total_tokens for r in self.records),
            "total_cost_usd": round(sum(r.cost_usd for r in self.records), 6),
            "avg_latency_ms": round(
                sum(r.latency_ms for r in self.records) / len(self.records), 2
            ),
            "max_latency_ms": max(r.latency_ms for r in self.records),
            "calls_by_type": {
                t: len([r for r in self.records if r.call_type == t])
                for t in set(r.call_type for r in self.records)
            },
        }
    
    def print_session_summary(self):
        summary = self.session_summary()
        print(f"\n{'='*55}")
        print(f"SESSION SUMMARY [{self.session_id}]")
        print(f"{'='*55}")
        for k, v in summary.items():
            print(f"  {k:<25}: {v}")


    # Demo: trace multiple call types in one session
def demo_tracing():
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatHuggingFace(
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=200,
            temperature=0.1,
        )
    )

    # Single shared tracer for the whole session
    tracer = LLMTracer(session_id="demo_session", call_type="direct")

    print("="*55)
    print("TRACING DEMO — watch each call logged")
    print("="*55)

    # Call 1: direct question
    print("\n[Call 1] Direct question")
    tracer.call_type = "direct"
    result = llm.invoke(
        "What is LangChain in one sentence?",
        config={"callbacks": [tracer]},
    )
    print(f"  Answer: {result.content[:80]!r}")

    # Call 2: RAG-style (longer prompt = more tokens)
    print("\n[Call 2] RAG-style with context")
    tracer.call_type = "rag"
    context = "LangGraph is a library for building stateful, multi-actor applications with LLMs. " * 10
    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY this context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke(
        {"context": context, "question": "What is LangGraph?"},
        config={"callbacks": [tracer]},
    )
    print(f"  Answer: {result[:80]!r}")

    # Call 3: agent-style (multiple tool decisions)
    print("\n[Call 3] Agent decision")
    tracer.call_type = "agent"
    result = llm.invoke(
        "Given these tools: calculator, search, calendar. Which tool would you use to find today's date? Reply with just the tool name.",
        config={"callbacks": [tracer]},
    )
    print(f"  Decision: {result.content[:40]!r}")

    # Print per-call table
    print(f"\n{'='*75}")
    print(f"{'CALL LOG':}")
    print(f"{'='*75}")
    print(f"{'#':<4} {'Type':<10} {'Tokens':>8} {'Latency':>10} {'Cost':>12} {'Response preview'}")
    print(f"{'-'*75}")
    for i, r in enumerate(tracer.records, 1):
        print(
            f"{i:<4} {r.call_type:<10} "
            f"{r.total_tokens:>8} "  
            f"{r.latency_ms:>9.0f}ms "
            f"${r.cost_usd:>11.6f}  "
            f"{r.response_preview[:30]!r}"
        )

    tracer.print_session_summary()

    return tracer


if __name__ == "__main__":
    tracer = demo_tracing()
    log_files = list(LOGS_DIR.glob("*.jsonl"))
    print(f"\nLog files created: {[f.name for f in log_files]}")