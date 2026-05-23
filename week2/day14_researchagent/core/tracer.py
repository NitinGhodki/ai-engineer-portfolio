"""
LLM call tracer — from Day 11, simplified for integration.
Tracks tokens, cost, latency per call.
"""

import time
import uuid
import json
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

load_dotenv()

LOGS_DIR = Path("./llm_logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


COST_PER_1K = {
    "mistralai/Mistral-7B-Instruct-v0.3": {"input": 0.0001, "output": 0.0001},
    "Qwen/Qwen2.5-7B-Instruct": {"input": 0.00004, "output": 0.00007},
    "meta-llama/Llama-3.2-3B-Instruct": {"input": 0.0, "output": 0.0},  # Free Tier
    "microsoft/Phi-3-mini-4k-instruct": {"input": 0.0, "output": 0.0},  # Free Tier
    "default": {"input": 0.0, "output": 0.0},
}

def approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def calc_cost(prompt_t: int, completion_t: int, model: str) -> float:
    p = COST_PER_1K.get(model, COST_PER_1K["default"])
    return round((prompt_t / 1000) * p["input"] + (completion_t / 1000) * p["output"], 8)


@dataclass
class CallRecord:
    call_id: str
    timestamp: str
    model: str
    agent_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    session_id: str


class AgentTracer(BaseCallbackHandler):
    """Plugs into LangChain — records every LLM call automatically."""

    def __init__(self, agent_name: str = "unknown", session_id: str = None):
        self.agent_name = agent_name
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.records: list[CallRecord] = []
        self._start_times: dict = {}
        self._prompts: dict = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        run_id = str(kwargs.get("run_id", uuid.uuid4()))
        self._start_times[run_id] = time.perf_counter()
        self._prompts[run_id] = "\n".join(prompts) if prompts else ""

    def on_llm_end(self, response: LLMResult, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        latency_ms = (time.perf_counter() - self._start_times.pop(run_id, time.perf_counter())) * 1000
        prompt = self._prompts.pop(run_id, "")

        response_text = ""
        if response.generations:
            gen = response.generations[0][0]
            response_text = getattr(gen, "text", str(gen))

        model = os.getenv("Hugging_face_model")
        prompt_t = approx_tokens(prompt)
        completion_t = approx_tokens(response_text)

        record = CallRecord(
            call_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            model=model,
            agent_name=self.agent_name,
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            total_tokens=prompt_t + completion_t,
            latency_ms=round(latency_ms, 2),
            cost_usd=calc_cost(prompt_t, completion_t, model),
            session_id=self.session_id,
        )
        self.records.append(record)
        self._save(record)

    def on_llm_error(self, error, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        self._start_times.pop(run_id, None)
        self._prompts.pop(run_id, None)

    def _save(self, record: CallRecord):
        date = datetime.now().strftime("%Y-%m-%d")
        log_file = LOGS_DIR / f"calls_{date}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def summary(self) -> dict:
        if not self.records:
            return {"total_calls": 0, "total_cost_usd": 0, "total_tokens": 0}
        return {
            "agent": self.agent_name,
            "total_calls": len(self.records),
            "total_tokens": sum(r.total_tokens for r in self.records),
            "total_cost_usd": round(sum(r.cost_usd for r in self.records), 6),
            "avg_latency_ms": round(
                sum(r.latency_ms for r in self.records) / len(self.records), 1
            ),
        }