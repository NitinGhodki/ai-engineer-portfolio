"""
Day 11 — Cost dashboard.

Reads the JSONL log files created by LLMTracer.
Answers real production questions:
- What did this system cost today?
- Which queries are most expensive?
- What is the average cost per session?
- Which call types dominate the cost?
- Are latencies getting worse over time?
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_all_logs(logs_dir: str = "./week2/day11_observability/llm_logs") -> list[dict]:
    """Load all JSONL log files from the logs directory."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"Logs directory not found: {logs_dir}")
        return []
    
    all_records = []
    for log_file in sorted(logs_path.glob("*.jsonl")):
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            
    print(f"Loaded {len(all_records)} records from {len(list(logs_path.glob('*.jsonl')))} log files")
    return all_records


def dashboard(records: list[dict]):
    """Print a full cost and performance dashboard."""
    if not records:
        print("No records to display.")
        return

    print("\n" + "="*65)
    print("LLM OBSERVABILITY DASHBOARD")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    # ── SECTION 1: Overall stats ──────────────────────────────────────────────
    total_calls = len(records)
    total_tokens = sum(r.get("total_tokens", 0) for r in records)
    total_cost = sum(r.get("cost_usd", 0) for r in records)
    avg_latency = sum(r.get("latency_ms", 0) for r in records) / total_calls
    max_latency = max(r.get("latency_ms", 0) for r in records)

    print(f"\n{'OVERALL':}")
    print(f"  Total calls:         {total_calls}")
    print(f"  Total tokens:        {total_tokens:,}")
    print(f"  Total cost:          ${total_cost:.6f} USD")
    print(f"  Cost per call avg:   ${total_cost/total_calls:.6f} USD")
    print(f"  Avg latency:         {avg_latency:.0f}ms")
    print(f"  Max latency:         {max_latency:.0f}ms")

    # Projection
    calls_per_day_estimate = total_calls
    projected_monthly = total_cost / total_calls * calls_per_day_estimate * 30
    print(f"\n  Projected monthly cost (at {calls_per_day_estimate} calls/day): ${projected_monthly:.4f}")

    # ── SECTION 2: Cost by call type ──────────────────────────────────────────
    by_type: dict = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "latencies": []})
    for r in records:
        t = r.get("call_type", "unknown")
        by_type[t]["calls"] += 1
        by_type[t]["tokens"] += r.get("total_tokens", 0)
        by_type[t]["cost"] += r.get("cost_usd", 0)
        by_type[t]["latencies"].append(r.get("latency_ms", 0))

    print(f"\n{'COST BY CALL TYPE':}")
    print(f"  {'Type':<15} {'Calls':>7} {'Tokens':>10} {'Cost':>12} {'Avg Latency':>12}")
    print(f"  {'-'*60}")
    for call_type, stats in sorted(by_type.items(), key=lambda x: x[1]["cost"], reverse=True):
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"])
        print(
            f"  {call_type:<15} "
            f"{stats['calls']:>7} "
            f"{stats['tokens']:>10,} "
            f"${stats['cost']:>11.6f} "
            f"{avg_lat:>11.0f}ms"
        )

    # ── SECTION 3: Most expensive individual calls ────────────────────────────
    print(f"\n{'TOP 5 MOST EXPENSIVE CALLS':}")
    print(f"  {'#':<4} {'Type':<10} {'Tokens':>8} {'Cost':>12} {'Latency':>10}  Prompt preview")
    print(f"  {'-'*70}")
    sorted_by_cost = sorted(records, key=lambda r: r.get("cost_usd", 0), reverse=True)[:5]
    for i, r in enumerate(sorted_by_cost, 1):
        prompt_preview = r.get("prompt_preview", "")[:30].replace("\n", " ")
        print(
            f"  {i:<4} "
            f"{r.get('call_type', '?'):<10} "
            f"{r.get('total_tokens', 0):>8,} "
            f"${r.get('cost_usd', 0):>11.6f} "
            f"{r.get('latency_ms', 0):>9.0f}ms  "
            f"{prompt_preview!r}"
        )

    # ── SECTION 4: Slowest calls ──────────────────────────────────────────────
    print(f"\n{'TOP 5 SLOWEST CALLS':}")
    print(f"  {'#':<4} {'Type':<10} {'Latency':>10} {'Tokens':>8}  Response preview")
    print(f"  {'-'*65}")
    sorted_by_latency = sorted(records, key=lambda r: r.get("latency_ms", 0), reverse=True)[:5]
    for i, r in enumerate(sorted_by_latency, 1):
        resp_preview = r.get("response_preview", "")[:30].replace("\n", " ")
        print(
            f"  {i:<4} "
            f"{r.get('call_type', '?'):<10} "
            f"{r.get('latency_ms', 0):>9.0f}ms "
            f"{r.get('total_tokens', 0):>8,}  "
            f"{resp_preview!r}"
        )

    # ── SECTION 5: Cost by session ────────────────────────────────────────────
    by_session: dict = defaultdict(lambda: {"calls": 0, "cost": 0.0})
    for r in records:
        sid = r.get("session_id", "unknown")
        by_session[sid]["calls"] += 1
        by_session[sid]["cost"] += r.get("cost_usd", 0)

    print(f"\n{'COST BY SESSION':}")
    print(f"  {'Session':<15} {'Calls':>7} {'Cost':>12}")
    print(f"  {'-'*36}")
    for sid, stats in sorted(by_session.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {sid:<15} {stats['calls']:>7} ${stats['cost']:>11.6f}")

    # ── SECTION 6: Token distribution ────────────────────────────────────────
    token_counts = [r.get("total_tokens", 0) for r in records]
    token_counts.sort()
    p50 = token_counts[len(token_counts) // 2]
    p90 = token_counts[int(len(token_counts) * 0.9)]
    p99 = token_counts[int(len(token_counts) * 0.99)] if len(token_counts) > 10 else token_counts[-1]

    print(f"\n{'TOKEN DISTRIBUTION':}")
    print(f"  p50 (median):   {p50:,} tokens")
    print(f"  p90:            {p90:,} tokens")
    print(f"  p99:            {p99:,} tokens")
    print(f"  Max:            {max(token_counts):,} tokens")
    print()
    print(f"  If p99 >> p50: you have expensive outlier queries.")
    print(f"  Consider: input length limits, context truncation, caching.")

    print("\n" + "="*65)


def generate_sample_logs():
    """
    Generate sample log data so the dashboard has something to display.
    In production this comes from real LLMTracer logs.
    """
    import random
    import uuid

    logs_dir = Path("./week2/day11_observability/llm_logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"llm_calls_{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    call_types = ["rag", "agent", "direct", "eval"]
    models = ["mistralai/Mistral-7B-Instruct-v0.3"]

    print("Generating sample log data...")
    records = []
    for _ in range(30):
        call_type = random.choice(call_types)
        # RAG calls are longer/costlier than direct calls
        base_tokens = {"rag": 800, "agent": 600, "direct": 200, "eval": 1200}[call_type]
        tokens = base_tokens + random.randint(-100, 300)
        tokens = max(50, tokens)

        prompt_t = int(tokens * 0.7)
        comp_t = tokens - prompt_t
        model = random.choice(models)
        cost = calculate_cost_simple(prompt_t, comp_t, model)

        record = {
            "call_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "call_type": call_type,
            "prompt_preview": f"Sample {call_type} prompt asking about AI systems and RAG pipelines",
            "response_preview": f"Sample response about {call_type} results from the LLM model",
            "prompt_tokens": prompt_t,
            "completion_tokens": comp_t,
            "total_tokens": tokens,
            "latency_ms": round(random.uniform(800, 5000), 2),
            "cost_usd": cost,
            "session_id": f"sess_{random.randint(1, 5):03d}",
            "metadata": {},
        }
        records.append(record)

    with open(log_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} sample records → {log_file}")


def calculate_cost_simple(prompt_tokens, completion_tokens, model):
    pricing = {"input": 0.0001, "output": 0.0001}
    return round((prompt_tokens / 1000) * pricing["input"] + (completion_tokens / 1000) * pricing["output"], 8)


if __name__ == "__main__":
    records = load_all_logs()
    if not records:
        print("No logs found. Generating sample data...")
        generate_sample_logs()
        records = load_all_logs()
    dashboard(records)