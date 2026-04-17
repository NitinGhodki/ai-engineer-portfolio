# Day 1 — LLM Basics

Building a clean LLM client wrapper and exploring prompt patterns.

## What's here

| File | Purpose |
|------|---------|
| `llm_client.py` | Reusable LLM wrapper — chat, streaming, system prompts |
| `prompt_patterns.py` | Zero-shot vs few-shot vs chain-of-thought comparison |
| `utils/token_utils.py` | Token counting + cost estimation |

## Setup

```bash
uv add huggingface_hub python-dotenv
cp .env.example .env  # add your HF_API_KEY
```

## Run

```bash
python llm_client.py          # test basic chat + streaming
python prompt_patterns.py     # see all 3 patterns side by side
python utils/token_utils.py   # cost comparison
```

## Model used
`mistralai/Mistral-7B-Instruct-v0.3` via HuggingFace Inference API

## Key observations
[Add your own notes here after running the code]