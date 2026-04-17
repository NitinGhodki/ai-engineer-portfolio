"""
Token counting for Hugging Face models.

Note: Exact token counts require loading the model's tokenizer (slow, needs transformers).
We use a fast approximation here: 1 token ≈ 0.75 words (standard rule of thumb).
For production, swap this with the actual tokenizer.
"""


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count using the 0.75 words-per-token heuristic.
    Accurate to within ~10% for English text on most models.
    """
    word_count = len(text.split())
    return int(word_count / 0.75)


def estimate_cost_hf(token_count: int) -> float:
    """
    Hugging Face Inference API pricing (Serverless, as of early 2026).
    Free tier: limited requests/month.
    Beyond free tier: ~$0.0001 per 1K tokens (varies by model).
    
    For comparison, we also show what the same call would cost on OpenAI.
    """
    HF_COST_PER_1K = 0.0001       # Serverless tier estimate
    OPENAI_GPT4O_PER_1K = 0.005   # gpt-4o input pricing
    OPENAI_GPT35_PER_1K = 0.0005  # gpt-3.5-turbo input pricing

    return (token_count / 1000) * HF_COST_PER_1K


def compare_costs(prompt: str, completion: str) -> dict:
    """
    Given a prompt and completion, return a cost breakdown across providers.
    Useful for production decisions.
    """
    prompt_tokens = count_tokens_approx(prompt)
    completion_tokens = count_tokens_approx(completion)
    total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_hf_usd": (total_tokens / 1000) * 0.0001,
        "cost_openai_gpt4o_usd": (total_tokens / 1000) * 0.005,
        "cost_openai_gpt35_usd": (total_tokens / 1000) * 0.0005,
    }


if __name__ == "__main__":
    sample = "This is a test sentence to check token counting accuracy."
    tokens = count_tokens_approx(sample)
    print(f"Text: {sample!r}")
    print(f"Approx tokens: {tokens}")

    breakdown = compare_costs("Tell me about recursion.", "Recursion is when a function calls itself...")
    print("\nCost comparison:")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")