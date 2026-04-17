from llm_client import LLMClient
from week1.utils.token_utils import count_tokens_approx, estimate_cost_hf
# from week1.utils.token_utils import count_tokens_approx, estimate_cost_hf

QUESTION = "Explain recursion to a 10-year-old."


def zero_shot(client: LLMClient) -> str:
    """No examples, no instruction on how to think — just the question."""
    return client.chat(QUESTION)


def few_shot(client: LLMClient) -> str:
    """
    Prime the model with 2 examples of good simple explanations
    before asking the actual question.
    """
    prompt = f"""Here are two examples of explaining complex ideas simply:

        Example 1:
        Concept: Gravity
        Explanation: Gravity is like an invisible magnet that pulls everything toward the ground. 
        That's why when you drop a ball, it falls down and not up.

        Example 2:
        Concept: Photosynthesis
        Explanation: Plants eat sunlight! They take sunlight, water, and air and turn it into food 
        so they can grow, just like you eat food to grow.

        Now explain this concept the same way:
        Concept: {QUESTION}
        Explanation:
    """

    return client.chat(prompt)


def chain_of_thought(client: LLMClient) -> str:
    """
    Force the model to reason step-by-step before giving the final answer.
    """
    prompt = f"""Think through this step by step before answering:
        1. What is the core idea of the concept?
        2. What everyday thing does a 10-year-old already understand that this is similar to?
        3. Now use that comparison to explain it simply.

        Question: {QUESTION}
    """

    return client.chat(prompt)


if __name__ == "__main__":
    client = LLMClient()
    results = {}

    patterns = {
        "zero_shot": zero_shot,
        "few_shot": few_shot,
        "chain_of_thought": chain_of_thought,
    }

    for name, fn in patterns.items():
        print(f"\n{'='*60}")
        print(f"PATTERN: {name.upper()}")
        print("=" * 60)
        response = fn(client)
        results[name] = response
        print(response)

        tokens = count_tokens_approx(response)
        cost = estimate_cost_hf(tokens)
        print(f"\n[{name}] ~{tokens} tokens | estimated cost: ${cost:.6f}")

    print("\n\n=== COMPARISON SUMMARY ===")
    for name, response in results.items():
        tokens = count_tokens_approx(response)
        print(f"{name:20s} | ~{tokens:4d} tokens | first 80 chars: {response[:80].strip()!r}")