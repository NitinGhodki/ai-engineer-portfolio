"""
Production-grade retry logic for LLM API calls.
Uses tenacity — the standard Python retry library.
"""
import time
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def llm_retry(max_attempts: int = 3, min_wait: float = 1, max_wait: float = 10):
    """
    Decorator: retry an LLM call on rate limit or connection errors.
    Uses exponential backoff — waits 1s, then 2s, then 4s between retries.

    Exponential backoff explained:
    - Attempt 1 fails → wait 1s
    - Attempt 2 fails → wait 2s
    - Attempt 3 fails → wait 4s
    Why? So you don't hammer a rate-limited API — you back off progressively.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )



def retry_with_backoff(fn, max_attempts: int = 3, base_delay: float = 1.0):
    """
    Manual retry loop with exponential backoff.
    Use this when you want to inspect or modify behavior between retries.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait = base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s
            print(f"  Attempt {attempt} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    call_count = 0

    def flaky_llm_call():
        """Simulates a function that fails 70% of the time (like a rate-limited API)."""
        global call_count
        call_count += 1
        print(f"  Making call #{call_count}...")
        if random.random() < 0.7:
            raise Exception("429 Rate limit exceeded")
        return "Success! Got a response."

    print("=== Testing retry_with_backoff ===")
    try:
        result = retry_with_backoff(flaky_llm_call, max_attempts=5, base_delay=0.5)
        print(f"Final result: {result}")
    except Exception as e:
        print(f"All retries exhausted: {e}")

    print(f"\nTotal API calls made: {call_count}")