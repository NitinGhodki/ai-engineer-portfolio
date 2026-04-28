import asyncio
import os
import time 
import sys
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient

load_dotenv()

class AsyncLLMClient:
    def __init__(self, system_prompt: str = None):
        self._api_key = os.getenv("HF_API_KEY")
        self.system_prompt = system_prompt
        self._client = AsyncInferenceClient(token=self._api_key)
        self._model = "Qwen/Qwen2.5-7B-Instruct"

    def with_system_prompt(self, system_prompt: str) -> "AsyncLLMClient":
        return AsyncLLMClient(system_prompt=system_prompt)
    
    def _buil_message(self, user_message: str) -> list:
        message = []

        if self.system_prompt:
            message.append({"role": "system", "content": self.system_prompt})

        message.append({"role": "user", "content": user_message})
        return message
    

    async def chat(self, user_message: str, max_token: int = 512) -> str:

        message = self._buil_message(user_message=user_message)
        response = await self._client.chat_completion(
            messages=message,
            model=self._model,
            max_tokens=max_token
        )

        return response.choices[0].message.content
    
    async def astream (self, user_message: str, max_token: int = 512):
        message = self._buil_message(user_message=user_message)
        stream = await self._client.chat_completion(
            messages=message,
            model=self._model,
            max_tokens=max_token,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    yield delta
        print()

    async def batch(self, prompts: list[str], max_token: int = 256) -> list[str]:

        tasks = [self.chat(prompt, max_token=max_token) for prompt in prompts]
        return await asyncio.gather(*tasks)


async def demo_sequential(client: AsyncLLMClient, prompts: list[str]) -> list[str]:
    results = []
    for prompt in prompts:
        result = await client.chat(prompt, max_token = 100)
        results.append(result)
    return result

async def demo_parallel(client: AsyncLLMClient, prompts: list[str]) -> list[str]:
    return await client.batch(prompts, max_token=100)

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():

    client = AsyncLLMClient()

    prompts = [
        "Name one famous physicist. One sentence only.",
        "Name one famous painter. One sentence only.",
        "Name one famous mathematician. One sentence only.",
    ]

    # Sequential timing
    print("=== Sequential calls ===")
    t0 = time.perf_counter()
    seq_results = await demo_sequential(client, prompts)
    seq_time = time.perf_counter() - t0
    for r in seq_results:
        print(f"  → {r.strip()}")
    print(f"Sequential time: {seq_time:.2f}s\n")

    # Parallel timing
    print("=== Parallel calls (asyncio.gather) ===")
    t0 = time.perf_counter()
    par_results = await demo_parallel(client, prompts)
    par_time = time.perf_counter() - t0
    for r in par_results:
        print(f"  → {r.strip()}")
    print(f"Parallel time: {par_time:.2f}s\n")

    # Speedup
    print(f"Speedup: {seq_time / par_time:.1f}x faster with parallel calls")

    # Streaming demo
    print("\n=== Async streaming ===")
    async for chunk in client.astream("Count from 1 to 5, one per line."):
        pass  # printing happens inside astream already


if __name__ == "__main__":
    asyncio.run(main())
