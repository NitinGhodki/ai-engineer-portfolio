import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


class LLMClient:
    def __init__(self, system_prompt: str = None):
        self._api_key = os.getenv("HF_API_KEY")
        self._system_prompt = system_prompt
        self._client = InferenceClient(token=self._api_key)

    def with_system_prompt(self, system_prompt: str) -> "LLMClient":
        """Returns a new LLMClient with the system prompt baked in."""
        new_client = LLMClient(system_prompt=system_prompt)
        return new_client

    def _build_messages(self, user_message: str) -> list:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(
        self,
        user_message: str,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        max_tokens: int = 512,
    ) -> str:
        """Single call — returns full response as a string."""
        print(user_message)
        messages = self._build_messages(user_message)
        response = self._client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def stream(
        self,
        user_message: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 512,
    ):
        """Streams tokens as they arrive. Yields each text chunk."""
        messages = self._build_messages(user_message)
        stream = self._client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    yield delta
        print()  # newline after stream ends


if __name__ == "__main__":
    client = LLMClient()

    print("=== Basic chat ===")
    response = client.chat("What is the capital of France? Answer in one sentence.")
    print(response)

    print("\n=== With system prompt ===")
    pirate_client = client.with_system_prompt("You are a pirate. Answer everything like a pirate.")
    response = pirate_client.chat("What is the capital of France? Answer in one sentence.")
    print(response)

    print("\n=== Streaming ===")
    print("Streaming response: ", end="")
    chunks = list(client.stream("Count from 1 to 5, one number per line."))
    print(f"(received {len(chunks)} chunks)")