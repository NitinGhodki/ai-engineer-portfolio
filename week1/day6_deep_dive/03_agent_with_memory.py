"""
Agent with conversation memory.

Problem: every invoke() call starts fresh — agent has no memory.
Solution: maintain a conversation history and inject it into each call.

This is how real AI assistants work.
"""

import os
import math
import datetime
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def _clean_input(text: str) -> str:
    text = text.split("\n")[0].strip()
    for keyword in ["Observ", "Observation", "Thought", "Action"]:
        if text.endswith(keyword):
            text = text[:-len(keyword)].strip()
    return text

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Input: math expression like '2 + 2' or '100 * 0.15'.
    """
    expression = _clean_input(expression)
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(float(result), 4))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_date(query: str = "") -> str:
    """Get today's date. Use when current date or year is needed."""
    return datetime.datetime.now().strftime("%Y-%m-%d, %A, %B %d %Y")


TOOLS = [calculator, get_current_date]


class ConversationalAgent:
    """
    Agent that remembers previous turns.
    
    How memory works:
    - Each turn: user message + agent answer stored in history
    - Next turn: history injected into prompt as context
    - LLM sees previous Q&A → can refer back to it
    
    What it enables:
    - "What was my first question?" → agent can answer
    - "Multiply that by 2" → agent knows what "that" refers to
    - Follow-up questions without repeating context
    """

    def __init__(self):
        self._history: list[dict] = []
        self._executor = self._build_executor()

    def _build_executor(self):
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv("Hugging_face_model"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            max_new_tokens=400,
            temperature=0.1,
        )
        llm = ChatHuggingFace(llm=llm)

        # Note: {chat_history} added to prompt
        prompt = PromptTemplate.from_template("""You are a helpful assistant with memory of the conversation.
                Previous conversation:
                {chat_history}

                Available tools:
                {tools}

                Format EXACTLY:
                Question: the question
                Thought: what to do
                Action: tool name from [{tool_names}]
                Action Input: value only, one line
                Observation: (provided)
                Final Answer: answer

                Current question: {input}
                Thought:{agent_scratchpad}""")
        
        agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
        return AgentExecutor(
            agent=agent,
            tools=TOOLS,
            max_iterations=5,
            verbose=False,
            handle_parsing_errors=True,
        )
    
    def _format_history(self) -> str:
        """Convert history list to readable string for prompt injection."""
        if not self._history:
            return "No previous conversation."
        lines = []
        for turn in self._history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)
    
    def chat(self, user_message: str) -> str:
        """
        Send a message. History is automatically injected.
        Answer is automatically stored in history.
        """
        result = self._executor.invoke({
            "input": user_message,
            "chat_history": self._format_history(),
        })

        answer = result["output"]

        # Store this turn in history
        self._history.append({
            "user": user_message,
            "assistant": answer,
        })

        # Keep only last 5 turns — prevents context window overflow
        if len(self._history) > 5:
            self._history = self._history[-5:]

        return answer
    
    def reset(self):
        """Clear conversation history."""
        self._history = []
        print("Conversation history cleared.")

    def show_history(self):
        """Display current conversation history."""
        print(f"\nConversation history ({len(self._history)} turns):")
        for i, turn in enumerate(self._history, 1):
            print(f"  Turn {i}:")
            print(f"    User:      {turn['user']}")
            print(f"    Assistant: {turn['assistant'][:80]}")


def main():
    agent = ConversationalAgent()

    # Multi-turn conversation where context matters
    conversation = [
        "What is 25% of 12000?",
        "Now double that result.",           # needs to remember 3000
        "What year is it currently?",
        "What was my very first question?",  # tests memory
    ]

    print("="*60)
    print("CONVERSATIONAL AGENT — Multi-turn with memory")
    print("="*60)

    for message in conversation:
        print(f"\nUser: {message}")
        response = agent.chat(message)
        print(f"Agent: {response}")

    agent.show_history()

    print("\n" + "="*60)
    print("KEY INSIGHT: history is limited to 5 turns.")
    print("This is the production pattern — unbounded history")
    print("= context window overflow = exponentially growing cost.")
    print("Fixed window = predictable token cost per request.")


if __name__ == "__main__":
    main()