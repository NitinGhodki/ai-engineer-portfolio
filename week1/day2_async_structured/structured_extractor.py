import asyncio
import json
import os
import re
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from huggingface_hub import InferenceClient
import ollama

load_dotenv()

class Person(BaseModel):
    name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    skills: list[str] = Field(default_factory=list)

class JobPosting(BaseModel):
    job_title: str
    company: str
    location: Optional[str] = None
    required_skill: list[str] = Field(default_factory=list)
    experience_years: Optional[int] = None
    salary_range: Optional[str] = None


class SupportTicket(BaseModel):
    category: str = Field(description="One of: billing, technical, account, general")
    priority: str = Field(description="One of: low, medium, high, urgent")
    summary: str = Field(description="One sentence summary of the issue")
    sentiment: str = Field(description="One of: positive, neutral, negative, angry")


class StructuredExtractor: 
    def __init__(self):
        self._client = InferenceClient(token=os.getenv("HF_API_KEY"))
        self._model = os.getenv("Hugging_face_model")

    def _build_extraction_prompt(self, schema: type[BaseModel], text: str) -> str:
        fields = schema.model_fields
        field_description = []

        for name, field in fields.items():
            annotation = field.annotation
            desc = field.description if field.description else ""
            field_description.append(f"     -{name} ({annotation}): {desc}")

        fields_str = "\n".join(field_description)

        return f"""Extract information from the text below and return ONLY a valid JSON object.
            Do not include any explanation, markdown, or code blocks. Just raw JSON.

            Required JSON fields:
            {fields_str}

            Text to extract from:
            \"\"\"{text}\"\"\"

            JSON output:"""
    
    def extract(self, schema: type[BaseModel], text: str, max_retries: int = 3) -> BaseModel:
        """
        Extract structured data from text. Retries on validation failure.
        This retry-on-validation pattern is used in production AI systems.
        """
        prompt = self._build_extraction_prompt(schema, text)
        last_error = None

        for attempt in range(1, max_retries + 1):
            print(f"  [Attempt {attempt}/{max_retries}]")

            response = self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()

            # response = ollama.chat(
            #     model="llama3.2",
            #     messages=[{"role": "user", "content": prompt}],
            #     options={"temperature": 0.3}
            # )
            # raw = response["message"]["content"].strip()

            # Clean up common LLM formatting mistakes
            raw = self._clean_json_response(raw)

            try:
                data = json.loads(raw)
                result = schema(**data)
                print(f"  Extracted successfully on attempt {attempt}")
                return result

            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                last_error = e
                print(f"  Failed: {type(e).__name__}: {e}")

                # On retry, add error context to force correction
                if attempt < max_retries:
                    prompt += f"\n\nYour previous response was invalid: {str(e)}\nTry again with valid JSON only:"

        raise ValueError(f"Extraction failed after {max_retries} attempts. Last error: {last_error}")

    def _clean_json_response(self, raw: str) -> str:
        """Strip markdown code blocks and whitespace that LLMs often add."""
        # Remove ```json ... ``` blocks
        raw = re.sub(r"```(?:json)?\s*", "", raw)
        raw = re.sub(r"```", "", raw)
        # Find the first { and last } — extract just the JSON object
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return raw[start:end]
        return raw.strip()
    


def main():
    extractor = StructuredExtractor()

    # Test 1: Person extraction
    print("\n" + "="*60)
    print("TEST 1: Person extraction")
    print("="*60)
    person_text = """
    Meet Sarah Chen, a TWINTY-NINE-year-old senior software engineer at a fintech startup.
    She has strong expertise in Python, Kubernetes, and distributed systems.
    She also knows React and has been learning Rust lately.
    """
    person = extractor.extract(Person, person_text)
    print(f"Result: {person.model_dump_json(indent=2)}")

    # Test 2: Job posting extraction
    print("\n" + "="*60)
    print("TEST 2: Job posting extraction")
    print("="*60)
    job_text = """
    We're hiring a Machine Learning Engineer at DataFlow Inc., based in Bangalore.
    You'll need 3+ years of experience, strong Python skills, knowledge of PyTorch or
    TensorFlow, and experience with MLOps tools like MLflow or Kubeflow.
    Compensation: 25-35 LPA depending on experience.
    """
    job = extractor.extract(JobPosting, job_text)
    print(f"Result: {job.model_dump_json(indent=2)}")

    # Test 3: Support ticket classification
    print("\n" + "="*60)
    print("TEST 3: Support ticket classification")
    print("="*60)
    ticket_text = """
    I've been charged TWICE this month and nobody is helping me!!
    I've sent 3 emails and called support twice. This is absolutely unacceptable.
    I want a full refund immediately or I'm disputing the charge with my bank.
    """
    ticket = extractor.extract(SupportTicket, ticket_text)
    print(f"Result: {ticket.model_dump_json(indent=2)}")

    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Person:  {person.name}, {person.age}, skills: {person.skills}")
    print(f"Job:     {job.job_title} at {job.company}, {job.experience_years}yr exp")
    print(f"Ticket:  [{ticket.priority.upper()}] {ticket.category} — {ticket.sentiment}")


if __name__ == "__main__":
    main()