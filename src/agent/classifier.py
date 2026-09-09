"""
classifier.py
-------------
Phase 3: Intent Classifier using Groq API (llama-3.3-70b-versatile).
Guarantees schema-conformant JSON output using Pydantic validation
and structured output / json_object mode, with tenacity exponential backoff.
"""

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from groq import Groq
from src.intents import INTENT_SCHEMA, INTENT_NAMES, get_intent_schema_prompt_text

# Load environment variables
load_dotenv()

log = logging.getLogger(__name__)


class IntentClassificationResult(BaseModel):
    """Structured output schema for intent classification."""
    intent: str = Field(
        description="The classified intent category from the allowed set."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    reasoning: str = Field(
        description="Concise 1-sentence reasoning explaining why this intent fits."
    )
    urgency: str = Field(
        default="medium",
        description="Urgency level: low, medium, or high."
    )

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in INTENT_NAMES:
            # If model returned something slightly different, map to closest or fallback
            for valid_name in INTENT_NAMES:
                if valid_name in clean_v or clean_v in valid_name:
                    return valid_name
            return "general_inquiry_feedback"
        return clean_v

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v in ("low", "medium", "high"):
            return clean_v
        return "medium"


SYSTEM_PROMPT = f"""You are an expert customer-support routing intelligence for Amazon (@AmazonHelp).
Your task is to classify incoming customer tweets into exactly ONE of the following 8 predefined intents:

{get_intent_schema_prompt_text()}

Rules:
1. You MUST choose one of the exact intent keys: {INTENT_NAMES}.
2. Provide a calibrated confidence score between 0.0 and 1.0.
3. If the customer is extremely hostile, angry, or threatens legal/social escalation, prioritize `service_complaint_escalation` with high urgency.
4. If the customer message mentions account lockout, unauthorized charges, or OTP, select `account_security_access` with high urgency.
5. Return ONLY a valid JSON object matching the requested schema. No conversational filler or markdown codeblocks.

JSON Schema:
{{
  "intent": "<one of {INTENT_NAMES}>",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<short explanation>",
  "urgency": "<low | medium | high>"
}}
"""


class IntentClassifier:
    """Zero-shot / Few-shot LLM intent classifier powered by Groq."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
    )
    def classify(self, message: str) -> IntentClassificationResult:
        """Classifies an incoming customer tweet into one of the predefined intents."""
        cleaned_message = message.strip()
        if not cleaned_message:
            return IntentClassificationResult(
                intent="general_inquiry_feedback",
                confidence=1.0,
                reasoning="Empty input message.",
                urgency="low",
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify this Amazon customer message:\n\"\"\"{cleaned_message}\"\"\"",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
        )

        raw_content = response.choices[0].message.content.strip()
        # Parse JSON directly or via regex if enclosed in markdown
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                raise ValueError(f"Could not decode JSON from response: {raw_content}")
        return IntentClassificationResult(**data)


if __name__ == "__main__":
    classifier = IntentClassifier()
    test_queries = [
        "3 different people gave 3 different answers and I still don't have my order! Says delivered Saturday, was not!",
        "Can I please cancel order #405-1929312 placed 5 minutes ago?",
        "Someone hacked my account and ordered an iPhone, urgent help needed!",
        "Echo dot won't pair with Bluetooth on my Samsung phone",
    ]
    print("Testing Intent Classifier with Groq Llama-3.3-70b:\n")
    for q in test_queries:
        res = classifier.classify(q)
        print(f"Message: {q}")
        print(f"-> Intent: {res.intent} (Confidence: {res.confidence}, Urgency: {res.urgency})")
        print(f"   Reasoning: {res.reasoning}\n")
