"""
reply_generator.py
------------------
Phase 3: RAG-Grounded Reply Generator.
Synthesizes customer message, predicted intent, and retrieved historical
Amazon support resolutions to draft a brand-aligned, grounded reply.
"""

import os
import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import Groq
from dotenv import load_dotenv

from src.agent.retriever import RetrievedResolution
from src.agent.escalation import EscalationDecision

load_dotenv()
log = logging.getLogger(__name__)


class DraftReply(BaseModel):
    """Structured generated reply output."""
    reply_text: str = Field(
        description="The customer-facing reply text, strictly grounded in Amazon support tone and policies."
    )
    grounded_in_evidence: bool = Field(
        description="True if the reply directly reflects historical resolution practices."
    )
    key_action_offered: str = Field(
        description="The primary next step provided (e.g. check tracking, contact via secure link, request replacement)."
    )
    confidence_in_reply: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence that this reply resolves or correctly routes the issue."
    )


REPLY_SYSTEM_PROMPT = """You are an official customer support representative for Amazon (@AmazonHelp) on Twitter.
Your goal is to draft a helpful, empathetic, and professional reply to an incoming customer tweet.

Strict Guidelines:
1. Grounding: You MUST ground your response in the historical resolutions retrieved from real Amazon support cases. Adopt Amazon's actual resolution protocols and helpful tone.
2. Twitter Length Limit: Keep the reply under 280 characters when possible (max 2-3 concise sentences).
3. Privacy & Safety: NEVER ask customers for sensitive personal information (credit cards, passwords, full address) over public Twitter. Instruct them to reach out via official secure chat/DM or visit their Orders page.
4. Tone: Empathetic, polite, proactive, and constructive (e.g. "I'm sorry for the delay!", "We'd love to help check this for you.").
5. Escalation Handoff: If this case is flagged for human escalation, acknowledge the severity with empathy and direct them immediately to real-time live support or senior specialist chat.
6. Output Format: Return a valid JSON object matching the schema.

JSON Schema:
{
  "reply_text": "<concise Twitter reply>",
  "grounded_in_evidence": true,
  "key_action_offered": "<action summary>",
  "confidence_in_reply": <float 0.0 to 1.0>
}
"""


class ReplyGenerator:
    """Generates grounded responses using retrieved historical support interactions."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=self.api_key)
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
    )
    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        escalation_decision: EscalationDecision,
        retrieved_resolutions: List[RetrievedResolution],
    ) -> DraftReply:
        """Drafts a grounded reply informed by intent, escalation status, and historical data."""

        # Format historical examples for context
        examples_str = ""
        for i, res in enumerate(retrieved_resolutions[:3], 1):
            examples_str += (
                f"Historical Case {i} (Similarity: {res.similarity_score:.2f}):\n"
                f"  Past Customer: \"{res.customer_query}\"\n"
                f"  Amazon Official Reply: \"{res.brand_reply}\"\n\n"
            )

        user_content = (
            f"CUSTOMER INCOMING MESSAGE: \"{customer_message}\"\n\n"
            f"PREDICTED INTENT: {predicted_intent}\n"
            f"ROUTING STATUS: {'ESCALATE TO HUMAN' if escalation_decision.should_escalate else 'AUTO-HANDLE'}\n"
            f"ROUTING REASON: {escalation_decision.reason}\n\n"
            f"HISTORICAL SIMILAR RESOLUTIONS:\n{examples_str or 'No direct historical matches found. Follow standard Amazon policy.'}\n\n"
            f"Draft Amazon's official reply:"
        )

        messages = [
            {"role": "system", "content": REPLY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )

        raw_content = response.choices[0].message.content.strip()
        
        # Try to parse structured JSON if returned
        clean_text = raw_content
        key_action = "Contact support via secure link"
        conf = 0.90
        
        import re
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                clean_text = data.get("reply_text", raw_content)
                key_action = data.get("key_action_offered", key_action)
                conf = data.get("confidence_in_reply", conf)
            except Exception:
                pass

        # Strip any extraneous wrapping quotes or markdown
        clean_text = re.sub(r'^["\']|["\']$', '', clean_text.strip())

        return DraftReply(
            reply_text=clean_text,
            grounded_in_evidence=True,
            key_action_offered=key_action,
            confidence_in_reply=conf,
        )
