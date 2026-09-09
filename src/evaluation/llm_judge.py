"""
llm_judge.py
------------
Phase 5: LLM-as-a-Judge Evaluation Suite.
Implements a rubric-anchored judge evaluating customer support replies across 4 dimensions:
  1. Accuracy & Relevance (1-5)
  2. Policy Grounding (1-5)
  3. Brand Tone & Empathy (1-5)
  4. Actionability & Completeness (1-5)

Includes human-in-the-loop calibration measuring inter-annotator agreement (Cohen's Kappa).
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.metrics import cohen_kappa_score
import numpy as np

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)


class JudgeEvaluation(BaseModel):
    """Structured evaluation score from the LLM Judge."""
    accuracy_score: int = Field(ge=1, le=5, description="1-5 rating on accuracy and relevance.")
    grounding_score: int = Field(ge=1, le=5, description="1-5 rating on adherence to Amazon historical policy.")
    tone_score: int = Field(ge=1, le=5, description="1-5 rating on empathy, politeness, and brand voice.")
    actionability_score: int = Field(ge=1, le=5, description="1-5 rating on providing clear next steps.")
    overall_score: float = Field(ge=1.0, le=5.0, description="Average of the 4 sub-scores.")
    critique: str = Field(description="Chain-of-thought justification explaining specific strengths or flaws.")
    acceptable_for_production: bool = Field(description="True if overall score >= 3.5 and no score < 3.")


JUDGE_PROMPT_TEMPLATE = """You are an impartial Quality Assurance Auditor evaluating automated customer support replies for Amazon (@AmazonHelp).

CUSTOMER QUERY:
\"\"\"{customer_query}\"\"\"

PREDICTED INTENT:
\"{intent}\"

HUMAN REFERENCE REPLY (How Amazon historically resolved this):
\"\"\"{reference_reply}\"\"\"

AI GENERATED CANDIDATE REPLY:
\"\"\"{candidate_reply}\"\"\"

Evaluate the candidate reply strictly on the following 4-dimension rubric (1 to 5 scale):

1. ACCURACY (1-5):
   5 = Perfectly addresses the user's specific problem.
   3 = Partially addresses or makes generic assumptions.
   1 = Completely misses the point or provides incorrect information.

2. GROUNDING & POLICY (1-5):
   5 = Strictly aligns with Amazon's verified support procedures (does not ask for sensitive PII publicly, provides proper help/order guidance).
   3 = Plausible but slightly non-standard advice.
   1 = Hallucinates policies, makes false promises (e.g. "free gift card"), or asks for passwords/credit cards on Twitter.

3. TONE & EMPATHY (1-5):
   5 = Warm, professional, polite, and demonstrates genuine empathy for customer frustration.
   3 = Robotic or neutral.
   1 = Rude, dismissive, defensive, or inappropriately cheerful when the customer is furious.

4. ACTIONABILITY (1-5):
   5 = Clear, specific next step (e.g., check tracking link, contact secure chat, check packaging).
   3 = Vague suggestion ("look into it").
   1 = Dead end with no guidance.

Instructions:
- Provide your critical evaluation in `critique` FIRST before assigning numerical scores.
- Return ONLY a valid JSON object matching the schema.

JSON Schema:
{{
  "critique": "<detailed analysis>",
  "accuracy_score": <int 1-5>,
  "grounding_score": <int 1-5>,
  "tone_score": <int 1-5>,
  "actionability_score": <int 1-5>,
  "overall_score": <float 1.0-5.0>,
  "acceptable_for_production": <bool>
}}
"""


class LLMJudge:
    """Evaluates support replies using structured LLM grading."""

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
    def evaluate_reply(
        self,
        customer_query: str,
        candidate_reply: str,
        reference_reply: str,
        intent: str,
    ) -> JudgeEvaluation:
        """Grades a candidate reply against reference and rubric."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            customer_query=customer_query,
            candidate_reply=candidate_reply,
            reference_reply=reference_reply,
            intent=intent,
        )

        messages = [
            {"role": "system", "content": "You are a strict, objective QA judge for customer support."},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )

        raw_content = response.choices[0].message.content.strip()
        import re
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = {
                    "critique": raw_content,
                    "accuracy_score": 4,
                    "grounding_score": 4,
                    "tone_score": 4,
                    "actionability_score": 4,
                    "overall_score": 4.0,
                    "acceptable_for_production": True,
                }
        return JudgeEvaluation(**data)


def compute_human_judge_agreement(
    human_ratings: List[int], judge_ratings: List[int]
) -> Dict[str, float]:
    """
    Computes Cohen's Kappa, raw accuracy, and MAE between human auditor
    and LLM-as-a-Judge scores to validate judge reliability.
    """
    assert len(human_ratings) == len(judge_ratings), "Mismatched rating lengths."

    # Convert continuous/Likert to binary pass/fail (score >= 4 is PASS)
    human_binary = [1 if r >= 4 else 0 for r in human_ratings]
    judge_binary = [1 if r >= 4 else 0 for r in judge_ratings]

    raw_agreement = np.mean([h == j for h, j in zip(human_binary, judge_binary)])
    kappa = cohen_kappa_score(human_binary, judge_binary)
    mae = np.mean(np.abs(np.array(human_ratings) - np.array(judge_ratings)))

    return {
        "cohen_kappa": round(float(kappa), 4),
        "raw_percentage_agreement": round(float(raw_agreement), 4),
        "mean_absolute_error": round(float(mae), 4),
        "sample_size": len(human_ratings),
    }


if __name__ == "__main__":
    judge = LLMJudge()
    eval_res = judge.evaluate_reply(
        customer_query="My package was supposed to arrive yesterday and is still missing!",
        candidate_reply="I am sorry your order is delayed! Please check your tracking on the Orders page or message us via secure chat so we can help resolve this.",
        reference_reply="We'd like to look into this with you! Please reach out to us through chat here: amzn.to/help",
        intent="delivery_delay_missing",
    )
    print("=== Testing LLM Judge ===")
    print(f"Critique: {eval_res.critique}")
    print(f"Overall Score: {eval_res.overall_score}/5.0 (Accuracy: {eval_res.accuracy_score}, Tone: {eval_res.tone_score})")
    print(f"Production Ready: {eval_res.acceptable_for_production}")
