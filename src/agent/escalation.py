"""
escalation.py
-------------
Phase 3: Escalation Decision Engine.
Decides whether an incoming customer message should be auto-handled or escalated
to a human agent — with an explicit, stated reason and priority score.

Combines deterministic safety/policy guardrails with contextual LLM reasoning.
"""

import re
import os
from typing import Optional, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.intents import INTENT_SCHEMA

load_dotenv()


class EscalationDecision(BaseModel):
    """Structured decision output for routing incoming customer queries."""
    should_escalate: bool = Field(
        description="True if message must be transferred to a human agent, False if AI can auto-handle."
    )
    decision: str = Field(
        description="Categorical decision: 'AUTO_HANDLE' or 'ESCALATE_TO_HUMAN'."
    )
    reason: str = Field(
        description="Specific, human-readable justification for the decision."
    )
    priority: str = Field(
        default="medium",
        description="Priority level: 'low', 'medium', 'high', 'critical'."
    )
    rule_triggered: Optional[str] = Field(
        default=None,
        description="The specific policy rule triggered, if applicable."
    )


# ── Hard Deterministic Guardrails ─────────────────────────────────────────────
SECURITY_KEYWORDS = [
    r"\bhack(ed|ing)?\b",
    r"\bunauthori[sz]ed\b",
    r"\bfraud\b",
    r"\bstolen\b",
    r"\bcredit card charge\b",
    r"\bidentity\b",
    r"\bbreach\b",
    r"\bclose(d)? (my )?account\b",
]

LEGAL_ESCALATION_KEYWORDS = [
    r"\blawyer\b",
    r"\blegal action\b",
    r"\bsue\b",
    r"\bsuing\b",
    r"\bbetter business bureau\b",
    r"\bbbb\b",
    r"\bconsumer protection\b",
    r"\bpolice\b",
]

FRUSTRATION_REPEATED_ATTEMPTS = [
    r"\b(spoke|talked|called) to \d+ (people|agents|reps)\b",
    r"\b(transfer(red)?|hung up) \d+ times\b",
    r"\bno one (is helping|helps|replied)\b",
    r"\blied to me\b",
    r"\bunacceptable\b",
]


class EscalationEngine:
    """Evaluates support queries to determine auto-handle vs. human escalation."""

    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = float(
            os.getenv("CONFIDENCE_THRESHOLD", confidence_threshold)
        )

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        urgency: str = "medium",
    ) -> EscalationDecision:
        """
        Determines whether to auto-handle or escalate to human agent.
        Applies a hierarchical decision flow:
          1. Security/Fraud Rules (Critical Priority)
          2. Legal/Regulatory Rules (High Priority)
          3. Severe Hostility/Repeated Failure Rules (High Priority)
          4. Policy Category Checks (e.g. Account Security requires human intervention)
          5. Confidence Threshold (Low confidence = Human Triage)
          6. Standard Auto-Handling
        """
        msg_lower = customer_message.lower()

        # Rule 1: Security & Fraud Guardrail
        for pattern in SECURITY_KEYWORDS:
            if re.search(pattern, msg_lower):
                return EscalationDecision(
                    should_escalate=True,
                    decision="ESCALATE_TO_HUMAN",
                    reason="Message involves sensitive account security, suspected compromise, or fraud.",
                    priority="critical",
                    rule_triggered="SECURITY_FRAUD_TRIGGER",
                )

        # Rule 2: Legal or Regulatory Threat
        for pattern in LEGAL_ESCALATION_KEYWORDS:
            if re.search(pattern, msg_lower):
                return EscalationDecision(
                    should_escalate=True,
                    decision="ESCALATE_TO_HUMAN",
                    reason="Customer threatens legal action or regulatory escalation.",
                    priority="high",
                    rule_triggered="LEGAL_REGULATORY_TRIGGER",
                )

        # Rule 3: Repeated Support Failure / Extreme Frustration
        for pattern in FRUSTRATION_REPEATED_ATTEMPTS:
            if re.search(pattern, msg_lower):
                return EscalationDecision(
                    should_escalate=True,
                    decision="ESCALATE_TO_HUMAN",
                    reason="Customer experienced repeated prior support failures or severe frustration.",
                    priority="high",
                    rule_triggered="REPEATED_FAILURE_TRIGGER",
                )

        # Rule 4: Intent-level policy restrictions
        intent_def = INTENT_SCHEMA.get(predicted_intent)
        if intent_def and not intent_def.auto_handle_allowed:
            return EscalationDecision(
                should_escalate=True,
                decision="ESCALATE_TO_HUMAN",
                reason=intent_def.escalation_reason_default,
                priority="high" if urgency == "high" else "medium",
                rule_triggered="INTENT_POLICY_RESTRICTION",
            )

        # Rule 5: Confidence Threshold Guardrail
        if confidence < self.confidence_threshold:
            return EscalationDecision(
                should_escalate=True,
                decision="ESCALATE_TO_HUMAN",
                reason=f"Classification confidence ({confidence:.2f}) is below safety threshold ({self.confidence_threshold:.2f}).",
                priority="low",
                rule_triggered="LOW_CONFIDENCE_FALLBACK",
            )

        # Default: Safe for Auto-Handling
        return EscalationDecision(
            should_escalate=False,
            decision="AUTO_HANDLE",
            reason=f"Standard transactional query under intent '{predicted_intent}' with high confidence ({confidence:.2f}). Safe for automated resolution.",
            priority=urgency,
            rule_triggered=None,
        )


if __name__ == "__main__":
    engine = EscalationEngine()
    test_cases = [
        ("Where is my package? Tracking says it's running late.", "delivery_delay_missing", 0.92, "medium"),
        ("I want my payments account CLOSED right now!", "account_security_access", 0.88, "high"),
        ("3 different reps gave me 3 different answers and lied to me!", "service_complaint_escalation", 0.95, "high"),
        ("My card was charged $500 without my authorization!", "account_security_access", 0.96, "high"),
        ("I'm going to sue Amazon and report this to the BBB", "service_complaint_escalation", 0.91, "high"),
        ("how does trade in work", "general_inquiry_feedback", 0.45, "low"),
    ]

    print("=== Testing Escalation Engine ===\n")
    for msg, intent, conf, urg in test_cases:
        dec = engine.evaluate(msg, intent, conf, urg)
        print(f"Message: \"{msg}\"")
        print(f"-> Decision: {dec.decision} (Escalate: {dec.should_escalate}, Priority: {dec.priority})")
        print(f"   Reason: {dec.reason}")
        if dec.rule_triggered:
            print(f"   Rule: {dec.rule_triggered}")
        print()
