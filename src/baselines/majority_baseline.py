"""
majority_baseline.py
--------------------
Phase 6: Baseline 1 (Trivial Baseline).
Represents the simplest zero-intelligence heuristic:
  1. Always predicts the majority intent class ('delivery_delay_missing').
  2. Always outputs a fixed canned reply template.
  3. Never escalates to a human (auto-handles everything).
"""

from typing import Dict, Any


class MajorityBaselineAgent:
    """Trivial baseline: always predicts majority class and static canned reply."""

    def __init__(self, majority_intent: str = "delivery_delay_missing"):
        self.majority_intent = majority_intent
        self.canned_reply = (
            "Hi there! Please reach out to our team via DM with your order details "
            "so we can take a closer look at this for you."
        )

    def predict(self, customer_message: str) -> Dict[str, Any]:
        return {
            "predicted_intent": self.majority_intent,
            "intent_confidence": 0.50,
            "should_escalate": False,
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason": "Trivial baseline default: never escalate.",
            "reply_text": self.canned_reply,
        }
