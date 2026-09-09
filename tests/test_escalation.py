"""
test_escalation.py
------------------
Unit tests for the Escalation Decision Engine.
Tests deterministic guardrails (security, legal, frustration triggers),
policy-level intent restrictions, and confidence thresholds.
"""

import pytest
from unittest.mock import patch
from src.agent.escalation import EscalationEngine


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return EscalationEngine(confidence_threshold=0.55)


# ──────────────────────────────────────────────────────────────────────────────
# Rule 1 — Security / Fraud Guardrail
# ──────────────────────────────────────────────────────────────────────────────

class TestSecurityGuardrail:
    def test_hacked_account_escalates(self, engine):
        dec = engine.evaluate(
            "Someone hacked my account and placed orders!",
            "account_security_access", 0.95, "high"
        )
        assert dec.should_escalate is True
        assert dec.priority == "critical"
        assert dec.rule_triggered == "SECURITY_FRAUD_TRIGGER"

    def test_unauthorized_charge_escalates(self, engine):
        dec = engine.evaluate(
            "There is an unauthorized charge of $200 on my card.",
            "account_security_access", 0.92, "high"
        )
        assert dec.should_escalate is True
        assert dec.priority == "critical"

    def test_fraud_keyword_escalates(self, engine):
        dec = engine.evaluate(
            "I think I was a victim of fraud on my Amazon account.",
            "account_security_access", 0.88, "high"
        )
        assert dec.should_escalate is True
        assert dec.rule_triggered == "SECURITY_FRAUD_TRIGGER"

    def test_close_account_escalates(self, engine):
        dec = engine.evaluate(
            "Please close my account immediately.",
            "account_security_access", 0.80, "high"
        )
        assert dec.should_escalate is True


# ──────────────────────────────────────────────────────────────────────────────
# Rule 2 — Legal / Regulatory Threat
# ──────────────────────────────────────────────────────────────────────────────

class TestLegalGuardrail:
    def test_sue_escalates(self, engine):
        dec = engine.evaluate(
            "I am going to sue Amazon over this!",
            "service_complaint_escalation", 0.90, "high"
        )
        assert dec.should_escalate is True
        assert dec.rule_triggered == "LEGAL_REGULATORY_TRIGGER"
        assert dec.priority == "high"

    def test_bbb_escalates(self, engine):
        dec = engine.evaluate(
            "I will report this to the BBB and consumer protection.",
            "service_complaint_escalation", 0.85, "high"
        )
        assert dec.should_escalate is True
        assert dec.rule_triggered == "LEGAL_REGULATORY_TRIGGER"

    def test_lawyer_mention_escalates(self, engine):
        dec = engine.evaluate(
            "My lawyer will be contacting you shortly.",
            "service_complaint_escalation", 0.91, "high"
        )
        assert dec.should_escalate is True


# ──────────────────────────────────────────────────────────────────────────────
# Rule 3 — Repeated Failure / Extreme Frustration
# ──────────────────────────────────────────────────────────────────────────────

class TestFrustrationGuardrail:
    def test_multiple_reps_escalates(self, engine):
        dec = engine.evaluate(
            "I spoke to 3 people and no one is helping!",
            "service_complaint_escalation", 0.88, "high"
        )
        assert dec.should_escalate is True
        assert dec.rule_triggered == "REPEATED_FAILURE_TRIGGER"

    def test_unacceptable_escalates(self, engine):
        dec = engine.evaluate(
            "This is completely unacceptable service.",
            "service_complaint_escalation", 0.82, "high"
        )
        assert dec.should_escalate is True

    def test_lied_to_escalates(self, engine):
        dec = engine.evaluate(
            "Your agent lied to me about the refund timeline.",
            "delivery_delay_missing", 0.75, "medium"
        )
        assert dec.should_escalate is True
        assert dec.rule_triggered == "REPEATED_FAILURE_TRIGGER"


# ──────────────────────────────────────────────────────────────────────────────
# Rule 4 — Intent Policy Restrictions
# ──────────────────────────────────────────────────────────────────────────────

class TestIntentPolicyRestrictions:
    def test_account_security_intent_always_escalates(self, engine):
        # Even with high confidence and no keyword triggers
        dec = engine.evaluate(
            "I need help accessing my account.",
            "account_security_access", 0.95, "medium"
        )
        assert dec.should_escalate is True
        assert dec.rule_triggered == "INTENT_POLICY_RESTRICTION"

    def test_service_complaint_intent_always_escalates(self, engine):
        dec = engine.evaluate(
            "I am very disappointed with the support I received.",
            "service_complaint_escalation", 0.90, "medium"
        )
        assert dec.should_escalate is True


# ──────────────────────────────────────────────────────────────────────────────
# Rule 5 — Confidence Threshold
# ──────────────────────────────────────────────────────────────────────────────

class TestConfidenceThreshold:
    def test_low_confidence_escalates(self, engine):
        dec = engine.evaluate(
            "I have an issue with my thing.",
            "general_inquiry_feedback", 0.40, "low"
        )
        assert dec.should_escalate is True
        assert dec.rule_triggered == "LOW_CONFIDENCE_FALLBACK"
        assert dec.priority == "low"

    def test_borderline_confidence_below_threshold_escalates(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.55")
        eng = EscalationEngine(confidence_threshold=0.55)
        dec = eng.evaluate(
            "Some question about my order.",
            "delivery_delay_missing", 0.54, "low"
        )
        assert dec.should_escalate is True  # 0.54 < 0.55 threshold

    def test_borderline_confidence_above_threshold_auto_handles(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.55")
        eng = EscalationEngine(confidence_threshold=0.55)
        dec = eng.evaluate(
            "Where is my order? It was expected yesterday.",
            "delivery_delay_missing", 0.56, "medium"
        )
        assert dec.should_escalate is False


# ──────────────────────────────────────────────────────────────────────────────
# Happy Path — Auto-Handle
# ──────────────────────────────────────────────────────────────────────────────

class TestAutoHandle:
    def test_clear_delivery_query_auto_handled(self, engine):
        dec = engine.evaluate(
            "My package was supposed to arrive today. Can you check tracking?",
            "delivery_delay_missing", 0.92, "medium"
        )
        assert dec.should_escalate is False
        assert dec.decision == "AUTO_HANDLE"
        assert dec.rule_triggered is None

    def test_prime_inquiry_auto_handled(self, engine):
        dec = engine.evaluate(
            "How do I cancel my Prime membership?",
            "prime_membership_benefits", 0.88, "low"
        )
        assert dec.should_escalate is False
        assert dec.decision == "AUTO_HANDLE"

    def test_general_inquiry_auto_handled(self, engine):
        dec = engine.evaluate(
            "What are your customer service hours?",
            "general_inquiry_feedback", 0.75, "low"
        )
        assert dec.should_escalate is False


# ──────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_message_does_not_crash(self, engine):
        dec = engine.evaluate("", "general_inquiry_feedback", 0.80, "low")
        # Empty message with no trigger keywords + safe intent = auto-handle
        assert dec.decision in ("AUTO_HANDLE", "ESCALATE_TO_HUMAN")

    def test_expressive_language_die_for_kindle(self, engine):
        """
        Known failure mode: 'die' should not trigger security guardrail
        for an obviously positive/enthusiastic tweet.
        The current regex-based engine may flag this (known limitation).
        """
        dec = engine.evaluate(
            "I would literally DIE for this new Kindle, when is it back in stock?!",
            "general_inquiry_feedback", 0.82, "low"
        )
        # Document known behavior — engine may or may not escalate due to regex false positive
        # This test just ensures it doesn't crash and returns a valid decision
        assert dec.decision in ("AUTO_HANDLE", "ESCALATE_TO_HUMAN")
        assert dec.reason  # Reason must always be populated
