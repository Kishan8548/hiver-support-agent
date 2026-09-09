"""
test_classifier.py
------------------
Unit tests for the Intent Classifier.
Uses mocking to avoid real API calls, testing the schema validation,
JSON parsing resilience, and Pydantic field validators.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from src.agent.classifier import IntentClassifier, IntentClassificationResult
from src.intents import INTENT_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_response(content: str):
    """Creates a mock Groq API response object."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ──────────────────────────────────────────────────────────────────────────────
# Schema & Pydantic Validation Tests (no API calls needed)
# ──────────────────────────────────────────────────────────────────────────────

class TestIntentClassificationResultSchema:
    def test_all_valid_intents_accepted(self):
        for intent in INTENT_NAMES:
            r = IntentClassificationResult(
                intent=intent,
                confidence=0.9,
                reasoning="Test",
                urgency="medium",
            )
            assert r.intent == intent

    def test_invalid_intent_falls_back_to_general(self):
        r = IntentClassificationResult(
            intent="completely_unknown_intent_xyz",
            confidence=0.5,
            reasoning="Test",
            urgency="low",
        )
        assert r.intent == "general_inquiry_feedback"

    def test_partial_match_intent_resolves(self):
        """'delivery' should resolve to 'delivery_delay_missing'."""
        r = IntentClassificationResult(
            intent="delivery",
            confidence=0.7,
            reasoning="Partial match test",
            urgency="medium",
        )
        assert r.intent == "delivery_delay_missing"

    def test_confidence_clamped_valid_range(self):
        r = IntentClassificationResult(
            intent="refund_return",
            confidence=0.75,
            reasoning="Valid",
            urgency="low",
        )
        assert 0.0 <= r.confidence <= 1.0

    def test_invalid_urgency_falls_back_to_medium(self):
        r = IntentClassificationResult(
            intent="general_inquiry_feedback",
            confidence=0.8,
            reasoning="Test",
            urgency="URGENT_NOW",
        )
        assert r.urgency == "medium"

    def test_valid_urgency_levels(self):
        for level in ("low", "medium", "high"):
            r = IntentClassificationResult(
                intent="general_inquiry_feedback",
                confidence=0.8,
                reasoning="Test",
                urgency=level,
            )
            assert r.urgency == level

    def test_uppercase_intent_normalized(self):
        r = IntentClassificationResult(
            intent="REFUND_RETURN",
            confidence=0.85,
            reasoning="Test",
            urgency="medium",
        )
        assert r.intent == "refund_return"


# ──────────────────────────────────────────────────────────────────────────────
# Classifier Mock Tests (Groq API mocked — no network calls)
# ──────────────────────────────────────────────────────────────────────────────

class TestIntentClassifierMocked:
    @pytest.fixture
    def classifier(self):
        with patch("src.agent.classifier.Groq") as MockGroq:
            instance = MockGroq.return_value
            yield IntentClassifier(api_key="test-key-123"), instance

    def test_clean_json_response_parsed(self, classifier):
        clf, mock_client = classifier
        payload = {
            "intent": "delivery_delay_missing",
            "confidence": 0.94,
            "reasoning": "Customer mentions missing package.",
            "urgency": "medium",
        }
        mock_client.chat.completions.create.return_value = make_mock_response(
            json.dumps(payload)
        )
        result = clf.classify("My package hasn't arrived yet!")
        assert result.intent == "delivery_delay_missing"
        assert result.confidence == 0.94

    def test_markdown_fenced_json_recovered(self, classifier):
        """Model occasionally wraps JSON in ```json``` fences — must recover."""
        clf, mock_client = classifier
        payload = {
            "intent": "refund_return",
            "confidence": 0.88,
            "reasoning": "Customer wants refund.",
            "urgency": "low",
        }
        fenced = f"```json\n{json.dumps(payload)}\n```"
        mock_client.chat.completions.create.return_value = make_mock_response(fenced)
        result = clf.classify("I want to return this broken product.")
        assert result.intent == "refund_return"

    def test_empty_message_returns_general(self, classifier):
        clf, _ = classifier
        result = clf.classify("   ")
        assert result.intent == "general_inquiry_feedback"
        assert result.confidence == 1.0

    def test_security_related_message_classified(self, classifier):
        clf, mock_client = classifier
        payload = {
            "intent": "account_security_access",
            "confidence": 0.97,
            "reasoning": "Unauthorized access mentioned.",
            "urgency": "high",
        }
        mock_client.chat.completions.create.return_value = make_mock_response(
            json.dumps(payload)
        )
        result = clf.classify("Someone hacked my Amazon account!")
        assert result.intent == "account_security_access"
        assert result.urgency == "high"


# ──────────────────────────────────────────────────────────────────────────────
# Metrics Unit Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_intent_metrics(self):
        from src.evaluation.metrics import compute_intent_metrics
        y_true = ["delivery_delay_missing", "refund_return", "account_security_access"]
        y_pred = ["delivery_delay_missing", "refund_return", "account_security_access"]
        m = compute_intent_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_worst_intent_metrics(self):
        from src.evaluation.metrics import compute_intent_metrics
        y_true = ["delivery_delay_missing", "delivery_delay_missing"]
        y_pred = ["refund_return", "refund_return"]
        m = compute_intent_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_perfect_escalation_metrics(self):
        from src.evaluation.metrics import compute_escalation_metrics
        y_true = [True, False, True, False]
        y_pred = [True, False, True, False]
        m = compute_escalation_metrics(y_true, y_pred)
        assert m["f1"] == 1.0
        assert m["false_auto_handle_rate"] == 0.0

    def test_zero_false_auto_handles_critical(self):
        """No missed escalations = 0.0 false auto-handle rate."""
        from src.evaluation.metrics import compute_escalation_metrics
        y_true = [True, True, False]
        y_pred = [True, True, False]
        m = compute_escalation_metrics(y_true, y_pred)
        assert m["false_auto_handle_rate"] == 0.0
        assert m["false_negatives_missed_escalate"] == 0

    def test_rouge_computation(self):
        from src.evaluation.metrics import compute_generation_metrics
        hyps = ["We are sorry your order is delayed. Please check Your Orders."]
        refs = ["Sorry about the delay! Check your orders page for updates."]
        m = compute_generation_metrics(hyps, refs)
        assert "rougeL" in m
        assert 0.0 <= m["rougeL"] <= 1.0

    def test_length_compliance_rate(self):
        from src.evaluation.metrics import compute_generation_metrics
        short = ["Short reply"]
        long = ["x" * 300]  # Exceeds 280 chars
        m_short = compute_generation_metrics(short, short)
        m_long = compute_generation_metrics(long, long)
        assert m_short["length_compliance_rate"] == 1.0
        assert m_long["length_compliance_rate"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Cohen's Kappa Calibration Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHumanJudgeAgreement:
    def test_perfect_agreement_kappa_one(self):
        from src.evaluation.llm_judge import compute_human_judge_agreement
        ratings = [5, 4, 3, 5, 4]
        result = compute_human_judge_agreement(ratings, ratings)
        assert result["cohen_kappa"] == 1.0
        assert result["raw_percentage_agreement"] == 1.0
        assert result["mean_absolute_error"] == 0.0

    def test_total_disagreement_kappa_negative(self):
        from src.evaluation.llm_judge import compute_human_judge_agreement
        humans = [5, 5, 5, 5]
        judges = [1, 1, 1, 1]
        result = compute_human_judge_agreement(humans, judges)
        # Binary: humans all pass (>=4), judges all fail (<4) -> kappa = -1
        assert result["cohen_kappa"] <= 0.0

    def test_sample_size_returned(self):
        from src.evaluation.llm_judge import compute_human_judge_agreement
        result = compute_human_judge_agreement([4, 3, 5], [4, 4, 5])
        assert result["sample_size"] == 3

    def test_mismatched_lengths_raise_error(self):
        from src.evaluation.llm_judge import compute_human_judge_agreement
        with pytest.raises(AssertionError):
            compute_human_judge_agreement([4, 5], [4])
