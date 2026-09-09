"""
pipeline.py
-----------
Phase 3: Unified AI Customer Support Agent Pipeline.
Coordinates the end-to-end flow:
  Incoming Tweet -> Intent Classification -> Escalation Decision -> RAG Retrieval -> Grounded Reply
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from src.agent.classifier import IntentClassifier, IntentClassificationResult
from src.agent.escalation import EscalationEngine, EscalationDecision
from src.agent.retriever import SupportKnowledgeRetriever, RetrievedResolution
from src.agent.reply_generator import ReplyGenerator, DraftReply

log = logging.getLogger(__name__)


@dataclass
class AgentTurnOutput:
    """Complete output record for a customer service turn."""
    customer_message: str
    predicted_intent: str
    intent_confidence: float
    intent_reasoning: str
    should_escalate: bool
    escalation_decision: str
    escalation_reason: str
    escalation_priority: str
    rule_triggered: Optional[str]
    reply_text: str
    reply_grounded: bool
    key_action_offered: str
    retrieved_resolutions: list
    latency_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SupportAgentPipeline:
    """The complete autonomous customer support agent."""

    def __init__(
        self,
        retriever: Optional[SupportKnowledgeRetriever] = None,
        classifier: Optional[IntentClassifier] = None,
        escalation_engine: Optional[EscalationEngine] = None,
        reply_generator: Optional[ReplyGenerator] = None,
        confidence_threshold: float = 0.55,
    ):
        log.info("Initializing SupportAgentPipeline components...")
        self.classifier = classifier or IntentClassifier()
        self.escalation_engine = escalation_engine or EscalationEngine(confidence_threshold=confidence_threshold)
        self.reply_generator = reply_generator or ReplyGenerator()
        self.retriever = retriever or SupportKnowledgeRetriever()

    def handle_message(self, message: str) -> AgentTurnOutput:
        """Processes an incoming customer message end-to-end."""
        start_time = time.time()

        # Step 1: Classify Intent
        classification: IntentClassificationResult = self.classifier.classify(message)

        # Step 2: Escalation Evaluation
        escalation: EscalationDecision = self.escalation_engine.evaluate(
            customer_message=message,
            predicted_intent=classification.intent,
            confidence=classification.confidence,
            urgency=classification.urgency,
        )

        # Step 3: Retrieve Historical Support Context
        retrieved: list[RetrievedResolution] = self.retriever.retrieve(message, top_k=3)

        # Step 4: Generate Grounded Reply
        draft: DraftReply = self.reply_generator.generate_reply(
            customer_message=message,
            predicted_intent=classification.intent,
            escalation_decision=escalation,
            retrieved_resolutions=retrieved,
        )

        latency = round(time.time() - start_time, 2)

        return AgentTurnOutput(
            customer_message=message,
            predicted_intent=classification.intent,
            intent_confidence=classification.confidence,
            intent_reasoning=classification.reasoning,
            should_escalate=escalation.should_escalate,
            escalation_decision=escalation.decision,
            escalation_reason=escalation.reason,
            escalation_priority=escalation.priority,
            rule_triggered=escalation.rule_triggered,
            reply_text=draft.reply_text,
            reply_grounded=draft.grounded_in_evidence,
            key_action_offered=draft.key_action_offered,
            retrieved_resolutions=[
                {
                    "similarity": r.similarity_score,
                    "past_query": r.customer_query,
                    "brand_reply": r.brand_reply,
                }
                for r in retrieved
            ],
            latency_seconds=latency,
        )
