"""
tfidf_baseline.py
-----------------
Phase 6: Baseline 2 (Simple Baseline).
Represents a classic machine learning approach without modern LLM reasoning:
  1. Intent Classification: TF-IDF feature extraction + Linear Classifier (SGDClassifier)
  2. Reply Generation: Nearest-neighbor copy-paste of past historical brand reply
  3. Escalation: Simple keyword thresholding heuristic
"""

import json
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors

from src.intents import INTENT_NAMES


class TfidfBaselineAgent:
    """Simple baseline using classical TF-IDF and Nearest-Neighbors retrieval."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        self.classifier = SGDClassifier(loss="log_loss", random_state=42)
        self.retrieval_nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        self.corpus_replies: List[str] = []
        self.is_fitted = False

    def fit(self, threads_path: str = "data/amazon_threads.jsonl", max_samples: int = 5000):
        """Fits TF-IDF vectorizer and nearest neighbor retriever on historical data."""
        queries, replies = [], []

        with open(threads_path, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                c_msg = t["customer_message"].strip()
                b_reply = t["brand_reply"].strip()
                if len(c_msg) > 15 and len(b_reply) > 15:
                    queries.append(c_msg)
                    replies.append(b_reply)
                if len(queries) >= max_samples:
                    break

        self.corpus_replies = replies
        X = self.vectorizer.fit_transform(queries)
        self.retrieval_nn.fit(X)

        # Train a basic classifier with dummy intent assignment using keywords
        y = []
        for q in queries:
            assigned = "general_inquiry_feedback"
            ql = q.lower()
            if "late" in ql or "where" in ql or "track" in ql:
                assigned = "delivery_delay_missing"
            elif "refund" in ql or "return" in ql:
                assigned = "refund_return"
            elif "cancel" in ql:
                assigned = "order_cancellation_change"
            elif "prime" in ql:
                assigned = "prime_membership_benefits"
            elif "hack" in ql or "password" in ql:
                assigned = "account_security_access"
            elif "worst" in ql or "rep" in ql:
                assigned = "service_complaint_escalation"
            y.append(assigned)

        self.classifier.fit(X, y)
        self.is_fitted = True

    def predict(self, customer_message: str) -> Dict[str, Any]:
        """Predicts intent, escalation, and retrieves nearest past reply."""
        if not self.is_fitted:
            self.fit()

        vec = self.vectorizer.transform([customer_message])

        # Intent prediction
        pred_intent = self.classifier.predict(vec)[0]
        try:
            probs = self.classifier.predict_proba(vec)[0]
            conf = float(max(probs))
        except Exception:
            conf = 0.60

        # Retrieval of nearest past reply (copy-paste)
        distances, indices = self.retrieval_nn.kneighbors(vec)
        nearest_idx = indices[0][0]
        raw_retrieved_reply = self.corpus_replies[nearest_idx]

        # Simple keyword-based escalation
        msg_lower = customer_message.lower()
        should_esc = any(k in msg_lower for k in ["sue", "lawyer", "bbb", "hacked", "fraud", "stolen"])
        if pred_intent in ("account_security_access", "service_complaint_escalation"):
            should_esc = True

        return {
            "predicted_intent": pred_intent,
            "intent_confidence": round(conf, 2),
            "should_escalate": should_esc,
            "escalation_decision": "ESCALATE_TO_HUMAN" if should_esc else "AUTO_HANDLE",
            "escalation_reason": "Simple keyword trigger or high-risk intent classification.",
            "reply_text": raw_retrieved_reply,
        }
