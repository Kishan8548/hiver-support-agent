"""
metrics.py
----------
Phase 5: Automated Evaluation Metrics Suite.
Computes quantitative performance across all three agent tasks:
  1. Intent Classification (Accuracy, Macro/Weighted F1, Precision, Recall)
  2. Escalation Routing (Precision, Recall, F1, False Auto-Handle Rate)
  3. Reply Quality (ROUGE-1, ROUGE-2, ROUGE-L, Length compliance)
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from rouge_score import rouge_scorer


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Computes comprehensive intent classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "per_class": {
            k: {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1": round(v["f1-score"], 4),
                "support": v["support"],
            }
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
    }


def compute_escalation_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    """
    Computes escalation decision performance.
    In support routing, False Auto-Handle (False Negative) is the most critical risk,
    as sending a security/angry customer to a bot leads to severe churn/exposure.
    """
    y_true_arr = np.array(y_true, dtype=bool)
    y_pred_arr = np.array(y_pred, dtype=bool)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[False, True]).ravel()

    accuracy = (tp + tn) / max(len(y_true_arr), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)

    # Risk metrics:
    false_auto_handle_rate = fn / max(tp + fn, 1)  # Missed escalations
    false_escalation_rate = fp / max(tn + fp, 1)   # Unnecessary human handoffs

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives_unnecessary_escalate": int(fp),
        "false_negatives_missed_escalate": int(fn),
        "false_auto_handle_rate": round(false_auto_handle_rate, 4),
        "false_escalation_rate": round(false_escalation_rate, 4),
    }


def compute_generation_metrics(
    hypotheses: List[str], references: List[str]
) -> Dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, ROUGE-L surface grounding metrics."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    r1_scores, r2_scores, rl_scores = [], [], []
    length_violations = 0

    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
        if len(hyp) > 280:
            length_violations += 1

    total = max(len(hypotheses), 1)
    return {
        "rouge1": round(float(np.mean(r1_scores)), 4),
        "rouge2": round(float(np.mean(r2_scores)), 4),
        "rougeL": round(float(np.mean(rl_scores)), 4),
        "length_compliance_rate": round((total - length_violations) / total, 4),
        "avg_length_chars": round(float(np.mean([len(h) for h in hypotheses])), 1),
    }
