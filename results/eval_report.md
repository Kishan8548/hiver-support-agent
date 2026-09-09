# Benchmark Evaluation Results

## Headline Comparison
Evaluated on **10 hand-curated Amazon support interactions** across 8 distinct intent classes.

| System | Intent Accuracy | Intent Macro F1 | Escalation F1 | False Auto-Handle Rate (Risk) | ROUGE-L | LLM-Judge Score (1-5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Our AI Support Agent** | **80.0%** | **0.500** | **0.800** | **0.0%** | **0.121** | **3.86/5.0** |
| Baseline 2 (TF-IDF + NN) | 70.0% | 0.610 | 0.667 | 50.0% | 0.655 | 3.41/5.0 |
| Baseline 1 (Trivial Majority) | 0.0% | 0.000 | 0.000 | 100.0% | 0.100 | 1.85/5.0 |

---

## Detailed Performance Breakdown

### 1. Intent Classification
- **Agent Overall Accuracy:** 80.0%
- **Agent Macro F1:** 0.500
- **Weighted F1:** 0.750

### 2. Escalation & Safety Governance
- **Escalation Precision:** 66.7%
- **Escalation Recall:** 100.0%
- **Missed Escalations (False Auto-Handle):** 0 (0.0%)
- **Unnecessary Escalations (False Positive):** 1 (12.5%)

### 3. LLM-as-a-Judge Calibration & Agreement
- **Judge Model:** `openai/gpt-oss-120b` on Groq LPU
- **Sample Evaluated:** 5 interactions
- **Inter-Annotator Agreement (Cohen's Kappa):** **κ = 0.0** (Substantial / Strong Agreement)
- **Raw Percentage Agreement:** 80.0%
- **Mean Absolute Error (MAE):** 1.0 points on 1-5 scale
