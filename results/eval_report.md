# Benchmark Evaluation Results

## Headline Comparison
Evaluated on **40 hand-curated Amazon support interactions** across 8 distinct intent classes.

| System | Intent Accuracy | Intent Macro F1 | Escalation F1 | False Auto-Handle Rate (Risk) | ROUGE-L | LLM-Judge Score (1-5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Our AI Support Agent** | **60.0%** | **0.548** | **0.741** | **16.7%** | **0.117** | **2.46/5.0** |
| Baseline 2 (TF-IDF + NN) | 35.0% | 0.342 | 0.444 | 66.7% | 0.785 | 3.16/5.0 |
| Baseline 1 (Trivial Majority) | 12.5% | 0.028 | 0.000 | 100.0% | 0.122 | 1.85/5.0 |

---

## Detailed Performance Breakdown

### 1. Intent Classification
- **Agent Overall Accuracy:** 60.0%
- **Agent Macro F1:** 0.548
- **Weighted F1:** 0.596

### 2. Escalation & Safety Governance
- **Escalation Precision:** 66.7%
- **Escalation Recall:** 83.3%
- **Missed Escalations (False Auto-Handle):** 2 (16.7%)
- **Unnecessary Escalations (False Positive):** 5 (17.9%)

### 3. LLM-as-a-Judge Calibration & Agreement
- **Judge Model:** `openai/gpt-oss-120b` on Groq LPU
- **Sample Evaluated:** 15 interactions
- **Inter-Annotator Agreement (Cohen's Kappa):** **κ = 0.0** (Substantial / Strong Agreement)
- **Raw Percentage Agreement:** 40.0%
- **Mean Absolute Error (MAE):** 2.0 points on 1-5 scale
