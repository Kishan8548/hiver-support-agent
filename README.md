# 🛒 Amazon Support AI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Fast Inference: Groq LPU](https://img.shields.io/badge/Inference-Groq%20LPU-orange.svg)](https://groq.com)
[![Vector DB: Chroma](https://img.shields.io/badge/Vector%20DB-ChromaDB-green.svg)](https://www.trychroma.com/)

An autonomous AI customer support system for **Amazon (@AmazonHelp)** built on the real-world **Twitter Customer Support dataset** (Kaggle). 

The system classifies customer intents, drafts empathetic grounded replies using historical support resolutions (RAG), and enforces an enterprise-grade safety escalation policy — accompanied by an automated evaluation harness and LLM-as-a-Judge benchmark.

---

## ⚡ Reproduce Headline Results in Under 15 Minutes

Follow these 4 quick steps to install, run, and reproduce the headline benchmark results:

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Kishan8548/hiver-support-agent.git
cd hiver-support-agent

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Free API Key
Copy the example environment file and insert your free **Groq API key** ([console.groq.com](https://console.groq.com)):
```bash
cp .env.example .env
```
Open `.env` and set:
```ini
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run a Single Test Tweet or Interactive Console
Test the agent instantly on any customer tweet:
```bash
python main.py --message "Where is my order #102-9481? It was supposed to arrive yesterday!"
```
Or launch the live interactive shell:
```bash
python main.py --interactive
```

### 4. Reproduce Headline Evaluation Benchmark
Run the master evaluation harness against the **200-sample Golden Evaluation Set** (comparing our AI Agent vs. 2 Baselines with LLM Judge):
```bash
python eval/run_eval.py --golden eval/golden_set.jsonl --samples 30
```
Outputs are automatically written to `results/eval_report.md` and `results/eval_report.json`.

---

## 📊 Headline Benchmark Results

Evaluated across **200 hand-curated Amazon support interactions** spanning 8 distinct operational intents:

| System | Intent Accuracy | Intent Macro F1 | Escalation F1 | False Auto-Handle Rate (Safety Risk) | ROUGE-L | LLM-Judge Score (1-5) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Our AI Support Agent** | **94.0%** | **0.938** | **0.952** | **3.8%** | **0.286** | **4.62 / 5.0** |
| Baseline 2 (TF-IDF + NN) | 68.5% | 0.641 | 0.720 | 26.9% | 0.201 | 2.94 / 5.0 |
| Baseline 1 (Trivial Majority) | 12.5% | 0.028 | 0.000 | 100.0% | 0.118 | 1.85 / 5.0 |

* **Human-Judge Calibration:** Cohen's Kappa $\kappa = \mathbf{0.812}$ (near-perfect agreement between LLM Judge and human auditors).
* **Critical Risk Reduction:** False Auto-Handle Rate dropped from **26.9% to 3.8%**, preventing bot responses to furious or hacked customers.

---

## 🏗️ System Architecture

```
                       [Incoming Customer Tweet]
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │   1. Intent Classification (Groq LPU)     │
             │   Model: openai/gpt-oss-120b              │
             │   Output: Intent, Confidence, Reasoning   │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │   2. Hybrid Escalation Engine             │
             │   • Deterministic Security/Fraud Regex    │
             │   • Legal / Regulatory Threat Triggers    │
             │   • Repeated Agent Failure Heuristics     │
             │   • Confidence Gating (< 0.55 Threshold)  │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │   3. Semantic RAG Knowledge Retriever     │
             │   • ChromaDB Persistent Vector Store      │
             │   • Embeddings: all-MiniLM-L6-v2 (Local)  │
             │   • Top-k Historical Resolved Cases       │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │   4. Grounded Reply Generator             │
             │   • Twitter ≤280 Character Compliance     │
             │   • Brand Voice & Empathy Conditioning    │
             │   • No-Public-PII Governance Guarantee    │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
                [Structured Turn Output & Routing]
```

---

## 📁 Repository Structure

```
hiver-support-agent/
├── README.md                     # Quickstart & reproduction guide (<15 min)
├── REPORT.md                     # Technical report (problem framing, failure analysis, etc.)
├── DECISION_LOG.md               # 14 non-obvious engineering decisions & rationale
├── requirements.txt              # All dependencies (free-tier & local)
├── .env.example                  # Environment variable configuration template
├── main.py                       # CLI for single-query, interactive shell, and indexing
├── src/
│   ├── data_pipeline.py          # Extracts & reconstructs 164k Amazon threads from Kaggle
│   ├── intents.py                # 8-intent taxonomy schema with policy definitions
│   ├── agent/
│   │   ├── classifier.py         # Structured Groq intent classifier with Pydantic validation
│   │   ├── escalation.py         # Multi-tiered rule + LLM escalation engine
│   │   ├── retriever.py          # Persistent ChromaDB vector knowledge base
│   │   ├── reply_generator.py    # RAG grounded reply generator in Amazon brand voice
│   │   └── pipeline.py           # Unified agent pipeline coordinating all modules
│   ├── baselines/
│   │   ├── majority_baseline.py  # Baseline 1: Trivial majority-class agent
│   │   └── tfidf_baseline.py     # Baseline 2: Classical TF-IDF + nearest-neighbor agent
│   └── evaluation/
│       ├── metrics.py            # Intent F1, Escalation Risk, and ROUGE grounding metrics
│       └── llm_judge.py          # 4-dimension LLM judge with Cohen's Kappa calibration
├── eval/
│   ├── golden_set.jsonl          # 200 hand-curated ground truth interactions
│   ├── golden_set_creation_notes.md # Sampling & annotation methodology
│   └── run_eval.py               # Master benchmark runner comparing agent vs. baselines
├── scripts/
│   ├── cluster_topics.py         # TF-IDF + KMeans clustering discovering natural intents
│   ├── quick_eda.py              # Rapid inspection script for languages and thread lengths
│   └── create_golden_set.py      # Stratified dataset sampling and ground-truth pairing
├── data/                         # twcs.csv and reconstructed JSONL files (gitignored)
└── results/
    ├── eval_report.md            # Auto-generated markdown benchmark report
    └── eval_report.json          # Machine-readable evaluation metrics
```

---

## 📚 Citations & Attributions

* **Dataset:** *Customer Support on Twitter* hosted on Kaggle by Thought Vector (`thoughtvector/customer-support-on-twitter`).
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` by Hugging Face / UKPLab (Wang et al., 2020).
* **Vector Store:** ChromaDB (`chromadb`) open-source vector database.
* **LLM Inference:** Groq Cloud LPU API (`openai/gpt-oss-120b`).
* **Metrics:** Scikit-Learn (Pedregosa et al., 2011) for Cohen's Kappa, Precision, Recall, and Macro-F1; Google Research `rouge-score` for ROUGE-L.
