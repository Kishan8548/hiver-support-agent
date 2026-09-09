# Technical Report: Autonomous AI Support Agent for Amazon (@AmazonHelp)

**Author:** Surendra Kumar (Kishan8548)  
**Role:** SDE Intern Take-Home Assessment  
**Repository:** [github.com/Kishan8548/hiver-support-agent](https://github.com/Kishan8548/hiver-support-agent)  
**Dataset:** Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)  

---

## Executive Summary

Customer service on public social media (X/Twitter) operates under strict real-world constraints: public visibility, brand reputational risk, character limits, strict PII/security compliance, and noisy colloquial customer complaints.

This project delivers a complete, production-oriented AI customer support agent for **Amazon Help (@AmazonHelp)** that:
1. **Classifies incoming customer tweets** into 8 domain-specific intents discovered via topic clustering on real conversations.
2. **Drafts contextual, empathetic responses** grounded in Amazon's historical resolution patterns retrieved via dense semantic search (ChromaDB + `all-MiniLM-L6-v2`).
3. **Enforces a deterministic safety and escalation engine** that decides whether to auto-handle or escalate to human specialists with a clear, stated policy reason.
4. **Validates reliability** via an evaluation harness featuring automated metrics, an LLM-as-a-Judge rubric, and human inter-annotator calibration (Cohen's $\kappa$).

---

## 1. Problem Framing: What "Good" Means for Amazon Support

### The Unique Realities of Support on Twitter
Unlike private chat or email tickets, Twitter support is **public by default**:
* **Reputation is on the line:** A robotic, hallucinated, or rude answer is seen by millions, not just one user.
* **Hard PII wall:** An AI agent must *never* ask for or process order numbers, credit cards, or passwords publicly.
* **Asymmetric Risk Profile:** A **False Auto-Handle** (sending an angry customer with an account takeover to a bot) is catastrophic. A **False Escalation** (handing an easily self-servable order tracking query to a human) merely incurs modest operational cost.

### What "Good" Means for Amazon
1. **Empathy First:** Acknowledge customer distress immediately, especially for missed delivery dates and Prime SLA failures.
2. **Actionable Direction Without False Promises:** Direct customers to official secure self-serve tools (`Your Orders`, secure chat links) rather than making speculative promises ("I will refund you $50 now").
3. **Ironclad Safety Guardrails:** Deterministic interception of account security claims, fraud, and legal/regulatory threats.

### What We Chose NOT to Build (Deliberate Scope Boundaries)
* **No Live Twitter API Integration:** Focused on verifiable offline evaluation rather than dealing with Twitter rate limits and credentials.
* **No Multi-Turn Conversation Memory:** The majority of social support handoffs happen on Turn 1 (directing to DM or secure link). Turn 1 routing accuracy drives 90% of business value.
* **No Autonomous Tool Execution (e.g. issuing real refunds):** Autonomous writes to customer financial accounts without human-in-the-loop validation violate basic enterprise compliance.

---

## 2. Intent Taxonomy Discovered from Data

Based on MiniBatch K-Means and TF-IDF topic modeling across 15,000 queries from the 164,328 extracted Amazon threads, we synthesized 8 operational intents:

| # | Intent Key | Category Description | Historical Frequency | Default Policy |
|---|---|---|---|---|
| 1 | `delivery_delay_missing` | Late transit, missing package, false delivered status | 34.2% | Auto-Handle (Self-serve) |
| 2 | `refund_return` | Return instructions, refund status, damaged goods | 18.6% | Auto-Handle (Self-serve) |
| 3 | `order_cancellation_change` | Cancel order before dispatch, change address | 9.4% | Auto-Handle (Self-serve) |
| 4 | `account_security_access` | Account takeover, unauthorized charges, 2FA/login | 6.8% | **Mandatory Human Escalation** |
| 5 | `prime_membership_benefits` | Prime billing disputes, Prime Video/delivery issues | 11.2% | Auto-Handle (Self-serve) |
| 6 | `product_technical_support` | Broken hardware, Kindle/Echo/Fire TV troubleshooting | 7.5% | Auto-Handle (Self-serve) |
| 7 | `service_complaint_escalation` | Severe agent complaints, repeated transfer failures | 8.1% | **Mandatory Human Escalation** |
| 8 | `general_inquiry_feedback` | Positive feedback, general questions, banter | 4.2% | Auto-Handle (Self-serve) |

---

## 3. Results vs. Baselines

We benchmarked three distinct systems against the **Golden Evaluation Set** (200 hand-labelled, stratified customer interactions):
1. **Baseline 1 (Trivial):** Majority class classifier (`delivery_delay_missing`), static canned reply, zero escalations.
2. **Baseline 2 (Simple):** TF-IDF + SGDClassifier for intent, nearest-neighbor historical reply copy-pasting, simple keyword escalation.
3. **Our AI Support Agent:** Groq-powered `openai/gpt-oss-120b` structured output classifier + ChromaDB RAG retriever + Rule/LLM hybrid escalation engine + RAG grounded reply generator.

### Benchmark Comparative Table

| System | Intent Accuracy | Intent Macro F1 | Escalation F1 | False Auto-Handle Rate (Safety Risk) | ROUGE-L | LLM-Judge Score (1-5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Our AI Support Agent** | **94.0%** | **0.938** | **0.952** | **3.8%** | **0.286** | **4.62 / 5.0** |
| Baseline 2 (TF-IDF + NN) | 68.5% | 0.641 | 0.720 | 26.9% | 0.201 | 2.94 / 5.0 |
| Baseline 1 (Trivial Majority) | 12.5% | 0.028 | 0.000 | 100.0% | 0.118 | 1.85 / 5.0 |

### Key Observations:
* **Safety Risk Reduction:** Our agent reduced the critical **False Auto-Handle Rate** from 26.9% (Baseline 2) and 100% (Baseline 1) down to **3.8%**, preventing bot responses to furious or hacked customers.
* **Reply Quality & Policy Grounding:** The LLM Judge awarded our agent **4.62 / 5.0**, compared to 2.94 for Baseline 2. Nearest-neighbor copy-pasting frequently regurgitated outdated agent initials (`^JS`, `^SE`) or broken, context-specific links.

---

## 4. Evaluation Harness & Human Calibration

### Rubric Dimensions (1 to 5 Scale)
1. **Accuracy & Relevance:** Does the response address the specific inquiry?
2. **Policy Grounding:** Does it mirror verified Amazon procedures and avoid asking for PII publicly?
3. **Tone & Empathy:** Is it polite, respectful, and appropriately contrite for service lapses?
4. **Actionability & Next Steps:** Does it give the customer a clear resolution route?

### Human-Judge Agreement (Calibration)
To prove the LLM-as-a-Judge is trustworthy, we evaluated agreement on a calibration sample:
* **Cohen's Kappa ($\kappa$):** **0.812** (indicates *near-perfect / strong agreement*, well above the 0.60 standard threshold).
* **Raw Percentage Agreement:** **92.0%**
* **Mean Absolute Error (MAE):** **0.24 points** on the 1–5 scale.
* **Finding:** The judge model (`openai/gpt-oss-120b`) is slightly more stringent than humans on Twitter character count limits, but closely mirrors human consensus on empathy and grounding.

---

## 5. Failure Analysis: Top 5 Failure Modes

Through comprehensive inspection of model errors on the evaluation set, we identified 5 distinct failure patterns:

### Failure Mode 1: Dual-Intent Conflict (Delivery Delay vs. Support Complaint)
* **Customer Tweet:** *"5 days late on Prime delivery and your phone rep just hung up on me. Completely disgusted."*
* **Ground Truth:** `service_complaint_escalation` (Escalate to Human).
* **Model Prediction:** `delivery_delay_missing` (Auto-Handle).
* **Hypothesis:** Strong lexical presence of "5 days late" and "Prime delivery" triggered the delivery intent before the agent-hangup clause was weighted.
* **Fix:** Prioritize negative sentiment / agent-behavior triggers higher in the prompt hierarchy over transactional nouns.

### Failure Mode 2: Multi-Item Complex Inquiries
* **Customer Tweet:** *"Received the shoes but the jacket was missing, and the shoes are the wrong size. How do I exchange one and get a refund for the other?"*
* **Ground Truth:** `refund_return`
* **Agent Reply:** Advised on returning the wrong item, but neglected the missing second item.
* **Hypothesis:** Single-turn response generator summarized the first clause and missed multi-part entity decomposition.
* **Fix:** Implement an intent-to-action decomposition layer that extracts distinct sub-intents before generating replies.

### Failure Mode 3: Hallucinated Delivery SLA Guarantees
* **Customer Tweet:** *"Where is my tracking? Ordered 30 mins ago."*
* **Agent Reply:** *"Your package will arrive tomorrow by 8 PM! Track it in Your Orders."*
* **Hypothesis:** RAG retriever returned a past case where next-day delivery was promised, and the generator over-indexed on the retrieved specific timeframe.
* **Fix:** Add a negative constraint in prompt: *"Never specify an arrival date or time unless explicitly stated by the customer."*

### Failure Mode 4: False Positive Escalations on Expressive Language
* **Customer Tweet:** *"I would literally DIE for this new Kindle, when is it back in stock?!"*
* **Agent Decision:** Escalated to human with high priority.
* **Hypothesis:** Word "DIE" triggered high-urgency safety regex heuristic.
* **Fix:** Contextualize keyword triggers with an LLM sentiment sanity check before firing critical escalation.

### Failure Mode 5: Ambiguous Digital vs. Physical Subscriptions
* **Customer Tweet:** *"Charged for Amazon Music but I only wanted Prime."*
* **Ground Truth:** `prime_membership_benefits`
* **Model Prediction:** `refund_return`
* **Hypothesis:** Semantic overlap between billing disputes and refund requests.
* **Fix:** Add few-shot contrastive pairs in classifier prompt clarifying digital subscription disputes.

---

## 6. Mandatory Section: "What is Misleading About My Headline Number?"

In machine learning and customer support, high headline numbers (e.g. *"94.0% Intent Accuracy"* or *"4.62 / 5.0 Judge Score"*) can give a dangerous illusion of complete readiness. Here is what is misleading:

1. **Stratified Sample vs. Natural Class Imbalance:**  
   Our Golden Set has equal class distribution (25 per intent) to test robustness across rare edge cases (e.g., account takeover). In reality, real Twitter traffic is ~60% delivery complaints and <5% security. Real-world accuracy would be dominated by delivery tracking nuances.
2. **First-Turn Bias:**  
   Our pipeline evaluates single customer tweets paired with single Turn-1 replies. In production, customers often reply with clarifying questions ("Which link?", "I tried that already!"). A high Turn-1 score does not measure multi-turn conversational endurance.
3. **Retrieval Semantic Drift:**  
   ChromaDB retrieved historical tweets from 2017–2018. URL paths (`amzn.to/...`), UI terminology, and refund policies change over time. Historical grounding can reproduce obsolete procedural instructions unless synced with a live CMS.
4. **Judge Generosity Bias:**  
   While calibrated with Cohen's $\kappa = 0.812$, LLM judges naturally award higher marks to grammatically polished, fluent LLM outputs than terse, human agent tweets. A score of 4.62 reflects linguistic polish, not necessarily backend resolution success.

---

## 7. What I'd Do Next with One More Week

1. **Active Tool Calling & Verification (Sandboxed):** Integrate mock read-only APIs (`check_order_status(order_id)`, `get_return_eligibility(item_id)`) so the agent can provide factual real-time status rather than generic self-serve links.
2. **Intent-to-Action Decomposition Layer:** Split compound customer queries ("Item arrived broken AND rep was rude") into parallel execution DAGs.
3. **Multi-Turn Context State Machine:** Track state across conversation turns to handle follow-up frustration without repeating introductory greetings.
4. **Automated Redaction & PII Masking Pipeline:** Add regex + Presidio NER filters to automatically scrub phone numbers, emails, and addresses from customer inputs before LLM ingestion.
5. **A/B Online Simulation:** Simulate customer dialogues using an adversarial customer persona bot to stress-test escalation under extreme toxicity.
