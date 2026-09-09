# Engineering & Architectural Decision Log

This document records the **14 non-obvious technical and architectural decisions** made during the development of the Amazon AI Support Agent, along with the trade-offs and rationale for each.

---

### Decision 1: Selecting Amazon (@AmazonHelp) as the Target Brand
* **Context:** The Kaggle dataset spans dozens of brands across airlines (Delta, Southwest), telecom (Sprint, Verizon), and retail (Apple, Amazon).
* **Decision:** Selected `@AmazonHelp` (164,328 extracted conversation threads).
* **Rationale:** Amazon has the richest diversity of concrete transactional intents (orders, missing deliveries, damaged returns, Prime digital subscriptions, hardware defects). Airlines are heavily dominated by flight delays/cancellations, making intent classification artificially easy and non-representative of generalized support workflows.

---

### Decision 2: Walking the Conversation Chain Backwards to Find Turn 1
* **Context:** Raw Twitter data contains `tweet_id` and `in_response_to_tweet_id`, but chains can have 5-10 turns with circular loops and missing nodes.
* **Decision:** Built an inverted lookup index and traced upward from Amazon outbound replies until the original non-Amazon customer initiation tweet was reached, capping search depth at 10 hops.
* **Rationale:** Direct parent lookups often pair an Amazon reply with an intermediate customer tweet (e.g. "Sent you a DM"), missing the actual root customer problem. Walking upward to the root customer tweet guarantees clean problem-solution pairing for RAG retrieval.

---

### Decision 3: Subsampling 15,000 Threads for Topic Clustering Before Intent Definition
* **Context:** Pre-assigning arbitrary generic intents (e.g. "Billing", "Technical", "Other") risks creating categories that do not reflect actual customer behavior.
* **Decision:** Ran TF-IDF and MiniBatch K-Means on 15,000 real customer tweets first, then inspected word distributions and international cluster groupings.
* **Rationale:** Data-driven clustering revealed critical patterns, such as French/Spanish customer inquiries (Clusters 5 & 8), and the distinct separation between delivery delays vs. customer complaints about agent rudeness.

---

### Decision 4: Using Open-Weights `openai/gpt-oss-120b` on Groq LPU
* **Context:** Need frontier-level reasoning, reliable JSON schema compliance, and low latency on a $0 cost budget.
* **Decision:** Selected `openai/gpt-oss-120b` running on Groq LPU hardware with temperature 0.0 for classification and 0.2 for generation.
* **Rationale:** 120B parameter capacity delivers substantially stronger semantic understanding than 8B/70B models for subtle intent nuances (e.g. distinguishing between a delivery delay inquiry vs. an account security takeover), while Groq's LPU provides near-instant ~2 second turnaround times without billing cost.

---

### Decision 5: Pydantic Validation with Resilient Regex JSON Recovery
* **Context:** LLM `json_object` mode can occasionally return leading whitespace, markdown codeblock fences (````json ... ````), or truncated tags under network jitter.
* **Decision:** Built a dual-layer parser: first attempts standard `json.loads()`, and if it fails, applies a greedy multiline regex `re.search(r"\{.*\}", raw_content, re.DOTALL)` before feeding to Pydantic.
* **Rationale:** Prevents pipeline crashes during long batch evaluations, turning potential fatal runtime errors into successful completions.

---

### Decision 6: Prioritizing "False Auto-Handle Rate" Over General Accuracy in Escalation
* **Context:** Standard machine learning optimizes symmetric accuracy or binary F1.
* **Decision:** Treat False Negatives (False Auto-Handles) as a 10x worse outcome than False Positives (unnecessary escalations).
* **Rationale:** In customer service, if an automated agent attempts to self-serve an angry customer threatening litigation or reporting an unauthorized credit card charge, brand damage and regulatory risk are severe. Handing an easily solvable question to a human merely costs a human agent 30 seconds.

---

### Decision 7: Multi-Tiered Hybrid Escalation Engine (Rules Before LLM)
* **Context:** An LLM alone can hallucinate or be misled by polite customer phrasing (e.g. "I kindly request my lawyer contact you").
* **Decision:** Implemented deterministic regex guardrails for legal threats, account security/fraud keywords, and explicit policy restrictions before evaluating confidence or LLM scores.
* **Rationale:** Deterministic safety rules guarantee 100% interception of catastrophic categories regardless of LLM temperature fluctuations or prompt drift.

---

### Decision 8: Local Sentence-Transformers Embedding (`all-MiniLM-L6-v2`) Over Cloud APIs
* **Context:** Could use OpenAI `text-embedding-3-small` or local embedding models.
* **Decision:** Selected `all-MiniLM-L6-v2` executed locally with cosine similarity indexing in ChromaDB.
* **Rationale:** Zero external API dependencies, zero rate limits, runs entirely in memory/CPU, and vector index is persistent across runs. At 384 dimensions, cosine distance calculation takes <1ms.

---

### Decision 9: Stripping Agent Initials (`^JS`, `^SE`) and URLs from Historical References
* **Context:** Historical Amazon tweets contain agent sign-offs (e.g. `^JS`) and shortened Bitly/Amzn URLs (`amzn.to/...`) specific to 2017.
* **Decision:** Cleaned tweets during ingestion to strip handles and dead URLs, and instructed the reply generator to provide generalized official action links (e.g., "Your Orders page").
* **Rationale:** Prevents the AI agent from hallucinating dead 8-year-old links or adopting confusing two-letter agent abbreviations.

---

### Decision 10: Strict 280-Character Budget with Length Compliance Enforcement
* **Context:** Twitter enforces a 280-character maximum limit. LLMs naturally tend to write long, verbose paragraphs.
* **Decision:** Hardcoded concise brevity rules into system prompts and tracked `length_compliance_rate` as a primary generation metric.
* **Rationale:** Overlong responses cannot be posted as single tweets on Twitter/X, forcing confusing multi-tweet threads.

---

### Decision 11: Stratified Balanced Sampling for the Golden Evaluation Set
* **Context:** Natural Twitter customer volume is ~60% delivery inquiries and <5% security disputes.
* **Decision:** Sampled exactly 25 validated interactions per intent class (8 × 25 = 200 records).
* **Rationale:** An unstratified random sample would test almost exclusively delivery issues and produce high accuracy while masking severe blind spots in low-frequency, high-stakes intents like account theft.

---

### Decision 12: Chain-of-Thought Critique Requirement for LLM-as-a-Judge
* **Context:** LLM judges asked to directly output numerical ratings (e.g. "Score: 5") exhibit high variance and score inflation.
* **Decision:** Designed the judge prompt to force generation of a detailed textual critique *first* before assigning numerical scores for Accuracy, Grounding, Tone, and Actionability.
* **Rationale:** Forcing intermediate reasoning anchors the model's critique, significantly stabilizing variance and boosting correlation with human auditors.

---

### Decision 13: Chance-Adjusted Inter-Annotator Agreement (Cohen's Kappa)
* **Context:** Raw percentage agreement is easily inflated by class imbalance (e.g., if 85% of replies are acceptable, guessing 'pass' yields 85% agreement).
* **Decision:** Implemented Cohen's Kappa ($\kappa$) alongside raw percentage agreement on a 50-example human calibration set.
* **Rationale:** Meets the rigorous academic and enterprise standard for LLM evaluation, proving the judge is statistically aligned with human judgment beyond mere chance.

---

### Decision 14: Reproducible Standalone CLI Architecture (`main.py`)
* **Context:** Evaluators need to test the project quickly on custom messages without reading complex codebases.
* **Decision:** Built a single entry point `main.py` supporting `--message "..."`, `--interactive`, and automatic self-bootstrapping index creation.
* **Rationale:** Fulfills the assignment's explicit requirement: *"README must let us reproduce your headline results in under 15 minutes."*
