# Golden Evaluation Set Documentation

## Overview
- **Total Hand-Curated Samples:** 200
- **Source:** Real customer interactions from the Twitter Customer Support dataset (`@AmazonHelp`).
- **Target Distribution:** 25 balanced samples per intent across all 8 operational categories.

## Sampling Methodology
1. **Source Corpus:** Filtered from 164,328 historical multi-turn threads involving `@AmazonHelp`.
2. **Stratified Intent Sampling:** Queries were matched against verified lexical and contextual intent patterns, then manually audited to remove ambiguous fragments, gibberish, or non-English inquiries.
3. **Ground Truth Pairing:** Each customer query preserves the authentic historical Twitter reply provided by Amazon's support staff, serving as the high-quality human reference for generation metrics (ROUGE-L, BERTScore, and LLM-as-Judge grounding).
4. **Escalation Ground Truth Assignment:** 
   - Policy-restricted categories (`account_security_access`, `service_complaint_escalation`) are strictly marked as `should_escalate = true`.
   - Security/fraud triggers and threats of legal action (`sue`, `lawyer`, `BBB`) are marked as `should_escalate = true`.
   - Standard transactional inquiries with clear self-serve paths are marked as `should_escalate = false`.

## Distribution Breakdown

### Intent Distribution
| Intent | Count | Proportion |
| :--- | :--- | :--- |
| `account_security_access` | 25 | 12.5% |
| `delivery_delay_missing` | 25 | 12.5% |
| `general_inquiry_feedback` | 25 | 12.5% |
| `order_cancellation_change` | 25 | 12.5% |
| `prime_membership_benefits` | 25 | 12.5% |
| `product_technical_support` | 25 | 12.5% |
| `refund_return` | 25 | 12.5% |
| `service_complaint_escalation` | 25 | 12.5% |

### Escalation Distribution
- **Auto-Handle:** 144 (72.0%)
- **Escalate to Human:** 56 (28.0%)

## Quality & Edge Cases Resolved
- **Ambiguous Queries:** Messages expressing both a delayed package and frustration were classified under `service_complaint_escalation` if the tone was predominantly hostile or cited repeated agent failures, prioritizing customer retention.
- **Multilingual Queries:** Non-English queries in the dataset (French/German/Spanish) were excluded to maintain clean benchmark consistency.
- **PII Protection:** All references to personal account numbers or phone numbers were sanitized.
