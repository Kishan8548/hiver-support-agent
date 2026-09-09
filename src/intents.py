"""
intents.py
----------
Phase 2: Intent Schema definition grounded in Amazon Help Twitter dataset.
Contains 8 distinct, actionable intent categories with descriptions,
few-shot examples, and escalation policy guidelines.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class IntentDefinition:
    name: str
    display_name: str
    description: str
    example_phrases: List[str]
    auto_handle_allowed: bool
    escalation_reason_default: str


INTENT_SCHEMA: Dict[str, IntentDefinition] = {
    "delivery_delay_missing": IntentDefinition(
        name="delivery_delay_missing",
        display_name="Delivery Delay & Missing Packages",
        description=(
            "Customer reports their package is delayed, marked delivered but missing, "
            "tracking is not updating, or courier failed to deliver."
        ),
        example_phrases=[
            "Says delivered Saturday, was not, I was home all day",
            "why is my order at my local courier for the last 6 days and still hasn't been delivered?",
            "My package hasn't arrived yet and tracking says delayed",
            "Where is my package? It was supposed to arrive by 8pm tonight",
        ],
        auto_handle_allowed=True,
        escalation_reason_default="Missing package past delivery SLA requiring courier investigation",
    ),
    "refund_return": IntentDefinition(
        name="refund_return",
        display_name="Returns & Refunds",
        description=(
            "Customer asking how to return an item, tracking an existing refund, "
            "or reporting wrong/damaged items requiring replacement or refund."
        ),
        example_phrases=[
            "my package was opened.. 4 items missing worth £97. Need a refund",
            "How do I return this damaged coffee maker I received?",
            "Returned my shoes last week, when will my refund hit my card?",
            "I received the wrong size, how can I exchange or get money back?",
        ],
        auto_handle_allowed=True,
        escalation_reason_default="High-value refund or missing item claim requiring account review",
    ),
    "order_cancellation_change": IntentDefinition(
        name="order_cancellation_change",
        display_name="Order Cancellation & Modifications",
        description=(
            "Customer requests to cancel an order, change the shipping address, "
            "or modify ordered items before dispatch."
        ),
        example_phrases=[
            "I accidentally ordered twice, please cancel order #112-3456",
            "Can I change the delivery address for my order placed 10 mins ago?",
            "Need to cancel my pre-order before it ships tomorrow",
        ],
        auto_handle_allowed=True,
        escalation_reason_default="Order already in shipping process requiring manual logistics intervention",
    ),
    "account_security_access": IntentDefinition(
        name="account_security_access",
        display_name="Account Security & Access",
        description=(
            "Customer locked out of account, 2FA/password reset problems, "
            "unauthorized charges, or requests to close account."
        ),
        example_phrases=[
            "I want my amazon payments account CLOSED. dm me please",
            "Can't log into my account, OTP is not arriving on my phone",
            "I see a charge on my credit card that I never authorized",
            "Someone hacked my account and changed the email address",
        ],
        auto_handle_allowed=False,  # Security/PII issues MUST be escalated
        escalation_reason_default="Account security or unauthorized transaction requiring identity verification",
    ),
    "prime_membership_benefits": IntentDefinition(
        name="prime_membership_benefits",
        display_name="Prime Membership & Digital Services",
        description=(
            "Questions or complaints about Amazon Prime membership fee, "
            "guaranteed two-day delivery failures for Prime members, or Prime Video streaming issues."
        ),
        example_phrases=[
            "Why am I paying for Prime if packages take 5 days to arrive?",
            "Prime video sound and subtitles are out of sync on my TV",
            "Was charged $139 for Prime renewal without notification, please cancel",
        ],
        auto_handle_allowed=True,
        escalation_reason_default="Prime subscription fee dispute or recurring service entitlement failure",
    ),
    "product_technical_support": IntentDefinition(
        name="product_technical_support",
        display_name="Product Defects & Tech Support",
        description=(
            "Defective or broken products, setup issues with Amazon devices "
            "(Echo, Kindle, Fire TV), or digital content compatibility problems."
        ),
        example_phrases=[
            "My Kindle Paperwhite screen is frozen on the tree logo",
            "Echo Dot won't connect to my new Wi-Fi router",
            "The blender stopped working after 2 uses, is there warranty?",
        ],
        auto_handle_allowed=True,
        escalation_reason_default="Hardware defect requiring warranty claim or manufacturer escalation",
    ),
    "service_complaint_escalation": IntentDefinition(
        name="service_complaint_escalation",
        display_name="Support Experience Complaints",
        description=(
            "Customer express extreme frustration with customer service, "
            "spoke to multiple agents without resolution, or threatens legal/social media action."
        ),
        example_phrases=[
            "Way to drop the ball on customer service so pissed right now!",
            "3 different reps gave me 3 different answers and lied to me",
            "Worst customer service ever, closing my account and contacting BBB",
            "Been transferred 4 times and hung up on, unacceptable!",
        ],
        auto_handle_allowed=False,  # High sentiment risk MUST be escalated
        escalation_reason_default="Severe customer dissatisfaction or repeated support breakdown",
    ),
    "general_inquiry_feedback": IntentDefinition(
        name="general_inquiry_feedback",
        display_name="General Inquiries & Feedback",
        description=(
            "General inquiries, website navigation, positive feedback, "
            "social engagement, or queries not fitting specific transactional issues."
        ),
        example_phrases=[
            "Alexa says both styles are working for you! Thanks Amazon",
            "Do you price match with other retailers during Black Friday?",
            "Just wanted to say shoutout to the driver who delivered in the snow!",
        ],
        auto_handle_allowed=True,
        escalation_reason_default="Complex non-standard policy query requiring specialist guidance",
    ),
}

INTENT_NAMES = list(INTENT_SCHEMA.keys())


def get_intent_schema_prompt_text() -> str:
    """Formats the intent schema into a concise prompt block for the LLM classifier."""
    lines = []
    for key, item in INTENT_SCHEMA.items():
        lines.append(f"- `{key}`: {item.description}")
        lines.append(f"  Examples: {'; '.join(item.example_phrases[:2])}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Total defined intents: {len(INTENT_SCHEMA)}")
    print("\nPrompt format:\n")
    print(get_intent_schema_prompt_text())
