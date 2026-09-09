"""
main.py
-------
CLI entry point for the Amazon AI Customer Support Agent.
Supports interactive mode, single-message evaluation, and batch execution.

Usage:
  python main.py --index                      # Index historical support cases into ChromaDB
  python main.py --message "Where is my item?" # Run agent on a single tweet
  python main.py --interactive                # Live test console
"""

import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

from src.agent.pipeline import SupportAgentPipeline
from src.agent.retriever import SupportKnowledgeRetriever


def print_agent_output(out):
    """Pretty prints the full turn results."""
    print("=" * 65)
    print(" 🛒 AMAZON SUPPORT AI AGENT")
    print("=" * 65)
    print(f"📩 CUSTOMER: \"{out.customer_message}\"")
    print("-" * 65)
    print(f"🏷️  INTENT:       {out.predicted_intent.upper()} (Confidence: {out.intent_confidence:.2f})")
    print(f"💡 REASONING:    {out.intent_reasoning}")
    print("-" * 65)
    escalate_badge = "🚨 ESCALATE TO HUMAN" if out.should_escalate else "✅ AUTO-HANDLE"
    print(f"🚦 ROUTING:      {escalate_badge} [Priority: {out.escalation_priority.upper()}]")
    print(f"📋 POLICY NOTE:  {out.escalation_reason}")
    if out.rule_triggered:
        print(f"⚠️  TRIGGER:      {out.rule_triggered}")
    print("-" * 65)
    print(f"💬 DRAFT REPLY:\n   \"{out.reply_text}\"")
    print("-" * 65)
    print(f"⏱️  LATENCY:      {out.latency_seconds}s")
    if out.retrieved_resolutions:
        print(f"📚 TOP GROUNDING HISTORICAL CASE (Sim: {out.retrieved_resolutions[0]['similarity']:.2f}):")
        print(f"   Past Inquiry: \"{out.retrieved_resolutions[0]['past_query'][:80]}...\"")
        print(f"   Amazon Solved: \"{out.retrieved_resolutions[0]['brand_reply'][:80]}...\"")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Hiver SDE Intern - Amazon Support Agent CLI")
    parser.add_argument("--message", "-m", type=str, help="Process a single incoming customer tweet")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive CLI loop")
    parser.add_argument("--index", action="store_true", help="Build/update ChromaDB vector index from dataset")
    parser.add_argument("--index-size", type=int, default=5000, help="Number of cases to index (default: 5000)")
    args = parser.parse_args()

    retriever = SupportKnowledgeRetriever()

    if args.index:
        print(f"Building vector knowledge base from historical Amazon threads (max: {args.index_size:,})...")
        count = retriever.build_index_from_threads(max_records=args.index_size)
        print(f"✅ Vector index ready with {count:,} historical Amazon resolutions.")
        return

    # Check if index exists, if not index a starter set of 2000 records
    if retriever.count() < 100:
        print("Knowledge base is empty. Building starter index of 2,000 historical Amazon resolutions...")
        retriever.build_index_from_threads(max_records=2000)

    pipeline = SupportAgentPipeline(retriever=retriever)

    if args.message:
        out = pipeline.handle_message(args.message)
        print_agent_output(out)
        return

    if args.interactive:
        print("\n=== Amazon AI Support Agent Interactive Shell (Type 'quit' or 'exit' to stop) ===")
        while True:
            try:
                msg = input("\nCustomer tweet > ").strip()
                if not msg:
                    continue
                if msg.lower() in ("quit", "exit", "q"):
                    break
                out = pipeline.handle_message(msg)
                print_agent_output(out)
            except KeyboardInterrupt:
                break
        print("\nSession ended.")
        return

    # Default if no arguments provided: run a demonstration test case
    print("No arguments provided. Running demonstration test scenario...")
    demo_tweet = "My order was supposed to arrive yesterday for my son's birthday party and it's not here! Tracking has not updated in 48 hours."
    out = pipeline.handle_message(demo_tweet)
    print_agent_output(out)


if __name__ == "__main__":
    main()
