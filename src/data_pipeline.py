"""
data_pipeline.py
----------------
Phase 1: Load, filter, reconstruct, and clean the Twitter Customer Support dataset
for the Amazon (@AmazonHelp) brand.

Usage:
    python src/data_pipeline.py --input data/twcs.csv --output-dir data/

Outputs:
    data/amazon_threads.jsonl   — full conversation threads {customer_msg, brand_reply, ...}
    data/amazon_inbound.jsonl   — customer-only messages for classification/eval
    data/pipeline_stats.json    — summary statistics
"""

import json
import re
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BRAND_HANDLE = "AmazonHelp"
# Amazon also uses these handles in the dataset
AMAZON_HANDLES = {"amazonhelp", "amazon"}


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class ConversationThread:
    thread_id: str
    customer_message: str
    brand_reply: str
    customer_handle: str
    timestamp: Optional[str]
    num_turns: int
    raw_thread: list  # list of {tweet_id, text, author_id, created_at}


@dataclass
class InboundMessage:
    tweet_id: str
    customer_handle: str
    text: str
    timestamp: Optional[str]
    in_response_to: Optional[str]
    thread_id: Optional[str]


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    """Remove @mentions, URLs, and normalize whitespace from a tweet."""
    if not isinstance(text, str):
        return ""
    # Remove @mentions (e.g. @AmazonHelp)
    text = re.sub(r"@\w+", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove HTML entities
    text = re.sub(r"&amp;|&lt;|&gt;|&quot;|&#39;", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_amazon_author(author_id: str) -> bool:
    """Check if an author ID belongs to Amazon support."""
    return str(author_id).lower() in AMAZON_HANDLES


# ── Core pipeline ─────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the Twitter Customer Support dataset CSV."""
    log.info(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(
        csv_path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "in_response_to_tweet_id": str,
            "response_tweet_id": str,
        },
        low_memory=False,
    )
    log.info(f"  Loaded {len(df):,} rows with columns: {list(df.columns)}")
    return df


def filter_amazon_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only tweets involving Amazon support (inbound + outbound)."""
    log.info("Filtering Amazon-related tweets ...")

    # Outbound: tweets FROM Amazon
    amazon_out_ids = set(
        df[df["author_id"].str.lower().isin(AMAZON_HANDLES)]["tweet_id"].astype(str)
    )

    # Inbound: tweets TO Amazon (i.e. tweet that Amazon responded to)
    # Find tweet_ids that Amazon replied to
    amazon_replies = df[df["author_id"].str.lower().isin(AMAZON_HANDLES)].copy()
    replied_to_ids = set(
        amazon_replies["in_response_to_tweet_id"].dropna().astype(str)
    )

    # Keep tweets that are either FROM amazon or were REPLIED TO by amazon
    mask = (
        df["author_id"].str.lower().isin(AMAZON_HANDLES)
        | df["tweet_id"].astype(str).isin(replied_to_ids)
    )
    amazon_df = df[mask].copy()

    log.info(
        f"  Amazon tweets: {len(amazon_df):,} "
        f"(outbound: {len(amazon_out_ids):,}, inbound: {len(replied_to_ids):,})"
    )
    return amazon_df


def build_tweet_index(df: pd.DataFrame) -> dict:
    """Build a tweet_id → row dict for fast lookup."""
    index = {}
    for _, row in df.iterrows():
        tid = str(row["tweet_id"])
        index[tid] = row.to_dict()
    return index


def reconstruct_threads(df: pd.DataFrame, tweet_index: dict) -> list[ConversationThread]:
    """
    Reconstruct conversation threads.

    Strategy: find every Amazon OUTBOUND reply and walk back up the chain
    to get the original customer message. This gives us clean
    (customer_message, brand_reply) pairs.
    """
    log.info("Reconstructing conversation threads ...")

    amazon_replies = df[df["author_id"].str.lower().isin(AMAZON_HANDLES)].copy()
    threads = []

    for _, reply_row in tqdm(amazon_replies.iterrows(), total=len(amazon_replies), desc="Threading"):
        reply_tweet_id = str(reply_row["tweet_id"])
        parent_id = str(reply_row.get("in_response_to_tweet_id", ""))

        if not parent_id or parent_id == "nan":
            continue

        # Walk up to find the first customer message in the chain
        chain = [reply_row.to_dict()]
        current_id = parent_id
        customer_msg_row = None
        hops = 0

        while current_id and current_id != "nan" and hops < 10:
            parent_row = tweet_index.get(current_id)
            if parent_row is None:
                break
            chain.insert(0, parent_row)
            if not is_amazon_author(str(parent_row.get("author_id", ""))):
                customer_msg_row = parent_row
                break
            current_id = str(parent_row.get("in_response_to_tweet_id", ""))
            hops += 1

        if customer_msg_row is None:
            continue

        customer_text = clean_tweet(str(customer_msg_row.get("text", "")))
        brand_text = clean_tweet(str(reply_row.get("text", "")))

        # Skip if either is empty after cleaning
        if not customer_text or not brand_text:
            continue
        # Skip very short messages (likely noise)
        if len(customer_text) < 10 or len(brand_text) < 5:
            continue

        thread = ConversationThread(
            thread_id=reply_tweet_id,
            customer_message=customer_text,
            brand_reply=brand_text,
            customer_handle=str(customer_msg_row.get("author_id", "unknown")),
            timestamp=str(customer_msg_row.get("created_at", None)),
            num_turns=len(chain),
            raw_thread=[
                {
                    "tweet_id": str(r.get("tweet_id", "")),
                    "author_id": str(r.get("author_id", "")),
                    "text": str(r.get("text", "")),
                }
                for r in chain
            ],
        )
        threads.append(thread)

    log.info(f"  Reconstructed {len(threads):,} threads")
    return threads


def extract_inbound_messages(threads: list[ConversationThread]) -> list[InboundMessage]:
    """Extract customer-only messages for standalone classification tasks."""
    inbound = []
    seen_texts = set()

    for thread in threads:
        text = thread.customer_message
        # Deduplicate near-identical messages
        if text in seen_texts:
            continue
        seen_texts.add(text)

        inbound.append(
            InboundMessage(
                tweet_id=thread.thread_id,
                customer_handle=thread.customer_handle,
                text=text,
                timestamp=thread.timestamp,
                in_response_to=None,
                thread_id=thread.thread_id,
            )
        )
    return inbound


# ── Writers ───────────────────────────────────────────────────────────────────
def write_jsonl(records: list, path: Path) -> None:
    """Write a list of dataclass instances to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "__dataclass_fields__"):
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"  Written {len(records):,} records → {path}")


def write_stats(threads: list, inbound: list, output_dir: Path) -> None:
    """Write pipeline statistics for reproducibility."""
    stats = {
        "total_threads": len(threads),
        "total_inbound_messages": len(inbound),
        "avg_turns_per_thread": round(
            sum(t.num_turns for t in threads) / max(len(threads), 1), 2
        ),
        "avg_customer_msg_len": round(
            sum(len(t.customer_message) for t in threads) / max(len(threads), 1), 1
        ),
        "avg_brand_reply_len": round(
            sum(len(t.brand_reply) for t in threads) / max(len(threads), 1), 1
        ),
        "unique_customers": len(set(t.customer_handle for t in threads)),
        "brand_handle": BRAND_HANDLE,
    }
    stats_path = output_dir / "pipeline_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"  Pipeline stats → {stats_path}")
    log.info(f"  Stats: {stats}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Amazon Twitter support data pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="data/twcs.csv",
        help="Path to twcs.csv (the Kaggle dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/",
        help="Directory to write output JSONL files",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Cap number of threads (useful for fast dev runs, e.g. --max-threads 5000)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    df = load_dataset(args.input)

    # Step 2: Filter to Amazon only
    amazon_df = filter_amazon_tweets(df)

    # Step 3: Build index
    tweet_index = build_tweet_index(amazon_df)

    # Step 4: Reconstruct threads
    threads = reconstruct_threads(amazon_df, tweet_index)

    # Step 5: Optional cap (for fast dev runs)
    if args.max_threads:
        threads = threads[: args.max_threads]
        log.info(f"  Capped to {args.max_threads:,} threads (--max-threads flag)")

    # Step 6: Extract inbound messages
    inbound = extract_inbound_messages(threads)

    # Step 7: Write outputs
    log.info("Writing outputs ...")
    write_jsonl(threads, output_dir / "amazon_threads.jsonl")
    write_jsonl(inbound, output_dir / "amazon_inbound.jsonl")
    write_stats(threads, inbound, output_dir)

    log.info("✅ Phase 1 complete!")


if __name__ == "__main__":
    main()
