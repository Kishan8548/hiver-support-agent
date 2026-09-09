"""
quick_eda.py
Inspect languages, length, and top terms in the extracted Amazon data.
"""
import json
import sys
from collections import Counter
import re

# Set stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

threads_file = "data/amazon_threads.jsonl"

total = 0
english_like = 0
non_ascii_chars = 0
sample_english = []

with open(threads_file, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        t = json.loads(line)
        c_text = t["customer_message"]
        
        # Simple ASCII ratio check
        ascii_count = sum(1 for c in c_text if ord(c) < 128)
        ratio = ascii_count / max(len(c_text), 1)
        
        if ratio > 0.85:
            english_like += 1
            if len(sample_english) < 15:
                sample_english.append((t["customer_message"], t["brand_reply"]))

print(f"Total threads: {total:,}")
print(f"English-like threads: {english_like:,} ({english_like/total*100:.1f}%)")
print("\n--- 10 Sample Customer Messages & Brand Replies ---")
for i, (cust, brand) in enumerate(sample_english[:10], 1):
    print(f"\n[{i}] CUSTOMER: {cust}")
    print(f"    AMAZON:   {brand}")
