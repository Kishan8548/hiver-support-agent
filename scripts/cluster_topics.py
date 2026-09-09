"""
cluster_topics.py
Perform TF-IDF + KMeans clustering on 15,000 Amazon customer queries
to identify natural intent clusters in the dataset.
"""
import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

threads_file = "data/amazon_threads.jsonl"
texts = []

with open(threads_file, "r", encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        c = t["customer_message"]
        # Basic English filter
        if sum(1 for ch in c if ord(ch) < 128) / max(len(c), 1) > 0.85:
            if len(c) > 20:
                texts.append(c)
        if len(texts) >= 15000:
            break

print(f"Clustering {len(texts):,} customer queries...")

vectorizer = TfidfVectorizer(
    max_features=2500,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
)
X = vectorizer.fit_transform(texts)

k = 8
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=500)
kmeans.fit(X)

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

cluster_counts = np.bincount(kmeans.labels_)

print("\n=== TOPIC CLUSTERS IN AMAZON SUPPORT DATA ===")
for i in range(k):
    top_terms = [terms[ind] for ind in order_centroids[i, :8]]
    pct = cluster_counts[i] / len(texts) * 100
    print(f"\nCluster {i+1} ({pct:.1f}% of queries): {', '.join(top_terms)}")
