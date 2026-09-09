"""
retriever.py
------------
Phase 3: RAG Knowledge Retriever.
Indexes historical Amazon customer support threads into a persistent ChromaDB vector store.
Given an incoming customer query, retrieves top-k most semantically similar historical
cases and their verified Amazon brand resolutions to ground the response.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


@dataclass
class RetrievedResolution:
    thread_id: str
    customer_query: str
    brand_reply: str
    similarity_score: float


class SupportKnowledgeRetriever:
    """Embeds historical Amazon support pairs and retrieves nearest resolutions."""

    def __init__(
        self,
        collection_name: str = "amazon_support_resolutions",
        persist_dir: str = "chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.model_name = model_name

        log.info(f"Loading embedding model: {model_name} ...")
        self.embedder = SentenceTransformer(model_name)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Returns the number of indexed resolutions."""
        return self.collection.count()

    def build_index_from_threads(
        self,
        threads_path: str = "data/amazon_threads.jsonl",
        max_records: int = 10000,
        batch_size: int = 500,
    ) -> int:
        """
        Indexes up to `max_records` historical resolution pairs from the dataset.
        Skips if collection already contains data.
        """
        current_count = self.count()
        if current_count >= max_records:
            log.info(f"Index already populated with {current_count:,} records. Skipping re-indexing.")
            return current_count

        log.info(f"Indexing up to {max_records:,} historical resolutions from {threads_path}...")
        records = []
        with open(threads_path, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                c_msg = t["customer_message"].strip()
                b_reply = t["brand_reply"].strip()

                # Basic quality filtering: keep clean English messages of reasonable length
                if len(c_msg) > 15 and len(b_reply) > 15:
                    ascii_ratio = sum(1 for ch in c_msg if ord(ch) < 128) / len(c_msg)
                    if ascii_ratio > 0.85:
                        records.append({
                            "id": t["thread_id"],
                            "customer": c_msg,
                            "reply": b_reply,
                        })
                if len(records) >= max_records:
                    break

        log.info(f"Selected {len(records):,} high-quality threads. Generating embeddings in batches...")

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            texts = [r["customer"] for r in batch]
            ids = [f"rec_{r['id']}_{idx}" for idx, r in enumerate(batch)]
            metadatas = [
                {"customer_message": r["customer"], "brand_reply": r["reply"]}
                for r in batch
            ]
            embeddings = self.embedder.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            log.info(f"  Indexed {min(i + batch_size, len(records)):,}/{len(records):,} records")

        log.info(f"Indexing complete! Total in collection: {self.count():,}")
        return self.count()

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedResolution]:
        """
        Retrieves top_k most semantically similar historical customer inquiries
        and how Amazon support previously resolved them.
        """
        if self.count() == 0:
            log.warning("Vector collection is empty! Returning no results.")
            return []

        query_emb = self.embedder.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        retrieved = []
        if results and results["metadatas"] and len(results["metadatas"][0]) > 0:
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for meta, dist in zip(metas, distances):
                # Cosine distance in Chroma is 1 - cosine_similarity
                sim = max(0.0, min(1.0, 1.0 - dist))
                retrieved.append(
                    RetrievedResolution(
                        thread_id=meta.get("thread_id", ""),
                        customer_query=meta.get("customer_message", ""),
                        brand_reply=meta.get("brand_reply", ""),
                        similarity_score=round(sim, 3),
                    )
                )

        return retrieved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = SupportKnowledgeRetriever()
    # Populate a fast sample of 3000 records for testing
    retriever.build_index_from_threads(max_records=3000)

    test_q = "My order says it was delivered yesterday but I checked everywhere and no package!"
    hits = retriever.retrieve(test_q, top_k=3)
    print(f"\nQuery: {test_q}\n")
    print(f"Top {len(hits)} Historical Amazon Resolutions:")
    for idx, hit in enumerate(hits, 1):
        print(f"[{idx}] Similarity: {hit.similarity_score:.3f}")
        print(f"    Past Customer: {hit.customer_query}")
        print(f"    Amazon Solved: {hit.brand_reply}\n")
