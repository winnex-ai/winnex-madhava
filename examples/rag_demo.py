#!/usr/bin/env python3
"""
winnex-madhava RAG demo — encode texts, index, retrieve, rerank.

    pip install winnex-madhava numpy
    python examples/rag_demo.py

The demo uses a self-contained, deterministic bag-of-ngrams encoder so it runs
with zero external model downloads. To use real embeddings (e.g. SentenceTransformers,
OpenAI, or a local SBERT), just replace `encode_texts()` with your model's output and
quantize it to uint8 — the engine operates on uint8 (0-255) vectors.

Steps:
  1. chunk a small document set
  2. embed each chunk to a uint8 vector
  3. build a winnex_madhava engine
  4. retrieve the top-K chunks for a query and show the audit (0 violations)
"""
from __future__ import annotations

import re
from hashlib import sha1

import numpy as np
import winnex_madhava

DIM = 128
_HASH_BUCKET = 256  # values land in 0..255 after modulus


def _ngram_hash_features(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic bag-of-hashed-ngrams embedding -> uint8 (0-255)."""
    feats = np.zeros(dim, dtype=np.float32)
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    for tok in tokens:
        for gram_len in (1, 2, 3):
            if len(tok) < gram_len:
                continue
            for i in range(len(tok) - gram_len + 1):
                gram = tok[i : i + gram_len]
                h = int(sha1(gram.encode()).hexdigest()[:8], 16)
                feats[h % dim] += 1.0
    # normalize to 0..255 (uint8 range)
    mx = feats.max()
    if mx > 0:
        feats = feats / mx * 200.0
    return feats.astype(np.uint8)


def encode_texts(texts: list[str], dim: int = DIM) -> np.ndarray:
    return np.stack([_ngram_hash_features(t, dim) for t in texts])


def main() -> None:
    docs = [
        "Madhava provides deterministic vector search with Cauchy-Schwarz bounds.",
        "The engine proves every excluded document could not be in the top-K.",
        "HNSW builds a random proximity graph and hopes it pruned nothing relevant.",
        "Regulated industries need an auditable retrieval trail, not a gamble.",
        "The post-filter recomputes exact L2 on survivors, matching a perfect scan.",
        "BSL 1.1 licensing permits evaluation and internal production use.",
    ]
    corpus_u8 = encode_texts(docs)

    engine = winnex_madhava.build_engine(corpus_u8, dim=DIM, k=3)
    print(f"indexed {len(docs)} chunks -> {corpus_u8.shape} uint8")

    query = "Is this search provably complete for audits?"
    q_vec = encode_texts([query])[0].astype(np.float32)
    result = engine.search(q_vec)

    print("\nQuery:", query)
    for rank, idx in enumerate(result.indices, 1):
        print(f"  #{rank}  chunk {idx}: {docs[idx]}")
    print(f"\nbound_violations: {result.bound_violations}  |  latency: {result.latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
