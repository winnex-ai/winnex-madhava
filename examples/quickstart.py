#!/usr/bin/env python3
"""
winnex-madhava quickstart — build, search, and see the guarantee.

    pip install winnex-madhava
    python examples/quickstart.py

Prints:
  * how many vectors were indexed and how fast
  * the top-10 results for a query
  * the bound-violation counter (always 0 — the mathematical guarantee)
"""
import numpy as np
import winnex_madhava


def main() -> None:
    # 1. Build an engine over a uint8 corpus (shape (n, dim), values 0-255).
    corpus = np.random.randint(0, 256, size=(100_000, 128), dtype=np.uint8)
    engine = winnex_madhava.build_engine(corpus, dim=128, k=10)
    print(f"indexed {engine.num_vectors()} vectors in {engine.build_seconds():.2f}s")

    # 2. Search with a float32 query (the float32 *of the uint8 values*).
    query = corpus[0].astype(np.float32)
    result = engine.search(query)

    # 3. Inspect the result.
    print("top-10 dataset ids:", result.indices)
    print("latency:", round(result.latency_ms, 3), "ms")
    print("bound violations:", result.bound_violations, "(always 0)")
    print("stage-1 survivors (k1):", result.k1)
    print("post-filtered survivors (k3):", result.k3)

    # 4. Compare against the exact scan (the recall ceiling of this corpus).
    exact = engine.search_exact(query)
    print("top-1 matches exact scan:", result.indices[0] == exact.indices[0])
    print("recall@10 vs exact:", round(
        winnex_madhava.recall_at_k(result.indices, exact.indices, 10), 3))


if __name__ == "__main__":
    main()
