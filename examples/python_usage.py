"""
python_usage.py — using winnex-madhava from Python.

Install once:
    pip install winnex-madhava

Run:
    python examples/python_usage.py
"""
import numpy as np

import winnex_madhava

# 1. Build an engine over a small synthetic corpus.
rng = np.random.default_rng(42)
corpus = rng.integers(0, 256, size=(50_000, 128), dtype=np.uint8)

engine = winnex_madhava.build_engine(
    corpus,
    dim=128,
    stage1_dim=64,
    k=10,
    k1_fraction=0.05,
    postfilter=True,
)
print(f"Built {engine.num_vectors()} vectors ({engine.dim()}D) in {engine.build_seconds():.2f}s")

# 2. Search.
query = corpus[0].astype(np.float32)  # the first vector -> itself at rank 0
res = engine.search(query)
print(f"\nTop-{len(res.indices)} (k1={res.k1}, latency={res.latency_ms:.2f}ms):")
for i, idx in enumerate(res.indices):
    l2 = float(((corpus[idx].astype(np.float32) - query) ** 2).sum())
    print(f"  {i+1}. id={idx}  L2²={l2:.0f}")
print(f"Bound violations: {res.bound_violations}")

# 3. Exact-scan baseline (the recall ceiling of the subset).
exact = engine.search_exact(query)
print(f"\nExact-scan top-1: id={exact.indices[0]} (the query itself)")

# 4. Metrics against a ground-truth list.
print(f"\nPerfect result metrics: R@10={winnex_madhava.recall_at_k(res.indices, res.indices, 10):.2f}")

# 5. Batch evaluation vs ground truth.
gt = [res.indices]
out = winnex_madhava.benchmark_vs_groundtruth(
    engine, query.reshape(1, -1), gt, query_alignment=1, k=10
)
print(f"Benchmark vs GT: R@10={out['recall_at_k']:.3f} NDCG={out['ndcg_at_k']:.3f}")
