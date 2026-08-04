#!/usr/bin/env python3
"""
winnex-madhava batch benchmark — measure recall against the exact-scan ceiling.

    pip install winnex-madhava
    python examples/batch_benchmark.py

For each query, compares the engine's top-K against a brute-force L2 scan
over the *same* corpus and reports recall@K. The brute-force result is the
"recall ceiling" — the best any index could do on this subset.
"""
import numpy as np
import winnex_madhava


def exact_l2_topk(corpus_float: np.ndarray, q: np.ndarray, k: int) -> list:
    """Brute-force top-K by exact L2² over the corpus."""
    diff = corpus_float - q
    l2 = np.einsum("ij,ij->i", diff, diff)
    return np.argsort(l2)[:k].tolist()


def main() -> None:
    rng = np.random.default_rng(7)
    n, d, k = 10_000, 64, 10
    corpus = rng.integers(0, 256, size=(n, d), dtype=np.uint8)
    corpus_float = corpus.astype(np.float32)

    engine = winnex_madhava.build_engine(corpus, dim=d, k=k)
    nq = 100

    recalls, top1_hits = [], 0
    for i in range(nq):
        q = corpus[i].astype(np.float32)
        ann = engine.search(q).indices
        gt = exact_l2_topk(corpus_float, q, k)
        recalls.append(winnex_madhava.recall_at_k(ann, gt, k))
        top1_hits += int(ann[0] == gt[0])

    mean = float(np.mean(recalls))
    print(f"corpus: {n} x {d}D uint8 | queries: {nq} | k={k}")
    print(f"mean recall@{k} vs exact scan : {mean:.4f}")
    print(f"top-1 accuracy              : {top1_hits}/{nq}")
    print(f"bound violations            : {engine.search(corpus[0].astype(np.float32)).bound_violations}")


if __name__ == "__main__":
    main()
