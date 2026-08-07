"""
Correctness tests for the speed (GPU) mode of winnex-madhava.

These validate the MATH and CORRECTNESS of the GPU QK^T scan — not
performance. They check:
  1. build_engine(..., speed=True) returns a MadhavaSpeed.
  2. The GPU search equals numpy brute-force (cosine and L2).
  3. search_batch equals the per-query search.
  4. The L2 correction (||v||² term) is correct.

Skipped automatically when torch/CUDA is unavailable.
Run:  python -m pytest tests/python/ -v
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("speed mode requires CUDA", allow_module_level=True)

import winnex_madhava


def _clustered_corpus(n=20_000, d=64, n_clusters=16, seed=7):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, d).astype(np.float32) * 5.0
    labels = rng.randint(0, n_clusters, n)
    X = centers[labels] + rng.randn(n, d).astype(np.float32) * 0.5
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def test_build_engine_speed_returns_madhava_speed():
    X = _clustered_corpus(n=10_000)
    eng = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, metric="cosine", speed=True,
    )
    assert type(eng).__name__ == "MadhavaSpeed"
    assert eng.num_vectors() == X.shape[0]
    assert eng.built() is True


def test_speed_cosine_matches_brute_force():
    X = _clustered_corpus(n=20_000)
    eng = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, metric="cosine", speed=True,
    )
    q = X[0].astype(np.float32)
    res = eng.search(q, k=10)
    # Cosine: the engine normalizes the query. Normalize for the brute-force.
    qn = q / max(np.linalg.norm(q), 1e-12)
    scores = X.astype(np.float32) @ qn
    expected = np.argsort(-scores)[:10]
    assert list(res.indices) == list(expected)


def test_speed_l2_matches_brute_force():
    """L2 mode must use ||v||² + ||q||² - 2·<v,q>, not bare inner product."""
    X = _clustered_corpus(n=20_000)
    # uint8 corpus for the L2 (BIGANN-style) path
    Xu = (X - X.min()) / (X.max() - X.min()) * 255
    Xu = Xu.astype(np.uint8)
    eng = winnex_madhava.build_engine(
        Xu, dim=Xu.shape[1], k=10, metric="l2", speed=True,
    )
    q = Xu[0].astype(np.float32)
    res = eng.search(q, k=10)
    d2 = ((Xu.astype(np.float32) - q) ** 2).sum(axis=1)
    expected = np.argsort(d2)[:10]
    assert list(res.indices) == list(expected)


def test_speed_batch_equals_individual():
    X = _clustered_corpus(n=20_000)
    eng = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, metric="cosine", speed=True,
    )
    Q = X[:50].astype(np.float32)
    rb = eng.search_batch(Q, k=10, chunk=25)
    idx_batch = np.array(rb.indices).reshape(50, 10)
    for i in range(10):
        res = eng.search(Q[i], k=10)
        # Allow exact-tie reordering (identical scores can appear in any order
        # in torch.topk) — compare as sets.
        assert set(res.indices) == set(idx_batch[i]), f"batch/indiv mismatch at {i}"


def test_speed_exact_is_scan():
    """search_exact on speed returns the same as search (it IS exact)."""
    X = _clustered_corpus(n=10_000)
    eng = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, metric="cosine", speed=True,
    )
    q = X[0].astype(np.float32)
    assert list(eng.search(q, k=10).indices) == list(eng.search_exact(q, k=10).indices)
