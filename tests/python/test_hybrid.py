"""
Correctness tests for the hybrid (MadHybrid) mode of winnex-madhava.

These validate the MATH and CORRECTNESS of the clustered engine — not
performance. They check:
  1. float32 and uint8 corpus paths both build and search.
  2. On structured (clustered) data, hybrid reaches high recall vs exact.
  3. Global index mapping is correct (cell-local ids -> corpus ids).
  4. The default (full-scan) engine is unchanged and still correct.

Run:  python -m pytest tests/python/ -v
"""
import numpy as np
import pytest

import winnex_madhava


def _clustered_corpus(n=20_000, d=64, n_clusters=16, seed=7):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, d).astype(np.float32) * 5.0
    labels = rng.randint(0, n_clusters, n)
    X = centers[labels] + rng.randn(n, d).astype(np.float32) * 0.5
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


@pytest.fixture()
def structured_float32():
    return _clustered_corpus()


@pytest.fixture()
def structured_uint8(structured_float32):
    # Quantize to [0,255] for the uint8 (L2) path.
    X = structured_float32
    Xq = (X - X.min()) / (X.max() - X.min()) * 255
    return Xq.astype(np.uint8)


def test_hybrid_accepts_float32(structured_float32):
    """hybrid=True must accept float32 embeddings (cosine) and return a MadHybrid."""
    X = structured_float32
    eng = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, hybrid=True, nlist=16, nprobe=5, metric="cosine",
    )
    # API surface
    assert hasattr(eng, "search")
    assert hasattr(eng, "search_exact")
    assert eng.num_vectors() == X.shape[0]
    assert eng.dim() == X.shape[1]
    assert eng.built() is True


def test_hybrid_accepts_uint8(structured_uint8):
    """hybrid=True must accept uint8 raw bytes (L2) and return a MadHybrid."""
    X = structured_uint8
    eng = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, hybrid=True, nlist=16, nprobe=5, metric="l2",
    )
    assert eng.num_vectors() == X.shape[0]
    res = eng.search(X[0].astype(np.float32), k=10)
    assert len(res.indices) == 10
    assert all(0 <= int(i) < X.shape[0] for i in res.indices)


def test_hybrid_recall_high_on_clustered_float32(structured_float32):
    """On structured data, hybrid with enough nprobe reaches high recall vs exact."""
    X = structured_float32
    eng = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, hybrid=True, nlist=16, nprobe=8, metric="cosine",
    )
    rng = np.random.RandomState(0)
    qs = rng.choice(X.shape[0], 50, replace=False)
    recalls = []
    for qi in qs:
        q = X[qi].astype(np.float32)
        res = eng.search(q, k=10)
        exact = eng.search_exact(q, k=10)
        exact_set = set(int(i) for i in exact.indices[:10])
        hits = sum(1 for i in res.indices[:10] if int(i) in exact_set)
        recalls.append(hits / 10)
    assert np.mean(recalls) >= 0.8, f"hybrid recall too low: {np.mean(recalls):.3f}"


def test_hybrid_global_index_mapping(structured_float32):
    """Cell-local ids must map back to the correct GLOBAL corpus ids."""
    X = structured_float32
    eng = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, hybrid=True, nlist=16, nprobe=8, metric="cosine",
    )
    q = X[0].astype(np.float32)
    res = eng.search(q, k=10)
    # Every returned id must be a valid corpus index.
    assert all(0 <= int(i) < X.shape[0] for i in res.indices)
    # The query vector itself must appear in the top results on clustered data.
    assert 0 in res.indices, "query (index 0) should be found in hybrid top-k"


def test_hybrid_search_exact_is_brute_force(structured_float32):
    """search_exact on hybrid must equal numpy brute-force cosine."""
    X = structured_float32
    eng = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, hybrid=True, nlist=16, nprobe=8, metric="cosine",
    )
    q = X[0].astype(np.float32)
    res = eng.search_exact(q, k=10)
    scores = X.astype(np.float32) @ q
    expected = np.argsort(-scores)[:10]
    assert list(res.indices) == list(expected)


def test_default_mode_unchanged(structured_uint8):
    """default (hybrid=False) still returns MadhavaL2 with 0 violations."""
    X = structured_uint8
    eng = winnex_madhava.build_engine(X, dim=X.shape[1], k=10, metric="l2")
    assert type(eng).__name__ == "MadhavaL2"
    q = X[0].astype(np.float32)
    res = eng.search(q)
    assert res.bound_violations == 0


def test_hybrid_nprobe_tradeoff(structured_float32):
    """More nprobe must not reduce recall (monotone-ish), and must not crash."""
    X = structured_float32
    q = X[0].astype(np.float32)
    exact = set()
    eng_exact = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, hybrid=True, nlist=16, nprobe=16, metric="cosine",
    )
    exact = set(int(i) for i in eng_exact.search_exact(q, k=10).indices[:10])
    eng3 = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, hybrid=True, nlist=16, nprobe=3, metric="cosine",
    )
    r3 = set(int(i) for i in eng3.search(q, k=10).indices[:10])
    eng16 = winnex_madhava.build_engine(
        X, dim=X.shape[1], k=10, hybrid=True, nlist=16, nprobe=16, metric="cosine",
    )
    r16 = set(int(i) for i in eng16.search(q, k=10).indices[:10])
    # More nprobe must recover at least as many exact neighbors.
    assert len(r16 & exact) >= len(r3 & exact)
