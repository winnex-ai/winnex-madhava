"""
Python-level tests for the winnex-madhava package.

Run:  python -m pytest tests/python/ -v
"""
import numpy as np
import pytest

import winnex_madhava


@pytest.fixture()
def engine():
    rng = np.random.default_rng(42)
    corpus = rng.integers(0, 256, size=(5000, 64), dtype=np.uint8)
    return winnex_madhava.build_engine(corpus, dim=64, k=10, k1_fraction=0.10)


@pytest.fixture()
def l2_engine():
    """Explicit L2 engine (BIGANN-style, uint8 raw)."""
    rng = np.random.default_rng(42)
    corpus = rng.integers(0, 256, size=(5000, 64), dtype=np.uint8)
    return winnex_madhava.build_engine(corpus, dim=64, metric="l2", quant="int8",
                                        stage1_dim=64, stage2_dim=0,
                                        k=10, k1_fraction=0.10, postfilter=True)


def test_build_and_search(engine):
    assert engine.num_vectors() == 5000
    assert engine.dim() == 64

    q = np.zeros(64, dtype=np.float32)
    res = engine.search(q)
    assert len(res.indices) == 10
    assert res.bound_violations == 0
    assert res.k1 > 0
    assert res.latency_ms >= 0


def test_self_is_top1_l2(l2_engine):
    # A vector equal to the query must be found at rank 0 by exact L2.
    # We use the first corpus vector as the query.
    corpus = np.zeros((1, 64), dtype=np.uint8)
    rng = np.random.default_rng(7)
    corpus[0] = rng.integers(0, 256, size=(64,), dtype=np.uint8)
    q = corpus[0].astype(np.float32)

    cfg = winnex_madhava.Config()
    cfg.dim = 64
    cfg.metric = winnex_madhava.Metric.L2
    e2 = winnex_madhava.MadhavaL2(cfg)
    e2.build_numpy(corpus)
    res = e2.search(q)
    assert res.indices[0] == 0


def test_exact_scan_matches_brute_force_l2():
    rng = np.random.default_rng(42)
    corpus = rng.integers(0, 256, size=(5000, 64), dtype=np.uint8)
    cfg = winnex_madhava.Config()
    cfg.dim = 64
    cfg.metric = winnex_madhava.Metric.L2
    e2 = winnex_madhava.MadhavaL2(cfg)
    e2.build_numpy(corpus)

    q = np.random.default_rng(0).integers(0, 256, size=(64,), dtype=np.uint8).astype(np.float32)
    res = e2.search_exact(q)
    # brute force with numpy
    d2 = ((corpus.astype(np.float32) - q) ** 2).sum(axis=1)
    top = np.argsort(d2)[:10]
    assert list(res.indices) == list(top)


def test_cosine_cascade_self_is_top1():
    """Cosine metric + cascade [32,64] + modulation: the query vector must be
    its own top-1 (unit-normalized embeddings, stack behavior)."""
    rng = np.random.default_rng(7)
    centers = rng.normal(0, 1, size=(8, 64))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    n = 3000
    labels = rng.integers(0, 8, size=n)
    X = centers[labels] + 0.2 * rng.normal(0, 1, size=(n, 64))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xu8 = np.clip((X * 127 + 128), 0, 255).astype(np.uint8)

    eng = winnex_madhava.build_engine(Xu8, dim=64, metric="cosine", quant="int8",
                                       stage1_dim=32, stage2_dim=64, k=5,
                                       k1_fraction=0.2, modulation=True, postfilter=True)
    q = Xu8[0].astype(np.float32)
    res = eng.search(q)
    assert res.bound_violations == 0
    assert res.indices[0] == 0  # self is top-1


def test_quant_none_matches_exact():
    """quant='none' (float32 projections) must be exact-equivalent to scan."""
    rng = np.random.default_rng(11)
    X = rng.integers(0, 256, size=(3000, 64), dtype=np.uint8)
    eng = winnex_madhava.build_engine(X, dim=64, metric="l2", quant="none",
                                       stage1_dim=32, stage2_dim=0, k=5, postfilter=True)
    q = X[0].astype(np.float32)
    res = eng.search(q)
    exact = eng.search_exact(q)
    assert res.bound_violations == 0
    assert res.indices[0] == exact.indices[0] == 0


def test_metrics():
    gt = [5, 7, 9]
    perfect = gt
    assert winnex_madhava.recall_at_k(perfect, gt, 3) == 1.0
    assert abs(winnex_madhava.ndcg_at_k(perfect, gt, 3) - 1.0) < 1e-9
    assert winnex_madhava.recall_at_k([], gt, 3) == 0.0


def test_recall_robust_fewer_relevant():
    """Recall@K robusto: com |gt| < k, um resultado perfeito atinge 1.0."""
    gt = [5, 7]  # apenas 2 relevantes, k=5
    perfect = [5, 7, 100, 200, 300]
    assert winnex_madhava.recall_at_k(perfect, gt, 5) == 1.0
    assert abs(winnex_madhava.ndcg_at_k(perfect, gt, 5) - 1.0) < 1e-9


def test_recall_robust_all_relevant():
    """Intersecta com TODO o conjunto relevante, não só os top-k do GT."""
    gt = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]  # 12 relevantes
    result = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]  # acha 10, perde 25, 27
    # Definição robusta: |result[:10] ∩ gt| / 10 = 10/10 = 1.0
    assert winnex_madhava.recall_at_k(result, gt, 10) == 1.0


def test_benchmark_vs_groundtruth(engine):
    q = np.random.default_rng(1).integers(0, 256, size=(2, 64), dtype=np.uint8).astype(np.float32)
    res = engine.search(q[0])
    gt = [res.indices]
    out = winnex_madhava.benchmark_vs_groundtruth(engine, q, gt)
    assert out["n_queries"] == 1
    assert 0 <= out["recall_at_k"] <= 1.0
