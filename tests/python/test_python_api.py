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


def test_build_and_search(engine):
    assert engine.num_vectors() == 5000
    assert engine.dim() == 64

    q = np.zeros(64, dtype=np.float32)
    res = engine.search(q)
    assert len(res.indices) == 10
    assert res.bound_violations == 0
    assert res.k1 > 0
    assert res.latency_ms >= 0


def test_self_is_top1(engine):
    # A vector equal to the query must be found at rank 0 by exact L2.
    # We use the first corpus vector as the query.
    corpus = np.zeros((1, 64), dtype=np.uint8)
    rng = np.random.default_rng(7)
    corpus[0] = rng.integers(0, 256, size=(64,), dtype=np.uint8)
    q = corpus[0].astype(np.float32)

    cfg = winnex_madhava.Config()
    cfg.dim = 64
    e2 = winnex_madhava.MadhavaL2(cfg)
    e2.build_numpy(corpus)
    res = e2.search(q)
    assert res.indices[0] == 0


def test_exact_scan_matches_brute_force(engine):
    rng = np.random.default_rng(42)
    corpus = rng.integers(0, 256, size=(5000, 64), dtype=np.uint8)
    cfg = winnex_madhava.Config()
    cfg.dim = 64
    e2 = winnex_madhava.MadhavaL2(cfg)
    e2.build_numpy(corpus)

    q = np.random.default_rng(0).integers(0, 256, size=(64,), dtype=np.uint8).astype(np.float32)
    res = e2.search_exact(q)
    # brute force with numpy
    d2 = ((corpus.astype(np.float32) - q) ** 2).sum(axis=1)
    top = np.argsort(d2)[:10]
    assert list(res.indices) == list(top)


def test_metrics():
    gt = [5, 7, 9]
    perfect = gt
    assert winnex_madhava.recall_at_k(perfect, gt, 3) == 1.0
    assert abs(winnex_madhava.ndcg_at_k(perfect, gt, 3) - 1.0) < 1e-9
    assert winnex_madhava.recall_at_k([], gt, 3) == 0.0


def test_benchmark_vs_groundtruth(engine):
    q = np.random.default_rng(1).integers(0, 256, size=(2, 64), dtype=np.uint8).astype(np.float32)
    res = engine.search(q[0])
    gt = [res.indices]
    out = winnex_madhava.benchmark_vs_groundtruth(engine, q, gt)
    assert out["n_queries"] == 1
    assert 0 <= out["recall_at_k"] <= 1.0
