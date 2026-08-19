"""
Tests for the per-document audit certificate (`search_audited` / `audit_json`).

The audit certificate is the winnex-audit-cpp / GovAuditRecord-compatible
per-document mathematical proof, produced by the motor's own C++ math
(ub_raw / residuals / exact_score) — no reimplementation.

Run:  python -m pytest tests/python/ -v
"""
import numpy as np
import pytest

import winnex_madhava


@pytest.fixture()
def cosine_engine():
    rng = np.random.default_rng(42)
    corpus = rng.standard_normal((5000, 128)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    return winnex_madhava.build_engine(
        corpus, dim=128, metric="cosine", quant="none",
        stage1_dim=64, stage2_dim=0, k=10, normalize_input=True)


@pytest.fixture()
def l2_engine():
    rng = np.random.default_rng(42)
    corpus = rng.integers(0, 256, size=(5000, 128)).astype(np.uint8)
    return winnex_madhava.build_engine(
        corpus, dim=128, metric="l2", quant="int8",
        stage1_dim=64, stage2_dim=0, k=10, k1_fraction=0.10)


def test_audit_record_shape(cosine_engine):
    """Every audit record carries the GovAuditRecord-compatible fields."""
    query = np.zeros(128, dtype=np.float32)
    r = cosine_engine.search_audited(query, k=10, max_audit_records=500)
    required = {"doc_id", "true_cosine", "projected_cosine", "residual_norm",
                "upper_bound", "threshold", "excluded", "stage"}
    assert len(r["audit"]) > 0
    for rec in r["audit"]:
        assert required <= set(rec.keys())
    assert r["bound_violations"] == 0


def test_certificate_integrity_cosine(cosine_engine):
    """Cosine path: an excluded record must have true_cosine strictly BELOW
    the threshold (UB < worst ⇒ exact cos < threshold, by Cauchy-Schwarz)."""
    query = np.zeros(128, dtype=np.float32)
    r = cosine_engine.search_audited(query, k=10, max_audit_records=500)
    for rec in r["audit"]:
        if rec["excluded"]:
            assert rec["true_cosine"] < rec["threshold"] - 1e-9
    assert r["audit_excluded"] >= 0


def test_certificate_integrity_l2(l2_engine):
    """L2 path: an excluded doc must have exact L2² strictly ABOVE the
    threshold (the K-th best L2² — smallest is best, so a bigger distance
    provably falls outside top-K)."""
    query = np.zeros(128, dtype=np.float32)
    r = l2_engine.search_audited(query, k=10, max_audit_records=500)
    for rec in r["audit"]:
        if rec["excluded"]:
            assert rec["true_cosine"] > rec["threshold"] + 1e-9
    assert r["bound_violations"] == 0


def test_audit_covers_topk(cosine_engine):
    """The certificate always covers the returned top-K (stage='in_topk')."""
    query = np.zeros(128, dtype=np.float32)
    r = cosine_engine.search_audited(query, k=10, max_audit_records=500)
    staged = [rec["doc_id"] for rec in r["audit"] if rec["stage"] == "in_topk"]
    assert set(staged) == set(r["indices"])


def test_audit_json_valid(cosine_engine):
    """audit_json returns a parseable JSON string with the audit_trail."""
    import json
    query = np.zeros(128, dtype=np.float32)
    s = cosine_engine.audit_json(query, k=10, max_audit_records=500)
    data = json.loads(s)
    assert "audit_trail" in data
    assert "results" in data
    assert len(data["results"]) == 10


def test_certificate_deterministic(cosine_engine):
    """Same query → same certificate (seed-fixed engine)."""
    query = np.zeros(128, dtype=np.float32)
    r1 = cosine_engine.search_audited(query, k=10, max_audit_records=500)
    r2 = cosine_engine.search_audited(query, k=10, max_audit_records=500)
    assert r1["indices"] == r2["indices"]
    assert [rec["doc_id"] for rec in r1["audit"]] == [rec["doc_id"] for rec in r2["audit"]]


def test_certificate_consistent_with_pruned_by_bound(cosine_engine):
    """The certificate's excluded count is consistent with (a subset of) the
    motor's pruned_by_bound.

    Note on semantics: the motor's `pruned_by_bound` uses the K-th best exact
    score AMONG THE POST-FILTER CANDIDATES (the k1_fraction pool), which can
    be higher than the true global threshold. `search_audited` uses the GLOBAL
    threshold (the exact score of the K-th returned result). So
    audit_excluded (global) is always <= pruned_by_bound (pool threshold) —
    never equal when the pool threshold is higher. The certificate's count is
    the honest one: docs the bound proves outside the TRUE top-K."""
    query = np.zeros(128, dtype=np.float32)
    base = cosine_engine.search(query)
    r = cosine_engine.search_audited(query, k=10, max_audit_records=500)
    assert r["audit_excluded"] <= base.pruned_by_bound
    # With full coverage the certificate examines every doc; its count is the
    # number provably outside the TRUE global top-K, a lower bound of the
    # pool-threshold count.
    n = cosine_engine.num_vectors()
    r_full = cosine_engine.search_audited(query, k=10, max_audit_records=n)
    assert r_full["audit_excluded"] <= base.pruned_by_bound
    assert r_full["audit_candidates"] == n  # whole corpus examined
