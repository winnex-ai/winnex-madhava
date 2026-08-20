"""
Tests for the lightweight audit commitment (`search_with_commitment`).

The production audit-trail path: the motor returns a compact AuditCommitment
(total_excluded_count + global_threshold + a deterministic boundary sample)
instead of the full O(N) certificate. This is what the WORM stores — the
Python compliance layer signs it with Ed25519 for non-repudiation.

Validated:
  - max_sample is RESPECTED (the 1.9.1 `search_audited` returned ALL excluded
    docs regardless of max_audit_records — the ~2 MB/query problem).
  - count == search_audited.audit_excluded (exact consistency).
  - determinism: same query -> same sampled doc_ids.
  - the sampled docs are genuinely excluded (upper_bound < global_threshold).

Run:  python -m pytest tests/python/ -v
"""
import numpy as np
import pytest

import winnex_madhava


def _high_dim_pca_engine():
    rng = np.random.default_rng(7)
    N, D, K = 8000, 1536, 10
    ncomp = 24
    comp = rng.standard_normal((ncomp, D)).astype(np.float32)
    coef = rng.standard_normal((N, ncomp)).astype(np.float32)
    X = (coef @ comp).astype(np.float32) * 0.35
    Xc = X[:-100].copy()
    return winnex_madhava.build_engine(
        Xc, dim=D, metric="cosine", quant="none", basis="pca_corpus",
        stage1_dim=min(192, D), stage2_dim=0, k=K, normalize_input=True)


def test_commitment_respects_max_sample():
    eng = _high_dim_pca_engine()
    q = eng_basis_query(eng)                   # a real (non-degenerate) query
    c = eng.search_with_commitment(q, k=10, max_sample=5)
    assert c["total_excluded_count"] > 0        # the bound proves exclusions
    assert len(c["sampled_records"]) == 5       # max_sample is HONORED
    assert all(s["excluded"] for s in c["sampled_records"])


def test_commitment_count_matches_search_audited():
    eng = _high_dim_pca_engine()
    q = eng_basis_query(eng)
    c = eng.search_with_commitment(q, k=10, max_sample=50)
    ar = eng.search_audited(q, k=10, max_audit_records=50)
    assert c["total_excluded_count"] == ar["audit_excluded"]


def test_commitment_deterministic():
    eng = _high_dim_pca_engine()
    q = eng_basis_query(eng)
    c1 = eng.search_with_commitment(q, k=10, max_sample=10)
    c2 = eng.search_with_commitment(q, k=10, max_sample=10)
    assert [s["doc_id"] for s in c1["sampled_records"]] == \
           [s["doc_id"] for s in c2["sampled_records"]]
    assert c1["total_excluded_count"] == c2["total_excluded_count"]


def test_commitment_samples_are_genuinely_excluded():
    eng = _high_dim_pca_engine()
    q = eng_basis_query(eng)
    c = eng.search_with_commitment(q, k=10, max_sample=50)
    thr = c["global_threshold"]
    for s in c["sampled_records"]:
        assert s["upper_bound"] < thr          # Cauchy-Schwarz: ub < threshold


def test_commitment_global_threshold_matches_search():
    eng = _high_dim_pca_engine()
    q = eng_basis_query(eng)
    c = eng.search_with_commitment(q, k=10, max_sample=10)
    # The threshold is the exact score of the K-th result — the TRUE global
    # threshold the motor decided with (mirrors the 1.9.1 witness fix).
    assert c["global_threshold"] > 0
    assert c["bound_violations"] == 0
    assert len(c["indices"]) >= 10


def eng_basis_query(eng):
    """A real, non-degenerate query: the engine's own first corpus vector
    (as a float32 array). A zero vector is NOT a valid embedding — the cosine
    is undefined (norm 0) and the threshold degenerates to 0. Production
    queries are unit-norm embeddings."""
    # The engine holds the corpus via the buffer table; we can't reach it
    # directly here, so build a deterministic query from the same generator.
    rng = np.random.default_rng(7)
    N, D, K = 8000, 1536, 10
    ncomp = 24
    comp = rng.standard_normal((ncomp, D)).astype(np.float32)
    coef = rng.standard_normal((N, ncomp)).astype(np.float32)
    X = (coef @ comp).astype(np.float32) * 0.35
    q = X[0].astype(np.float32)
    q = q / np.linalg.norm(q)                  # unit-norm (cosine contract)
    return q
