"""
Tests for `recall_guarantee` and `audit_exhaustive` (1.9.10).

The "silent collapse" bug: on a weak-manifold corpus (word2vec-like), the
default `k1_fraction` prefilter cuts candidates WITHOUT a per-vector proof
(`pruned_by_prefilter`). When the pool does not contain the true global top-K,
`search()` returns a pool-top-K that differs from the exact top-K while still
reporting `bound_violations == 0` (the guarantee is per-document
bound-correctness WITHIN the pool, not global recall).

The fix makes the honest statement machine-readable:
  - `SearchResult.recall_guarantee` is "exact_global" when the post-filter
    re-scored ALL N (k3 == N), else "pool_only" (derived from observed state,
    never a promise).
  - `build_engine(audit_exhaustive=True)` forces k1 = k2 = N and disables
    early_exit, so search() returns the EXACT GLOBAL top-K and
    recall_guarantee == "exact_global". Cost: O(N·d) per query.
  - `search_exact()` always reports "exact_global".
  - `search_with_commitment()` carries the same recall_guarantee so the WORM
    record cannot be over-read as a global proof when it is pool-only.

Run:  python -m pytest tests/python/test_recall_guarantee.py -v
"""
import numpy as np
import pytest

import winnex_madhava


def _weak_manifold_corpus(n=3000, d=300, seed=123):
    """Word2vec-like: many latent dims (near-isotropic) + noise. This is the
    regime where the k1_fraction pool does NOT reliably contain the global
    top-K, so the default search() can diverge from search_exact()."""
    rng = np.random.default_rng(seed)
    ncomp = 200
    X = (rng.standard_normal((n, ncomp)) @ rng.standard_normal((ncomp, d))).astype(np.float32)
    X += 0.2 * rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.ascontiguousarray(X, dtype=np.float32)


def _build(ae=False):
    X = _weak_manifold_corpus()
    return X, winnex_madhava.build_engine(
        X[:-50], dim=300, metric="cosine", quant="none", basis="random",
        stage1_dim=64, stage2_dim=0, k=10, k1_fraction=0.2,
        normalize_input=True, early_exit=False, audit_exhaustive=ae)


def test_default_is_pool_only_when_k3_lt_n():
    """On a weak manifold with k1_fraction=0.2, the default engine does NOT
    score all N, so recall_guarantee must be 'pool_only' (honest: the result
    is the best within the pool, not proven global)."""
    X, eng = _build(ae=False)
    q = np.ascontiguousarray(X[0], dtype=np.float32)
    r = eng.search(q)
    assert r.recall_guarantee == "pool_only"
    assert r.k3 < eng.num_vectors()
    # bound_violations == 0 does NOT imply exact_global
    assert r.bound_violations == 0


def test_default_can_diverge_from_exact_on_weak_manifold():
    """Regression: the silent collapse — default search() may return a
    pool-top-K that differs from search_exact() with 0 violations. The fix does
    NOT change this behavior (backwards-compatible); it only EXPOSES it via
    recall_guarantee. Callers that need the global top-K must use
    audit_exhaustive=True or search_exact()."""
    X, eng = _build(ae=False)
    # at least one query diverges on this corpus (the trigger condition)
    diverged = 0
    for j in range(20):
        q = np.ascontiguousarray(X[j], dtype=np.float32)
        r = eng.search(q)
        rx = eng.search_exact(q)
        if r.indices != rx.indices:
            diverged += 1
            assert r.recall_guarantee == "pool_only"
            assert rx.recall_guarantee == "exact_global"
            assert r.bound_violations == 0  # the trap: viol=0 yet not global
    assert diverged > 0, "test corpus no longer triggers the weak-manifold case"


def test_audit_exhaustive_guarantees_global_exact():
    """audit_exhaustive=True forces k1=k2=N and disables early_exit: search()
    returns the EXACT GLOBAL top-K (search() == search_exact()) with
    recall_guarantee == 'exact_global' and k3 == N."""
    X, eng = _build(ae=True)
    n = eng.num_vectors()
    for j in range(20):
        q = np.ascontiguousarray(X[j], dtype=np.float32)
        r = eng.search(q)
        rx = eng.search_exact(q)
        assert r.indices == rx.indices
        assert r.recall_guarantee == "exact_global"
        assert r.k3 == n
        assert r.bound_violations == 0


def test_search_exact_always_exact_global():
    """search_exact() scans the whole corpus (k3 == N): recall_guarantee is
    always 'exact_global'."""
    X, eng = _build(ae=False)
    for j in range(5):
        q = np.ascontiguousarray(X[j], dtype=np.float32)
        rx = eng.search_exact(q)
        assert rx.recall_guarantee == "exact_global"
        assert rx.k3 == eng.num_vectors()


def test_commitment_carries_recall_scope():
    """O commitment NÃO pode ser lido além do que prova.

    MUDANÇA DE CONTRATO (2026-10-30) — este teste afirmava
    `c_pool == "pool_only"`, e essa era a assinatura de um defeito, não de uma
    garantia: o commitment pagava um scan exato O(N·d) só para achar o
    threshold global, e descartava o top-K exato que a MESMA passada tinha
    acabado de produzir. Medido (20k x 384d, s1=64/s2=128, 40 queries): o
    caminho auditado devolvia o top-K exato em 1 de 8 queries, recall 0.5083 —
    enquanto segurava a prova de que a resposta era 1.0.

    Agora a passada entrega threshold E top-K, então o commitment é
    `exact_global` por `scored_all`. O teste passou a exigir o que importa:

      1. o commitment devolve o MESMO top-K que search_exact();
      2. o escopo declarado corresponde ao que foi de fato escaneado;
      3. `pool_only` continua possível — mas só quando o escaneamento global
         NÃO aconteceu, que é o caso do caminho simples (gated).
    """
    Xq, eng_pool = _build(ae=False)
    _, eng_glob = _build(ae=True)
    q = np.ascontiguousarray(Xq[0], dtype=np.float32)
    c_pool = eng_pool.search_with_commitment(q, k=10, max_sample=5)
    c_glob = eng_glob.search_with_commitment(q, k=10, max_sample=5)

    # 1. o caminho auditado devolve o top-K exato (era 0.5083 antes)
    exato = set(int(i) for i in eng_pool.search_exact(q).indices)
    assert set(int(i) for i in c_pool["indices"]) == exato, (
        "o commitment devolveu um top-K diferente do exato")
    assert set(int(i) for i in c_glob["indices"]) == exato

    # 2. o escopo declarado corresponde ao escaneamento
    assert c_glob["recall_guarantee"] == "exact_global"
    if c_pool.get("exact_scored", 0) >= eng_pool.num_vectors():
        assert c_pool["recall_guarantee"] == "exact_global"
    else:
        assert c_pool["recall_guarantee"] == "pool_only"

    # 3. o caminho SIMPLES, gated, continua honestamente pool_only
    r = eng_pool.search(q)
    assert r.collect_audit_used is False
    assert r.recall_guarantee == "pool_only"


def test_default_matches_previous_behavior_on_clean_manifold():
    """Backwards compatibility: on a STRONG manifold (the common case), the
    default engine already returns the exact top-K — recall_guarantee may be
    'exact_global' if k1>=N, but the RESULTS are unchanged from 1.9.9."""
    rng = np.random.default_rng(7)
    ncomp = 16
    N, D = 3000, 300
    X = (rng.standard_normal((N, ncomp)) @ rng.standard_normal((ncomp, D))).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = np.ascontiguousarray(X, dtype=np.float32)
    eng = winnex_madhava.build_engine(
        X[:-50], dim=D, metric="cosine", quant="none", basis="random",
        stage1_dim=64, stage2_dim=0, k=10, k1_fraction=0.05,
        normalize_input=True, early_exit=False)
    same = 0
    for j in range(20):
        q = np.ascontiguousarray(X[j], dtype=np.float32)
        same += int(eng.search(q).indices == eng.search_exact(q).indices)
    # strong manifold: even with k1=0.05 the pool contains the top-K (the
    # bound orders well), so search() matches exact in most/all queries.
    assert same >= 18
