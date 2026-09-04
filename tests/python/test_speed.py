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
import os
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


def _bound_engine_and_index(tmp_path, N=3000, d=96, s1=48, seed=42):
    """Build a real bound engine (quant='none') and return (engine, parsed arrays)."""
    rng = np.random.RandomState(seed)
    comp = rng.randn(12, d).astype(np.float32)
    X = (rng.randn(N, 12).astype(np.float32) @ comp)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    eng = winnex_madhava.build_engine(
        X, dim=d, k=10, metric="cosine", quant="none",
        stage1_dim=s1, stage2_dim=0, normalize_input=True)
    path = str(tmp_path / "bound_test.idx")
    eng.save_index(path)
    import struct
    with open(path, "rb") as f:
        f.read(8)  # magic
        D, n, s1r, s2 = struct.unpack("iiii", f.read(16))
        P1 = np.frombuffer(f.read(s1r * D * 4), dtype=np.float32).reshape(s1r, D)
        pr1 = np.frombuffer(f.read(n * s1r * 4), dtype=np.float32).reshape(n, s1r)
        f.read(s1r * 4)  # scale
        e1 = np.frombuffer(f.read(n * 4), dtype=np.float32)
        vn = np.frombuffer(f.read(n * 4), dtype=np.float32)
        vn_eff = np.frombuffer(f.read(n * 4), dtype=np.float32)
    return eng, (X, P1, pr1, e1, vn_eff)


def test_bound_stage1_gpu_parity(tmp_path):
    """Phase-2: the bound_stage1 GPU kernel must reproduce the CPU Stage-1 k1
    survivor set (the O(N) scan), using the motor's real pr1_f_ projections.

    The kernel computes UB(v,q) = <pr1[v],pq> + e1[v]*qr + eps and the same
    sortable score the CPU nth_element uses, so the k1 set (the prefilter pool)
    must be BIT-IDENTICAL to the CPU path. Skipped when no OpenCL GPU backend.
    """
    native = winnex_madhava._native
    eng, (X, P1, pr1, e1, vn_eff) = _bound_engine_and_index(tmp_path)
    N, s1 = pr1.shape
    # A SpeedEngine GPU is the host for the bound_stage1 hook. The OpenCL
    # loader is not on the standard path on every host; honor WINNEX_OPENCL_LIB
    # (the engine's documented env var) so CI/hosts can point at their driver.
    loader = os.environ.get("WINNEX_OPENCL_LIB", "")
    try:
        gpu_eng = native.SpeedEngine(
            np.zeros((1, s1), dtype=np.float32), s1, 0, 0, 4, True, loader)
    except RuntimeError:
        pytest.skip("bound_stage1_gpu requires an OpenCL GPU backend")
    if not gpu_eng.has_gpu():
        pytest.skip("bound_stage1_gpu requires an OpenCL GPU backend")
    cfg = eng.config()
    k1 = max(cfg.k1_min, int(N * cfg.k1_fraction))

    def k1_set(scores):
        # CPU Stage-1: nth_element keeps the k1 SMALLEST scores (cosine: -ub).
        # argpartition over the k1 smallest is the same set.
        return set(np.argpartition(scores, k1)[:k1])

    for idx in [0, 7, 100, 500, 1500, 2500]:
        q = X[idx].astype(np.float32)
        pq = (P1 @ q).astype(np.float32)                 # host projection
        qr = float(np.sqrt(max(0.0, 1.0 - pq @ pq)))     # query residual
        bias = np.array([[qr, 0.0, 1.0]], dtype=np.float32)  # qr, qm=0, ||q||^2=1
        s_gpu = gpu_eng.bound_stage1_gpu(
            pq.reshape(1, s1), pr1, e1, vn_eff, bias, 0)[0]
        # CPU reference on the SAME real pr1/e1: ub = dot + e1*qr + eps, score=-ub.
        ub = pr1 @ pq + e1 * qr + 1e-4
        s_cpu = -ub
        # Scores must match to float32 tolerance (the GPU kernel is the same math).
        assert np.allclose(s_gpu, s_cpu, atol=1e-5), f"q{idx}: score mismatch"
        assert k1_set(s_gpu) == k1_set(s_cpu), f"q{idx}: k1 survivor set diverged"
