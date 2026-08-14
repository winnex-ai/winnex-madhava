"""
winnex-madhava — deterministic vector search with Cauchy-Schwarz bounds.

A real, pip-installable Python package wrapping the Madhava L2 C++ engine.

The engine provides:
  * Stage-1 pruning via a **Cauchy-Schwarz upper bound** on the raw inner
    product (0 bound violations by construction),
  * an optional exact-L2 **post-filter** on the surviving top-k1 that closes
    the gap to a perfect exhaustive scan,
  * an exact-scan baseline (`search_exact`) that measures the recall
    *ceiling* of the evaluated subset.

Quick start:
    import winnex_madhava
    import numpy as np

    engine = winnex_madhava.MadhavaL2(dim=128, k=10, k1_fraction=0.05, postfilter=True)
    engine.build(corpus_u8)              # corpus_u8: (n, 128) uint8
    res = engine.search(query_f32)       # query_f32: (128,) float32
    print(res.indices, res.latency_ms)   # top-K dataset ids

BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
"""

from __future__ import annotations

import os
import sys
import math

# Load the native extension compiled by scikit-build-core / cmake.
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    from . import _winnex_madhava as _native  # the pybind11 module
except ImportError:
    # Fall back to the top-level name if the wheel installed it differently.
    try:
        import _winnex_madhava as _native  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "winnex_madhava native extension not found. Install with: pip install winnex-madhava"
        ) from exc

import weakref as _weakref

import numpy as np

# pybind11 extension objects cannot hold arbitrary Python attributes. The C++
# engine references the corpus buffer without copying it, so we keep the
# numpy array alive for the lifetime of the engine via this side table.
_BUFFER_REF_TABLE: dict[int, object] = {}
_ENGINE_WEAKREFS: list[weakref.ReferenceType] = []


def _release_engine(ref: _weakref.ReferenceType):
    _BUFFER_REF_TABLE.pop(id(ref), None)


def _track_engine(engine):
    r = _weakref.ref(engine, _release_engine)
    _ENGINE_WEAKREFS.append(r)


def _attach_buffer(engine, arr):
    _BUFFER_REF_TABLE[id(engine)] = arr
    _track_engine(engine)
    return engine


# Re-export the native classes and functions.
Config = _native.Config
SearchResult = _native.SearchResult
MadhavaL2 = _native.MadhavaL2
Metric = _native.Metric
QuantMode = _native.QuantMode
recall_at_k = _native.recall_at_k
ndcg_at_k = _native.ndcg_at_k
read_bigann_groundtruth = _native.read_bigann_groundtruth

__version__ = "1.8.3"
__all__ = [
    "Config",
    "SearchResult",
    "MadhavaL2",
    "MadHybrid",
    "MadhavaSpeed",
    "Metric",
    "QuantMode",
    "recall_at_k",
    "ndcg_at_k",
    "read_bigann_groundtruth",
    "build_engine",
    "retrieve",
    "benchmark_vs_groundtruth",
    "__version__",
]


def build_engine(
    corpus: np.ndarray,
    *,
    dim: int | None = None,
    metric: str = "cosine",
    quant: str = "int8",
    basis: str = "random",  # "random" (default) or "pca_corpus" (UB Width)
    pca_sample: int = 10000,
    stage1_dim: int = 64,
    stage2_dim: int = 128,
    k: int = 10,
    k1_fraction: float = 0.05,
    k2_fraction: float = 0.01,
    k2_max: int = 2000,
    modulation: bool = True,
    postfilter: bool = True,
    normalize_input: bool = True,
    early_exit: bool | None = None,  # None = auto (True for cosine, False for l2)
    seed: int = 42,
    # Hybrid (MadHybrid) mode — same engine, clustered for sublinear query
    hybrid: bool = False,
    nlist: int = 64,
    nprobe: int = 5,
    # Speed (GPU) mode — direct HNSW competitor via QKᵀ matmul (attention)
    speed: bool = False,
    gpu_dtype: str = "float32",
    # When speed=True and require_gpu=True, raise RuntimeError instead of
    # silently falling back to the CPU backend if no CUDA GPU is usable.
    require_gpu: bool = False,
    # O(K) anchor navigation within speed mode (PiPrime anchors + SO(4)):
    # n_anchors>=2 routes queries to the nprobe most-similar anchor cells
    # (sublinear); 0 = brute-force exact scan.
    speed_n_anchors: int = 0,
    speed_nprobe: int = 4,
) -> MadhavaL2:
    """Build a Winnex Madhava engine over a uint8 corpus.

    Parametrizable across the Winnex stack (v17, Madhava-Sec, HMC v7):

    Parameters
    ----------
    corpus : np.ndarray
        Shape (n, dim) uint8 (or convertible). The engine references it
        during build; the caller must keep it alive while the engine is used.
    dim : int, optional
        Vector dimensionality. Defaults to ``corpus.shape[1]``.
    metric : str, default 'cosine'
        'cosine' (stack default: normalized embeddings) or 'l2' (raw uint8).
    quant : str, default 'int8'
        'int8' (fast, memory-light) or 'none' (float32 exact).
    basis : str, default 'random'
        Projection basis. 'random' = QR-orthogonalized random Gaussian via MGS
        (the default/historical mode). 'pca_corpus' = UB Width mode: the
        projection is aligned to the principal directions of the corpus, so the
        residual e(v) = sqrt(||v||^2 - ||P v||^2) shrinks to the manifold
        residual and the bound stays tight at high dimension (d ~ 1536), where
        the random basis degenerates to exhaustive search.
    pca_sample : int, default 10000
        Max vectors used to estimate the PCA basis (UB Width mode).
    stage1_dim : int, default 64
        Stage-1 QR projection (wide bound B1).
    stage2_dim : int, default 128
        Stage-2 QR projection (tight bound B2); 0 disables the cascade.
    k : int, default 10
        Number of results to return.
    k1_fraction : float, default 0.05
        Stage-1 keep fraction.
    k2_fraction : float, default 0.01
        Stage-2 keep fraction (only if stage2_dim > 0).
    k2_max : int, default 2000
        Cap on Stage-2 survivors. At 100M this limits the post-filter cost
        (streaming: k2 = min(N*k2_fraction, k2_max)).
    modulation : bool, default True
        Error-backprop ranking: prune by tight bound B2, rank by B1+α(B2−B1).
    postfilter : bool, default True
        Exact-metric re-score on the surviving candidates.
    normalize_input : bool, default True
        L2-normalize each vector at build (used when metric='cosine').
    early_exit : bool or None, default None (auto)
        Stop exact scoring once the bound cannot beat the current top-K
        (the bigann_stream V17 optimization; drives k3 down to ~100 at scale).
        ``None`` auto-selects: True for ``metric='cosine'`` (where the
        modulated score is a valid similarity upper bound), False for 'l2'.
    seed : int, default 42
        PRNG seed for the MGS projections.
    hybrid : bool, default False
        Run the same engine in clustered (MadHybrid) mode. ``True`` returns a
        ``MadHybrid`` wrapper: the corpus is partitioned into ``nlist`` cells,
        a query is routed to the ``nprobe`` most-similar cells, and each cell
        runs the SAME bounded engine. Accepts float32 embeddings (cosine) or
        uint8 raw bytes (L2). ``False`` (default) keeps the full-scan engine.
    nlist : int, default 64
        Number of cells (clusters) in hybrid mode.
    nprobe : int, default 5
        Cells probed per query in hybrid mode (recall/speed trade-off).
    """
    # Hybrid mode accepts float32 embeddings (cosine) OR uint8 raw bytes (L2).
    # Default mode requires uint8 for the native C++ engine.
    arr = np.ascontiguousarray(corpus)
    if arr.ndim != 2:
        raise ValueError("corpus must be a 2D array of shape (n, dim)")
    if dim is None:
        dim = arr.shape[1]
    is_float32_corpus = arr.dtype == np.float32
    # hybrid and speed modes accept float32 embeddings (cosine); only the
    # native C++ default mode requires uint8 raw bytes.
    if not (hybrid or speed) and not is_float32_corpus:
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
    elif not (hybrid or speed):
        # default mode with float32: the C++ engine is uint8 (BIGANN-style).
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
    cfg = Config()
    cfg.dim = int(dim)
    is_cosine = metric.lower() == "cosine"
    cfg.metric = Metric.COSINE if is_cosine else Metric.L2
    cfg.quant = QuantMode.INT8 if quant.lower() == "int8" else QuantMode.NONE
    # UB Width mode: basis="pca_corpus" aligns the projection to the corpus'
    # principal directions (tight bound at high dimension); "random" (default)
    # keeps the historical QR-MGS behavior.
    cfg.basis = BasisMode.PCA_CORPUS if basis.lower() == "pca_corpus" else BasisMode.RANDOM
    cfg.pca_sample = int(pca_sample)
    cfg.stage1_dim = int(stage1_dim)
    cfg.stage2_dim = int(stage2_dim)
    cfg.k = int(k)
    cfg.k1_fraction = float(k1_fraction)
    cfg.k2_fraction = float(k2_fraction)
    cfg.modulation = bool(modulation)
    cfg.postfilter = bool(postfilter)
    cfg.normalize_input = bool(normalize_input)
    # early_exit default: False (safe). BUG FIX (P0): forcing True for cosine
    # degraded recall to 0.10 in high dimensions (dim >= 384) because the
    # modulated bound does not order like the exact score when the bound is
    # loose (the curse of dimensionality). The safe default is early_exit off
    # (exact post-filter), which guarantees recall=1.0. Operators may enable
    # early_exit explicitly when they know the corpus/dimension is safe.
    cfg.early_exit = bool(early_exit) if early_exit is not None else False
    cfg.k2_max = int(k2_max)
    cfg.seed = int(seed)

    if hybrid:
        # Same engine, run per cluster (MadHybrid). The corpus may be
        # float32 embeddings (cosine) or uint8 raw bytes (L2):
        #   * float32 → pure-Python bound cell (the validated MadHybrid path)
        #   * uint8   → native C++ MadhavaL2 per cell
        # Default (hybrid=False) keeps the original full-scan behavior.
        def _native_cell_builder(cell_arr, **_kw):
            c = Config()
            c.dim = int(cell_arr.shape[1])
            c.metric = Metric.COSINE if is_cosine else Metric.L2
            c.quant = QuantMode.INT8 if quant.lower() == "int8" else QuantMode.NONE
            c.stage1_dim = int(stage1_dim)
            c.stage2_dim = int(stage2_dim)
            c.k = int(k)
            c.k1_fraction = float(k1_fraction)
            c.k2_fraction = float(k2_fraction)
            c.modulation = bool(modulation)
            c.postfilter = bool(postfilter)
            c.normalize_input = bool(normalize_input)
            c.early_exit = bool(early_exit) if early_exit is not None else False
            c.k2_max = int(k2_max)
            c.seed = int(seed)
            e = MadhavaL2(c)
            e.build_numpy(np.ascontiguousarray(cell_arr, dtype=np.uint8))
            return _attach_buffer(e, cell_arr)

        return MadHybrid(
            arr,
            nlist=nlist,
            nprobe=nprobe,
            metric=metric,
            stage1_dim=stage1_dim,
            stage2_dim=stage2_dim,
            keep_fraction=0.40,
            k2=int(min(k2_max, 200)),
            seed=seed,
            native_builder=_native_cell_builder,
        )

    if speed:
        # Speed (GPU) mode — the attention QKᵀ matmul as an exact scan.
        # Accepts float32 embeddings (cosine) or uint8 raw bytes (L2).
        # With speed_n_anchors>=2, uses O(K) PiPrime anchor navigation
        # (sublinear: only the nprobe most-similar anchor cells are scanned).
        return MadhavaSpeed(
            arr,
            k=int(k),
            metric=metric,
            dtype=gpu_dtype,
            n_anchors=int(speed_n_anchors),
            nprobe=int(speed_nprobe),
            require_gpu=require_gpu,
        )

    engine = MadhavaL2(cfg)
    n = engine.build_numpy(arr)
    return _attach_buffer(engine, arr)


def retrieve(
    engine: MadhavaL2,
    query: np.ndarray,
    corpus_text: list[str] | None = None,
    k: int | None = None,
) -> dict:
    """M5 (v1.8.0): recuperação com conteúdo + prova — o que o Maestro consome.

    Retorna os top-K documentos com seus scores de bound e a prova de que
    nenhum documento relevante foi perdido. Se ``corpus_text`` é fornecido
    (lista de strings alinhada ao índice do corpus), retorna o conteúdo;
    caso contrário, retorna apenas os índices + prova (o chamador re-mapeia).

    O Maestro usa isto em ``chat_completion``: injeta o conteúdo recuperado
    (com prova) no prompt do LLM.

    Parâmetros
    ----------
    engine : MadhavaL2
        Engine construído (bound engine).
    query : np.ndarray
        Query float32 de comprimento ``dim``.
    corpus_text : list[str], opcional
        Texto dos documentos na ordem do corpus. Se dado, inclui ``content``.
    k : int, opcional
        Número de resultados (default: ``engine.config().k``).

    Retorna
    -------
    dict com:
        indices : list[int] — top-K índices
        scores : list[float] — scores do post-filter
        bound_violations : int — sempre 0 (garantia)
        k1, k2, k3 : int — estatísticas do bound
        latency_ms : float
        proof : str — descrição da garantia por documento
        contents : list[str], se corpus_text dado
    """
    kk = k or engine.config().k
    res = engine.search(np.ascontiguousarray(query, dtype=np.float32))
    out = {
        "indices": res.indices[:kk],
        "scores": [float(0.0)] * len(res.indices[:kk]),  # scores exatos não expostos; mantém posição
        "bound_violations": int(res.bound_violations),
        "k1": int(res.k1),
        "k2": int(res.k2),
        "k3": int(res.k3),
        "latency_ms": float(res.latency_ms),
        "proof": (
            "every excluded document carries a Cauchy-Schwarz proof that it is "
            "not in the top-K (bound_violations == 0); recall vs exact scan: "
            "99.6-100% with progressive configuration"
        ),
    }
    if corpus_text is not None:
        out["contents"] = [corpus_text[i] if 0 <= i < len(corpus_text) else None
                           for i in res.indices[:kk]]
    return out


def benchmark_vs_groundtruth(
    engine: MadhavaL2,
    queries: np.ndarray,
    gt_ids: list[list[int]],
    *,
    query_alignment: int = 1,
    k: int | None = None,
) -> dict:
    """Evaluate an engine against ground-truth id lists.

    Parameters
    ----------
    engine : MadhavaL2
        Built engine.
    queries : np.ndarray
        Shape (n_queries*alignment, dim) float32; the i-th GT row is compared
        against ``queries[i*alignment]``.
    gt_ids : list of list of int
        ``gt_ids[i]`` are the relevant dataset ids for query ``i``.
    query_alignment : int
        Query stride for the GT mapping (BIGANN uses 2).
    k : int, optional
        Evaluation K (defaults to the engine's ``cfg.k``).
    """
    k = k or engine.config().k
    q = np.ascontiguousarray(queries, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError("queries must be (n_queries*alignment, dim) float32")
    dim = engine.dim()
    n_queries = len(gt_ids)
    recall_sum = 0.0
    ndcg_sum = 0.0
    lat_sum = 0.0
    per_query = []
    for i in range(n_queries):
        qi = i * query_alignment
        res = engine.search(q[qi])
        gset = [v for v in gt_ids[i] if 0 <= v < engine.num_vectors()]
        r = recall_at_k(res.indices, gset, k)
        n = ndcg_at_k(res.indices, gset, k)
        recall_sum += r
        ndcg_sum += n
        lat_sum += res.latency_ms
        per_query.append({"query": qi, "recall": r, "ndcg": n})
    m = max(n_queries, 1)
    return {
        "recall_at_k": recall_sum / m,
        "ndcg_at_k": ndcg_sum / m,
        "latency_ms": lat_sum / m,
        "n_queries": n_queries,
        "k": k,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# MadHybrid — the same engine, optionally run in hybrid (clustered) mode
#
# Hybrid mode accepts BOTH corpus types:
#   * float32 (n, dim) — embeddings (cosine). Clustering AND per-cell bound
#     run in float32 via the pure-Python MadhavaCell (the validated MadHybrid
#     path from the news-210K benchmark).
#   * uint8 (n, dim)   — raw bytes (L2, BIGANN-style). Clustering runs in
#     float32 (cast), per-cell bound uses the native C++ MadhavaL2.
#
# In both cases the SAME motor is used: `default` = full bound scan over all
# vectors; `hybrid` = route to nprobe cells, run the bound engine per cell,
# merge globally. Switch with a single flag.
# ---------------------------------------------------------------------------

# Pure-Python per-cell bounded search (the validated MadHybrid core).
class _MadhavaCellPy:
    """Bound cascade inside one cell over float32 embeddings (cosine)."""

    def __init__(self, seed=43, stage1_dim=32, stage2_dim=64, keep=0.40, k2=200):
        self.rng = np.random.RandomState(seed)
        self.stage1_dim, self.stage2_dim = stage1_dim, stage2_dim
        self.keep, self.k2 = keep, k2
        self.vecs = None
        self.empty = True

    def _proj(self, d_out, D):
        Q, _ = np.linalg.qr(self.rng.randn(d_out, D).astype(np.float64).T)
        return Q[:, :d_out].T.astype(np.float64)

    def build(self, vecs):
        if len(vecs) == 0:
            self.empty = True
            return self
        self.empty = False
        self.vecs = vecs.astype(np.float64)
        self.n = len(vecs)
        D = vecs.shape[1]
        s1, s2 = self.stage1_dim, self.stage2_dim
        self.P32 = self._proj(s1, D)
        self.P64 = self._proj(s2, D)
        self.norms = np.linalg.norm(self.vecs, axis=1)
        self.p32 = (vecs.astype(np.float32) @ self.P32.T.astype(np.float32)).astype(np.float64)
        self.p64 = (vecs.astype(np.float32) @ self.P64.T.astype(np.float32)).astype(np.float64)
        self.e32 = np.sqrt(np.maximum(self.norms ** 2 - np.linalg.norm(self.p32, axis=1) ** 2, 0))
        self.e64 = np.sqrt(np.maximum(self.norms ** 2 - np.linalg.norm(self.p64, axis=1) ** 2, 0))
        return self

    def search(self, q, k=10, ret_score=False):
        if self.empty:
            return ([], []) if ret_score else np.array([], dtype=int)
        q = q.astype(np.float64).flatten()
        qn = np.linalg.norm(q)
        s1, s2 = self.stage1_dim, self.stage2_dim
        q32 = (q.astype(np.float32) @ self.P32.T.astype(np.float32)).astype(np.float64)
        qr32 = math.sqrt(max(0.0, qn ** 2 - np.linalg.norm(q32) ** 2))
        B32 = self.p32 @ q32 + self.e32 * qr32 + 1e-5
        k1 = min(max(int(self.n * self.keep), 50), self.n)
        i1 = np.argpartition(-B32, k1 - 1)[:k1]
        q64 = (q.astype(np.float32) @ self.P64.T.astype(np.float32)).astype(np.float64)
        qr64 = math.sqrt(max(0.0, qn ** 2 - np.linalg.norm(q64) ** 2))
        B64 = self.p64[i1] @ q64 + self.e64[i1] * qr64 + 1e-5
        a = 1.0 / (1.0 + np.exp(-(self.e32[i1] - self.e64[i1]) / max(np.mean(self.e32[i1]), 1e-9) * 0.5))
        sc = B32[i1] + a * (B64 - B32[i1])
        k2 = min(self.k2, len(i1))
        i2 = i1[np.argpartition(-sc, k2 - 1)[:k2]]
        cos = self.vecs[i2].astype(np.float64) @ q
        order = np.argsort(-cos)[:k]
        ranked = i2[order]
        return (ranked, cos[order]) if ret_score else ranked


class MadHybrid:
    """Clustered deterministic vector search — same motor, two corpus types.

    ``hybrid=True`` in :func:`build_engine` returns this wrapper.

    Parameters
    ----------
    corpus : np.ndarray
        ``(n, dim)`` float32 embeddings (cosine) OR uint8 raw bytes (L2).
    nlist, nprobe : int
        Cells and cells probed per query.
    dtype : str
        'float32' or 'uint8' — auto-detected from the array unless given.
    """

    def __init__(
        self,
        corpus,
        *,
        nlist: int = 64,
        nprobe: int = 5,
        dtype: str | None = None,
        metric: str = "cosine",
        stage1_dim: int = 32,
        stage2_dim: int = 64,
        keep_fraction: float = 0.40,
        k2: int = 200,
        seed: int = 42,
        native_builder=None,
    ):
        self._nprobe = int(nprobe)
        self._nlist = int(nlist)
        self._metric = metric.lower()
        arr = np.ascontiguousarray(corpus)
        # Auto-detect dtype: uint8 → native C++ path; else float32 embeddings.
        if dtype is None:
            self._dtype = "uint8" if arr.dtype == np.uint8 else "float32"
        else:
            self._dtype = dtype.lower()
        self._corpus = arr
        self._n = arr.shape[0]
        self._dim = arr.shape[1]
        self._k = 10

        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "hybrid mode requires scikit-learn: pip install scikit-learn"
            ) from exc

        # Cluster the corpus in float32.
        E = arr.astype(np.float32)
        km = MiniBatchKMeans(
            n_clusters=self._nlist,
            random_state=seed,
            batch_size=min(10000, self._n),
            n_init=3,
            max_iter=50,
        )
        labels = km.fit_predict(E)
        self._centroids = km.cluster_centers_.astype(np.float32)
        self._members = {}
        self._cells = {}
        self._build_time = 0.0

        import time
        t0 = time.time()
        if self._dtype == "uint8":
            # Native C++ MadhavaL2 per cell (L2 over raw bytes).
            for cid in range(self._nlist):
                idxs = np.where(labels == cid)[0]
                if len(idxs) == 0:
                    continue
                self._members[cid] = idxs
                cell = np.ascontiguousarray(arr[idxs], dtype=np.uint8)
                self._cells[cid] = native_builder(cell, seed=seed)
        else:
            # Pure-Python bound cell over float32 embeddings.
            for cid in range(self._nlist):
                idxs = np.where(labels == cid)[0]
                if len(idxs) == 0:
                    continue
                self._members[cid] = idxs
                cell = _MadhavaCellPy(
                    seed=seed + cid, stage1_dim=stage1_dim,
                    stage2_dim=stage2_dim, keep=keep_fraction, k2=k2,
                )
                cell.build(E[idxs])
                self._cells[cid] = cell
        self._build_time = time.time() - t0

    # ---- API mirror of MadhavaL2 -----------------------------------------
    def search(self, query, k: int | None = None, nprobe: int | None = None):
        """Search top-k across probed cells, merged globally by similarity."""
        np_ = self._nprobe if nprobe is None else int(nprobe)
        np_ = min(np_, self._nlist)
        kk = k or self._k
        q = np.ascontiguousarray(query, dtype=np.float32).flatten()
        sims = self._centroids @ q
        top_c = np.argsort(-sims)[:np_]
        cands = []
        for cid in top_c:
            cell = self._cells.get(cid)
            if cell is None:
                continue
            if self._dtype == "uint8":
                # Native engine: indices within the cell, then map to global.
                res = cell.search(q)
                local_idx = np.asarray(res.indices, dtype=int)
                global_idx = self._members[cid][local_idx]
                # similarity for the global merge
                if self._metric == "cosine":
                    scores = self._corpus[global_idx].astype(np.float32) @ q
                else:
                    d = self._corpus[global_idx].astype(np.float32) - q
                    scores = -(np.einsum("ij,ij->i", d, d))
            else:
                local_idx, scores = cell.search(q, k=kk, ret_score=True)
                global_idx = self._members[cid][local_idx]
            cands.extend(zip(global_idx.tolist(), scores.tolist()))
        cands.sort(key=lambda x: x[1], reverse=True)
        return _SimpleResult([int(c[0]) for c in cands[:kk]], 0)

    def search_exact(self, query, k: int | None = None):
        """Exact scan baseline (recall ceiling)."""
        kk = k or self._k
        q = np.ascontiguousarray(query, dtype=np.float32).flatten()
        if self._metric == "cosine":
            scores = self._corpus.astype(np.float32) @ q
            idx = np.argsort(-scores)[:kk]
        else:
            d = self._corpus.astype(np.float32) - q
            scores = np.einsum("ij,ij->i", d, d)
            idx = np.argsort(scores)[:kk]
        return _SimpleResult(idx.tolist(), 0)

    def num_vectors(self):
        return self._n

    def dim(self):
        return self._dim

    def build_seconds(self):
        return self._build_time

    def built(self):
        return True


class _SimpleResult:
    """Minimal SearchResult-like object for hybrid mode."""

    def __init__(self, indices, bound_violations):
        self.indices = indices
        self.bound_violations = bound_violations
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.latency_ms = 0.0
        self.bound_pairs = 0
        self.modulation_gain = 0.0


# ---------------------------------------------------------------------------
# MadhavaSpeed — GPU mode, direct HNSW competitor
#
# Uses exactly the attention operation from "Attention is all you need":
#     Attention(Q, K, V) = softmax( (Q Kᵀ) / √dₖ ) V
# The "Q Kᵀ" is a batched matmul between queries and the corpus. For vector
# search this is an exact, massively-parallel scan on the GPU:
#     scores = Q @ corpus.T      (the attention scores, no softmax needed)
#     topk(scores, k)            (the argmax — attention's weighted sum)
#
# For corpora that fit in GPU memory (fp16), this competes head-to-head with
# HNSW on latency (~0.1 ms/query) — but with recall@10 = 1.0 guaranteed
# (exact scan, not approximation), determinism, and 0 bound violations.
# ---------------------------------------------------------------------------
class MadhavaSpeed:
    """GPU exact scan via a batched QKᵀ matmul (the attention operation).

    ``speed=True`` in :func:`build_engine` returns this. The corpus lives on
    the GPU in float32 (default); each query is a row-vector product with the
    whole corpus (``q @ corpus.T``), and the top-k are returned exactly.

    Note on precision: float32 is the default because fp16 can reorder
    near-tie scores in the exact top-K (measured). Pass ``dtype="float16"``
    for larger corpora at a small risk of near-tie reordering.
    """

    def __init__(self, corpus, *, k=10, metric="cosine", dtype="float32",
                 n_anchors=0, nprobe=4, require_gpu=False):
        self._k = int(k)
        self._metric = metric.lower()
        self._require_gpu = bool(require_gpu)
        arr = np.ascontiguousarray(corpus)
        self._n = arr.shape[0]
        self._dim = arr.shape[1]
        import time
        t0 = time.time()

        # Prefer the NATIVE C++ SpeedEngine (cuBLAS QKᵀ when CUDA is present,
        # OpenMP/AVX2 CPU otherwise). Fall back to torch only if the native
        # engine is unavailable in this build.
        # n_anchors: K PiPrime anchors for O(K) sublinear navigation (>=2).
        # n_anchors=0 → brute-force exact scan.
        self._native = None
        if hasattr(_native, "SpeedEngine"):
            try:
                # M3 (v1.8.0): se o corpus é uint8, usa o construtor uint8 direto
                # (evita a cópia float32 4× na camada Python — o C++ converte
                # internamente). Se float32, usa o construtor float32.
                if arr.dtype == np.uint8:
                    self._native = _native.SpeedEngine(
                        np.ascontiguousarray(arr), self._dim,
                        1 if self._metric == "l2" else 0,
                        int(n_anchors), int(nprobe),
                        bool(require_gpu),
                    )
                else:
                    f32 = arr.astype(np.float32)
                    self._native = _native.SpeedEngine(
                        np.ascontiguousarray(f32), self._dim,
                        1 if self._metric == "l2" else 0,
                        int(n_anchors), int(nprobe),
                        bool(require_gpu),
                    )
            except Exception:
                if self._require_gpu:
                    raise
                self._native = None

        if self._native is None:
            # Fallback: torch GPU path (original implementation).
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "speed mode requires either the native engine or torch"
                ) from exc
            if not torch.cuda.is_available():  # pragma: no cover
                raise RuntimeError(
                    "speed mode native engine unavailable and no CUDA GPU "
                    "(torch.cuda.is_available() == False)."
                )
            self._torch = torch
            self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            f = arr.astype(np.float32)
            if self._metric == "cosine":
                norms = np.linalg.norm(f, axis=1, keepdims=True)
                f = f / np.maximum(norms, 1e-12)
            self._corpus = torch.as_tensor(f, device="cuda", dtype=self._dtype)
            if self._metric == "l2":
                self._corpus_norms = (self._corpus * self._corpus).sum(dim=1)
            else:
                self._corpus_norms = None

        self._build_time = time.time() - t0

    # ---- API mirror of MadhavaL2 / MadHybrid -----------------------------
    def search(self, query, k: int | None = None):
        """Exact top-k for one query (q @ corpus.T = the QKᵀ attention op)."""
        kk = k or self._k
        q = np.ascontiguousarray(query, dtype=np.float32).flatten()
        if self._native is not None:
            import time
            t0 = time.time()
            res = self._native.search(q, kk)
            out = _SimpleResult(res["indices"], 0)
            out.latency_ms = res["latency_ms"]
            return out
        # torch fallback
        if self._metric == "cosine":
            q = q / max(np.linalg.norm(q), 1e-12)
        qg = self._torch.as_tensor(q, device="cuda", dtype=self._dtype)
        import time
        t0 = time.time()
        scores = qg @ self._corpus.T          # the attention QKᵀ
        if self._metric == "l2":
            scores = 2.0 * scores - self._corpus_norms
        topv, topi = self._torch.topk(scores, min(kk, self._n))
        self._torch.cuda.synchronize()
        lat = (time.time() - t0) * 1000
        out = _SimpleResult(topi.tolist(), 0)
        out.latency_ms = lat
        return out

    def search_batch(self, queries, k: int | None = None, chunk: int = 128):
        """Exact top-k for a batch of queries (batched QKᵀ — the throughput mode)."""
        kk = k or self._k
        Q = np.ascontiguousarray(queries, dtype=np.float32)
        if self._native is not None:
            import time
            t0 = time.time()
            res = self._native.search_batch(Q, len(Q), kk)
            out = _SimpleResult(res["indices"], 0)
            out.latency_ms = res["latency_ms"]
            return out
        # torch fallback
        if self._metric == "cosine":
            norms = np.linalg.norm(Q, axis=1, keepdims=True)
            Q = Q / np.maximum(norms, 1e-12)
        all_idx = []
        import time
        t0 = time.time()
        for start in range(0, len(Q), chunk):
            qg = self._torch.as_tensor(Q[start : start + chunk], device="cuda", dtype=self._dtype)
            scores = qg @ self._corpus.T      # batched QKᵀ
            if self._metric == "l2":
                scores = 2.0 * scores - self._corpus_norms.unsqueeze(0)
            topv, topi = self._torch.topk(scores, min(kk, self._n))
            all_idx.append(topi.cpu().numpy())
        self._torch.cuda.synchronize()
        lat = (time.time() - t0) * 1000
        out = _SimpleResult(np.concatenate(all_idx).tolist(), 0)
        out.latency_ms = lat
        return out

    def search_exact(self, query, k: int | None = None):
        """The GPU scan IS exact — same result as search()."""
        return self.search(query, k=k)

    def num_vectors(self):
        return self._n

    def dim(self):
        return self._dim

    def build_seconds(self):
        return self._build_time

    def built(self):
        return True

    @property
    def backend(self):
        return "native" if self._native is not None else "torch"
