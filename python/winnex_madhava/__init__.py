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
recall_at_k = _native.recall_at_k
ndcg_at_k = _native.ndcg_at_k
read_bigann_groundtruth = _native.read_bigann_groundtruth

__version__ = "1.0.0"
__all__ = [
    "Config",
    "SearchResult",
    "MadhavaL2",
    "recall_at_k",
    "ndcg_at_k",
    "read_bigann_groundtruth",
    "build_engine",
    "benchmark_vs_groundtruth",
    "__version__",
]


def build_engine(
    corpus: np.ndarray,
    *,
    dim: int | None = None,
    stage1_dim: int = 64,
    k: int = 10,
    k1_fraction: float = 0.05,
    postfilter: bool = True,
    seed: int = 42,
) -> MadhavaL2:
    """Build a MadhavaL2 engine over a uint8 corpus.

    Parameters
    ----------
    corpus : np.ndarray
        Shape (n, dim) uint8 (or convertible). The engine references it
        during build; the caller must keep it alive while the engine is used.
    dim : int, optional
        Vector dimensionality. Defaults to ``corpus.shape[1]``.
    """
    arr = np.ascontiguousarray(corpus, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("corpus must be a 2D array of shape (n, dim)")
    if dim is None:
        dim = arr.shape[1]
    cfg = Config()
    cfg.dim = int(dim)
    cfg.stage1_dim = int(stage1_dim)
    cfg.k = int(k)
    cfg.k1_fraction = float(k1_fraction)
    cfg.postfilter = bool(postfilter)
    cfg.seed = int(seed)
    engine = MadhavaL2(cfg)
    n = engine.build_numpy(arr)
    return _attach_buffer(engine, arr)


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
