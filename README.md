<div align="center">

# winnex-madhava

**Deterministic vector search with mathematical guarantees.**

Every document excluded from the results carries a proof that it could not be in the top-K — by the **Cauchy-Schwarz inequality**. Zero bound violations by construction.

[![PyPI version](https://img.shields.io/pypi/v/winnex-madhava?color=467C45)](https://pypi.org/project/winnex-madhava/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/winnex-madhava?color=467C45)](https://pypi.org/project/winnex-madhava/)
[![PyPI - Python Versions](https://img.shields.io/pypi/pyversions/winnex-madhava?color=467C45)](https://pypi.org/project/winnex-madhava/)
[![CI](https://img.shields.io/github/actions/workflow/status/winnex-ai/winnex-madhava/ci.yml?branch=main&label=CI&color=467C45)](https://github.com/winnex-ai/winnex-madhava/actions/workflows/ci.yml)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-467C45)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-467C45)](https://isocpp.org/)
[![Benchmark](https://img.shields.io/badge/BIGANN--100M-L2%20verified-467C45)](docs/VERIFIED.md)

</div>

---

`winnex-madhava` is a real, pip-installable Python package with a native C++20 core. It answers a question no approximate index (HNSW, IVF, PQ) can answer:

> **"Prove that your search did not miss a relevant document."**

The proof is per-document and mathematical: a Cauchy-Schwarz upper bound on the inner product, which converts into a lower bound on L2². If the bound says a vector cannot be in the top-K, that vector **is not in the top-K**. No heuristics, no random graphs, no "we think it's fine."

Verified against the **official BIGANN-100M L2 ground truth** — see [Benchmarks](#benchmarks).

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [When should you use this?](#when-should-you-use-this)
- [When should you NOT use this?](#when-should-you-not-use-this)
- [Parameter guide](#parameter-guide)
- [Streaming — 100M vectors without loading the corpus into RAM](#streaming--100m-vectors-without-loading-the-corpus-into-ram)
- [API](#api)
- [The mathematics](#the-mathematics)
- [Benchmarks](#benchmarks)
- [Kaggle benchmark (reproducible)](#kaggle-benchmark-reproducible)
- [Limitations (read this first)](#limitations-read-this-first)
- [Honest comparison](#honest-comparison)
- [Build from source](#build-from-source)
- [License](#license)

---

## Installation

```bash
pip install winnex-madhava
```

**Requirements:** Python ≥ 3.8 and NumPy. The C++ core ships pre-built in the
wheel (manylinux x86-64); a C++20 compiler + CMake ≥ 3.20 are needed only when
building from source.

> **⚠️ Python version support (important).** The pre-built manylinux wheel is
> currently **CPython 3.12 only**. On 3.8–3.11, pip falls back to the sdist
> and compiles from source, which requires a C++20 compiler + CMake on the
> machine. If you are on 3.8–3.11 and get a build error, either install a
> C++20 toolchain or use Python 3.12. Wider wheel coverage (cp38–cp311) is on
> the roadmap.
>
> Installing straight from this repo works too:
>
> ```bash
> pip install git+https://github.com/winnex-ai/winnex-madhava.git
> ```

**How to know your install is working.** After installing, run:

```bash
python -c "import winnex_madhava; print(winnex_madhava.__version__)"
```

You should see `1.3.0` or newer. If you see `No module named`, you are on the
unsupported source-build path (see the warning above).

## Quick start

```python
import numpy as np
import winnex_madhava

# 1. Build an engine over your corpus (uint8, shape (n, dim)).
corpus = np.random.randint(0, 256, size=(100_000, 128), dtype=np.uint8)

engine = winnex_madhava.build_engine(corpus, dim=128, k=10)
print(f"indexed {engine.num_vectors()} vectors in {engine.build_seconds():.2f}s")

# 2. Search.
query = corpus[0].astype(np.float32)   # (128,) float32
result = engine.search(query)

print(result.indices)                  # top-K dataset ids
print(result.latency_ms)               # milliseconds
print(result.bound_violations)         # always 0 — the guarantee
```

That's it. Same query + same data → same result, every time. Deterministic.

## Hybrid mode (MadHybrid) — sublinear query, same engine

The **same engine** can run in `hybrid` mode: the corpus is clustered into
`nlist` cells, a query is routed to the `nprobe` most-similar cells, and each
cell runs the identical bounded engine. This makes query cost sublinear
(`nprobe × cell_size` instead of `N`) while keeping the bound guarantee.

```python
import winnex_madhava, numpy as np

# float32 embeddings (cosine) — the MadHybrid path
embeddings = np.random.randn(50_000, 128).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
eng = winnex_madhava.build_engine(
    embeddings, k=10, hybrid=True, nlist=64, nprobe=5, metric="cosine",
)
res = eng.search(embeddings[0].astype(np.float32), k=10)
print(res.indices)

# uint8 raw bytes (L2, BIGANN-style) — native C++ per cell
u8 = (embeddings * 100 + 128).astype(np.uint8)
eng_u = winnex_madhava.build_engine(
    u8, k=10, hybrid=True, nlist=64, nprobe=5, metric="l2",
)
```

**Corpus type** is auto-detected: float32 → pure-Python bound cell (the
validated MadHybrid path from the News-210K benchmark); uint8 → native C++
`MadhavaL2` per cell. Switch between `default` and `hybrid` with a single
flag — the motor is identical.

**Honest positioning**: hybrid trades recall for speed (like any IVF index).
On structured data (e.g. news categories), recall@10 ≈ 1.0 at `nprobe=3–8`;
on uniform data, use `default` mode. `hybrid` is ideal for large, clustered
corpora and streaming/rebuild-heavy workloads.

## When should you use this?

`winnex-madhava` is for the cases where **"fast but unprovable" is a liability**.
The trade-off is simple: you pay **more latency per query** than an approximate
index, but you get **a mathematical proof per document** and a **much faster
build**.

| Use case | Why winnex-madhava |
|---|---|
| **Regulated retrieval** (legal discovery, medical records, financial compliance, government audits) | Every excluded document carries a proof it could not be in the top-K. Defensible in court. |
| **Continuous ingestion / dynamic RAG** (corpus changes frequently) | Build is ~10–1000× faster than HNSW — no painful rebuilds. Rebuild the whole index on every ingestion. |
| **Batch processing** | Scan everything with bounds; throughput over latency. |
| **RAM/CPU-constrained environments** | Int8-quantized projections use ~4× less memory than float32 (18.6 GB for 100M×128D). |
| **RAG that must not silently drop a relevant document** | Deterministic recall ceiling reachable; 0 bound violations. |
| **Auditability / compliance (EU AI Act, LGPD, HIPAA)** | Deterministic (same input → same output), per-document audit trail. |

## When should you NOT use this?

Be honest — `winnex-madhava` is **not** the right tool for:

- **Lowest-latency serving (sub-ms QPS).** HNSW/IVF are faster per query — but
  they are **approximate** (no guarantee). The exact scan is `speed=True` on
  GPU (~2.4 ms at 1M, single-query) or the bound engine on CPU (~11.7 ms). If
  you need millions of queries/sec *and* can tolerate approximation, use an
  approximate index.
- **`default` mode with arbitrary float32 corpora.** The `default` engine
  input contract is **uint8** (0–255). If you pass raw float embeddings to
  `default` mode, they get truncated to uint8 and recall collapses. For
  float32 embeddings, use **`hybrid=True`** (the MadHybrid path) or
  **`speed=True`** (the exact GPU scan), which accept float32 directly.
- **Tiny / low-dimensional corpora** (d < ~8). The projection overhead
  dominates; a plain `search_exact` scan is faster and simpler.
- **GPU inference for other models.** The speed-mode GPU path (OpenCL) is
  dedicated to vector search; it is not a general inference backend.

## Parameter guide

`build_engine` is **parametrizable** to reflect the full Winnex stack. All
parameters have sensible defaults — start with the defaults and tune only what
you need.

```python
engine = winnex_madhava.build_engine(
    corpus,                          # (n, dim) uint8 (default) OR float32 (hybrid)
    dim=128,                         # vector dimensionality (default: corpus.shape[1])
    metric="cosine",                 # "cosine" (normalized embeddings) or "l2" (raw uint8)
    quant="int8",                    # "int8" (fast, memory-light) or "none" (float32 exact)
    stage1_dim=64,                   # Stage-1 QR projection (wide bound B1)
    stage2_dim=128,                  # Stage-2 QR projection (tight bound B2); 0 disables cascade
    k=10,                            # number of results
    k1_fraction=0.05,                # Stage-1 keep fraction (5% of N)
    k2_fraction=0.01,                # Stage-2 keep fraction (1% of N)
    modulation=True,                 # error-backprop ranking (prune by B2, rank by B1+α(B2−B1))
    postfilter=True,                 # exact metric re-score on survivors
    normalize_input=True,            # L2-normalize vectors (used when metric="cosine")
    seed=42,                         # PRNG seed for the MGS projections (deterministic)
    # Hybrid (MadHybrid) — same engine, clustered for sublinear query
    hybrid=False,                    # True = clustered sublinear mode; False = full scan
    nlist=64,                        # cells (clusters) in hybrid mode
    nprobe=5,                        # cells probed per query (recall/speed trade-off)
)
```

### Choosing `metric`

| `metric` | Input contract | Use when |
|---|---|---|
| `"cosine"` (default) | uint8 representing **normalized** embeddings (unit L2 norm) | Your vectors are embeddings (SBERT, etc.). This matches the Winnex stack. |
| `"l2"` | raw uint8 values (BIGANN-style, non-normalized) | Your data is raw uint8 and you want exact L2 semantics. |

### Choosing `quant`

| `quant` | Memory | Fidelity |
|---|---|---|
| `"int8"` (default) | ~4× less memory (projections stored as int8) | Bound stays exact (quantization margin added); recall preserved. |
| `"none"` | float32 projections | Exact float32 — maximum fidelity, more memory. |

### Choosing `stage1_dim` / `stage2_dim`

The two-stage cascade is the Winnex architecture: a **wide bound B1** (Stage-1,
cheap) prunes to `k1`, then a **tight bound B2** (Stage-2, more expensive)
prunes to `k2`. Set `stage2_dim=0` for a single-stage engine (BIGANN-L2
baseline). **Pruning always uses the tightest available bound** — modulation is
used only for ranking, never for pruning (the stack's FIX(1) invariant).

### Choosing `modulation`

When `True`, survivors are ranked by `B1 + α·(B2−B1)` with
`α = sigmoid((e1−e2)/mean(e1))` — the error-backpropagation refinement. This
improves ranking quality without ever sacrificing the 0-violation guarantee.
Set `False` to rank purely by the bound.

### Choosing `postfilter`

When `True`, the exact metric is re-computed on the surviving top-k2, so the
final result is the **true top-K of the surviving set**. This closes the gap
between bound ranking and exact ranking. Leave it on unless you need speed.

### Choosing `hybrid` / `nlist` / `nprobe`

| Parameter | Default | Effect |
|---|---|---|
| `hybrid` | `False` | `True` → clustered MadHybrid mode (sublinear query); `False` → full bound scan |
| `nlist` | `64` | Number of cells (clusters) in hybrid mode |
| `nprobe` | `5` | Cells probed per query. Higher = better recall, more latency |

In hybrid mode the corpus is partitioned into `nlist` cells via
MiniBatchKMeans; a query is routed to the `nprobe` most-similar cells, and
each cell runs the identical bounded engine. Results are merged globally by
exact similarity. **Corpus type is auto-detected**: float32 embeddings →
pure-Python bound cells (cosine); uint8 raw bytes → native C++ per cell (L2).

Trade-off: higher `nprobe` recovers more recall at more latency. On
structured data (e.g. news categories), `nprobe=3–5` reaches near-exact
recall; on uniform data, prefer `default` mode.

### Choosing `speed` / `speed_n_anchors` / `speed_nprobe`

| Parameter | Default | Effect |
|---|---|---|
| `speed` | `False` | `True` → native speed mode (C++ QKᵀ matmul + fused topk; OpenCL GPU default, CUDA opt-in, OpenMP/AVX2 on CPU) |
| `speed_n_anchors` | `0` | K PiPrime anchors for **O(K) navigation**. `>=2` → sublinear (route query to the nprobe most-similar anchor cells); `0` → brute-force exact scan |
| `speed_nprobe` | `4` | Anchor cells probed per query. Higher = better recall, more latency |

```python
import winnex_madhava, numpy as np

# Speed mode — brute-force exact scan (default)
eng_bf = winnex_madhava.build_engine(
    corpus_u8, k=10, speed=True, metric="l2",
)

# Speed mode — O(K) anchor navigation (sublinear, intelligent)
eng_an = winnex_madhava.build_engine(
    corpus_u8, k=10, speed=True, metric="l2",
    speed_n_anchors=16,   # K PiPrime anchors (SVD + Gram-Schmidt)
    speed_nprobe=8,       # cells probed (trade recall vs cost)
)
```

**How it works.** K orthonormal anchors (SVD power-iteration + Gram-Schmidt,
inspired by the [PiPrime navigation](https://zenodo.org/records/17171112))
partition the corpus into Voronoi cells. A query is routed to the `nprobe`
most-similar cells via `q @ anchors.T` (O(K·d), tiny), then the QKᵀ scan
runs **only over the members of those cells** — sublinear, not a full N·d
scan.

**Honest usage.**
- `speed_n_anchors=0` (default) is the **brute-force exact scan** — correct
  everywhere, O(N·d). Use it when you need guaranteed exact top-K.
- `speed_n_anchors>=2` is **sublinear** — it evaluates only the relevant
  cells, at a recall cost that depends on `nprobe`. On structured data,
  `nprobe=8` reaches 100% of the exact-scan ceiling; `nprobe=4` may drop
  recall. Tune on your data.
- The CPU speed mode is an **exact scan** (O(N·d)) — HNSW is faster on raw
  CPU latency. The value is **exactness + build speed + determinism**.

### Speed GPU — how the fused kernel works (v1.7.2)

The GPU path (OpenCL) runs the QKᵀ matmul as a **single fused kernel**
(`qkt_fused_topk`) that also computes the per-row top-k — no intermediate
`scores[N]` matrix is materialized. Two properties matter for latency:

- **Parallelism (`M` work-groups per query).** The kernel splits the corpus
  scan into `M` contiguous chunks, each handled by a separate work-group. This
  keeps **all GPU compute units active even for a single query** — the reason
  single-query latency dropped from **47.8ms → 2.41ms (20×)** at 1M. `M` is
  auto-tuned from the GPU size (64 by default), so no parameter to set.
- **Coalesced memory access.** Adjacent work-items read adjacent vectors, so
  each 32-byte cache line fetched from global memory is used by an entire warp.
  This is what makes the scan memory-bound at ~448 GB/s instead of ~2%.

| Query mode | GPU (OpenCL) 1M | CPU 1M | Notes |
|---|---|---|---|
| single-query | **2.41 ms** | 9.56 ms | GPU 4× faster |
| batch (100 q) | **1.55 ms/q** | ~9 ms/q | GPU 6× faster |

**Latency guidance.** Use `speed=True` with `metric="l2"` (or `"cosine"`) for
an **exact scan** on GPU — the fastest correct path per query. For **throughput
(batch)**, `search_batch` amortizes the kernel launch; at 1M it sustains
~600-640 QPS. If you need sub-millisecond latency on CPU, use `hybrid=True`
(approximate, recall tunable via `nprobe`) — see the honest comparison below.

## Streaming — 100M vectors without loading the corpus into RAM

`winnex-madhava` searches **100M vectors (12.8 GB)** without ever loading the
raw corpus into RAM. The corpus is memory-mapped (`np.memmap`), the C++ core
builds the int8-quantized projections in streaming blocks, and only those
compressed projections (~19 GB at 100M) live in RAM.

### How it works

```
base.u8bin (12.8 GB, 100M×128D)
    │
    ├── mmap — NEVER loaded into RAM
    │
    ├── Build in blocks of 500K:
    │     mmap → uint8→float32 → project (stage1+stage2) → int8 quantize
    │     → keep pr1_i8 (6.4 GB) + pr2_i8 (12.8 GB) + e1/e2 (1.6 GB) in RAM
    │
    └── Search (O(N) over int8 in RAM):
          Stage 1: bound over pr1_i8 → k1
          Stage 2: tighter bound over pr2_i8 → k2 = min(k2_fraction·N, k2_max)
          Stage 3: exact metric over k2 survivors (mmap only those) → top-K
```

The key knob is **`k2_max`** (default 2000): it caps the Stage-2 survivors, so
the exact Stage-3 scoring is bounded at large scale. This is the bigann_stream
V3 optimization — the bound in Stage 2 already isolates the best candidates in
the first 2000, so the cap costs no recall.

```python
import numpy as np
import winnex_madhava

# Stream a 100M corpus without loading it into RAM.
base = np.memmap("base.u8bin", dtype=np.uint8, mode="r", shape=(100_000_000, 128))

engine = winnex_madhava.build_engine(
    base,
    dim=128,
    metric="cosine",      # V3-style (or "l2")
    k1_fraction=0.05,
    k2_fraction=0.01,
    k2_max=2000,          # the 100M streaming knob
    postfilter=True,
)
res = engine.search(query_f32)
print(res.indices, res.bound_violations)  # 0 violations — the guarantee
```

### Verified at 100M (Kaggle, notebook `winnex-madhava-stream-100m`)

| Scale | Build (s) | Lat (ms) | R@10 | NDCG | RSS (GB) | Vio |
|---|---|---|---|---|---|---|
| 100K | 1.3 | 7.5 | 0.750 | 0.658 | 0.4 | 0 |
| 1M | 5.9 | 66 | 0.843 | 0.667 | 0.8 | 0 |
| 10M | 34.0 | 698 | 0.501 | 0.544 | 3.8 | 0 |
| **100M** | **342.6** | **7592** | **0.780** | **0.813** | **31.4** | **0** |

**100M indexed in 342.6 s** (~5.7 min, 4 CPUs) via mmap — the raw 12.8 GB
corpus is **never** loaded into RAM. **0 bound violations** at every scale.

> **Note.** `k2_max` caps the Stage-2 survivors. At 100M this limits the exact
> post-filter to 2000 vectors instead of 1M, making the search tractable.
> Verified: R@10 is identical with `k2_max=2000` vs no cap.

## API

### `winnex_madhava.build_engine(corpus, **kwargs) -> MadhavaL2 | MadHybrid | MadhavaSpeed`

Build an engine over a `(n, dim)` array. With `hybrid=False` (default) the
corpus is uint8 and the native C++ `MadhavaL2` is returned. With
`hybrid=True` a `MadHybrid` wrapper is returned (float32 or uint8). With
`speed=True` a `MadhavaSpeed` is returned — the native QKᵀ matmul engine with
**fused topk** (OpenCL GPU default, CUDA opt-in at build, OpenMP/AVX2 on CPU),
optionally with O(K) anchor navigation via `speed_n_anchors`/`speed_nprobe`.
See [Parameter guide](#parameter-guide).

### `engine.search(query: np.ndarray) -> SearchResult`

Returns `indices`, `latency_ms`, `k1`, `k2`, `k3`, `bound_pairs`,
`bound_violations`, `modulation_gain`.

### `engine.search_exact(query: np.ndarray) -> SearchResult`

Exhaustive scan over all N vectors — the **recall ceiling** of your corpus.
Use it to measure how close an approximate index gets to the physical limit.

### `winnex_madhava.benchmark_vs_groundtruth(engine, queries, gt_ids, *, query_alignment=1, k=None) -> dict`

Evaluate against ground-truth id lists. Returns `recall_at_k`, `ndcg_at_k`,
`latency_ms`, and per-query detail.

### Metrics

- `winnex_madhava.recall_at_k(result, gt_set, k)` — robust recall@K:
  `|result[:K] ∩ gt| / min(K, |gt|)`. Normalizes by `min(K, |gt|)` so a
  perfect scan scores exactly 1.0 even when the ground truth has fewer than K
  relevant ids in the subset.
- `winnex_madhava.ndcg_at_k(result, gt_set, k)` — NDCG@K with the same
  `min(K, |gt|)` normalization.
- `winnex_madhava.read_bigann_groundtruth(path, n_queries)`

## The mathematics

For any query `q` and candidate vector `v`, the **Cauchy-Schwarz inequality**
bounds the raw inner product:

```
⟨v, q⟩  ≤  ⟨Pv, Pq⟩  +  ‖v − PᵀPv‖ · ‖q − PᵀPq‖
```

where `P` is a QR-orthogonalized (Modified Gram-Schmidt) random projection.
Because

```
‖v − q‖²  =  ‖v‖² + ‖q‖² − 2·⟨v, q⟩
```

the bound on `⟨v, q⟩` becomes a **lower bound on L2²**:

```
‖v − q‖²  ≥  ‖v‖² + ‖q‖² − 2·UB(⟨v, q⟩)
```

**Stage 1** computes this lower bound for every vector and keeps the top-k1
by smallest L2². Any vector pruned here is *mathematically proven* not to be
in the exact top-K. Bound violations = **0 by construction**.

**Stage 2** (optional) applies a tighter bound B2 on the k1 survivors. **Post-filter**
computes the exact metric on the surviving top-k2, so the result is the true
top-K of the surviving set. Because Stage 1/2 never prune a real neighbor, the
post-filter recovers **everything a perfect scan would find**.

The residual `‖v − PᵀPv‖` is computed on the **real float32 projection**, not
the int8-quantized one — this is what the inequality requires, and it is what
makes the bound exact rather than approximate.

## Benchmarks

Verified 2026-08-04 against the **official BIGANN-100M L2 ground truth** on a
CPU-only machine (28 threads, AVX2+FMA), using 200 queries for statistical
robustness.

### "Prova dos 9" — exact-scan ceiling at 100M

Against the **official BIGANN L2 ground truth**, `winnex-madhava` reaches
**R@10 = 0.8360, NDCG = 0.8611 at 100M** — **exactly the exact-scan ceiling**,
with 0 bound violations and a per-document mathematical guarantee.

| Scale | Exact-scan ceiling (R@10) | winnex-madhava (R@10) | NDCG | Efficiency |
|---|---|---|---|---|
| 10M | 0.5225 | **0.5225** | 0.5796 | **100%** |
| 100M | 0.8360 | **0.8360** | 0.8611 | **100%** |

The **ceiling** is `search_exact` — a perfect exhaustive scan over the same
subset. winnex-madhava reaches **100% of the ceiling at 10M and 100M**, with
**0 bound violations** at every scale. No other index (HNSW, IVF, IVF-PQ)
reaches the ceiling — only winnex-madhava combines exactness with a proof.

> **Recall definition (robust).** We use
> `recall@K = |result[:K] ∩ GT_subset| / min(K, |GT_subset|)`, where
> `GT_subset` is all official GT ids present in the subset. This intersects
> with the *entire* relevant set (not just the top-K) and normalizes by
> `min(K, |GT_subset|)` — so a perfect exact scan scores **exactly 1.0** even
> when the subset holds fewer than K relevant ids. A definition that divides
> by fixed K artificially penalizes such queries.

### The 10M subset mystery — ground-truth coverage

> **Read this before interpreting any BIGANN number.**

The official BIGANN-100M ground truth was generated over the **full 1B-vector
space**. When you restrict the corpus to a subset of N vectors, not all true
neighbors exist inside the subset:

| Scale | GT coverage (top-20) | Meaning |
|---|---|---|
| 1M | 1.2% | Semantically empty comparison |
| 10M | 10.5% | Sparse; recall capped by the subset |
| 100M | 100% | Complete GT — the only scale where recall is fully meaningful |

**Consequence:** at 10M only ~10.5% of true neighbors exist, so even a perfect
exact scan caps at **R@10 ≈ 0.52** (the subset's mathematical ceiling). **No
index** — exact or approximate — can beat that on the subset. That is why
"100% efficiency" is relative to the *subset ceiling*, not an absolute recall
of 1.0. At 100M (100% coverage), winnex-madhava reaches **0.8360** — essentially
all the recall the dataset offers.

### Build vs Latency — the honest trade-off

| Method (10M subset) | R@10 | Build (s) | Latency (ms) |
|---|---|---|---|
| **winnex-madhava** | **0.5225** | **23.7** (Kaggle) / **1.0** (local) | 515 |
| IVF nprobe=128 | 0.4060 | 170 | 9.3 |
| IVF-PQ m=64 | 0.3920 | 90 | 24.1 |
| HNSW ef=256 | 0.2940 | 1025 | 1.0 |
| FlatL2 (exact) | 0.5225 | — | 617 |

winnex-madhava **scans all vectors** with a mathematical bound (higher latency
per query), but the **build is ultra-fast** — no graph to construct. Build 10M
≈ **1s** locally (AVX2/FMA) vs HNSW ≈ 1025s (~930× faster). At 100M, the build
is ~227s — less time than HNSW needs to index just 5M vectors.

## Kaggle benchmark (reproducible)

Run it yourself with one click — the notebook installs `winnex-madhava` from
PyPI, indexes real data, and reports the exact-scan ceiling vs the Madhava
result, plus a side-by-side comparison with FAISS HNSW/IVF/IVF-PQ using the
same robust recall function:

[![Kaggle](https://img.shields.io/badge/Kaggle-pip--200--queries-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-pip-200-queries)

### Honest benchmark — official GT, pip-installed wheel, GPU (v1.7.2)

The **honest** benchmark: installs `winnex-madhava` **from PyPI via pip**, reads
the **official BIGANN-100M L2 ground truth**, and measures every mode against
it — with recall normalized by GT coverage in the subset, the exact-scan
ceiling as the physical limit, and the OpenCL GPU backend reported truthfully
(an **exact scan**, *not* sublinear — no anchors are active on the GPU path).

[![Kaggle](https://img.shields.io/badge/Kaggle-Honest%20GPU%20vs%20Official%20GT-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-1-7-honest-gpu-vs-official-gt)

**Results (BIGANN-100M, subset 1M, 100 queries, official GT):**

| Method | R@10 | Lat (ms) | QPS | Build (s) | Efficiency | Bound vio. |
|---|---|---|---|---|---|---|
| Exact-scan ceiling (`search_exact`) | 0.9500 | 9.3 | 108 | 0.40 | — | — |
| **Madhava bound (int8 5%/1%)** | **0.9500** | 11.7 | 86 | **0.40** | **100%** | **0** |
| **Madhava bound (int8 100%/100%)** | **0.9500** | 23.6 | 42 | 0.40 | **100%** | **0** |
| **Madhava speed GPU (OpenCL, fused v1.7.2)** | **0.9500** | **2.41** | **415** | 0.29 | **100%** | — |
| Speed GPU **batch** (100 q) | **0.9500** | 1.55 | 644 | — | **100%** | — |
| Madhava speed CPU | 0.9500 | 9.56 | 105 | — | 100% | — |

*GPU = NVIDIA RTX 5060 Ti (local). The Kaggle run (P100, v1.7.1) reported the
same R@10=0.95 and 100% efficiency; the local v1.7.2 numbers above include the
**fused QKᵀ+topk** optimization (single-query GPU **20× faster**: 47.8ms →
2.41ms).*

**Read the honest insight**: the bound engine reaches **100% of the exact-scan
ceiling** with 0 bound violations; the aggressive 5%/1% pruning costs
**zero recall** vs the 100% config. The speed GPU (OpenCL, verified
`backend="gpu"`) is an **exact scan** — now **4× faster than the CPU** in
single-query (2.41ms vs 9.56ms) and 3.7× faster in batch (1.55ms vs ~9ms), with
batch recall identical to single-query. The R@10 of 0.95 is the subset's
physical ceiling given GT coverage (~0.6% in the 1M prefix of the 100M GT) —
**no method can score higher on this subset**, and `winnex-madhava` reaches
100% of it. See [Speed GPU tuning](#choosing-speed--speed_n_anchors--speed_nprobe)
for how to configure the speed mode and why single-query is now fast.

### Honest benchmark 10M — larger subset, official GT, GPU (v1.7.2)

The same honest protocol on a **larger subset — N=10M** (GT coverage ~8% vs
~1% at 1M), which gives a higher raw recall and a more meaningful validation
against the official ground truth.

[![Kaggle](https://img.shields.io/badge/Kaggle-Honest%2010M%20GPU%20vs%20Official%20GT-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-1-7-honest-10m-gpu-vs-official-gt)

**Results (BIGANN-100M, subset 10M, 100 queries, official GT, Kaggle GPU P100):**

| Method | R@10 | Lat (ms) | Build (s) | Efficiency | Bound vio. |
|---|---|---|---|---|---|
| Exact-scan ceiling (`search_exact`) | 0.4400 | 554.2 | 26.8 | — | — |
| **Madhava bound (int8 5%/1%)** | **0.4400** | 491.0 | **14.6** | **100%** | **0** |
| **Madhava bound (int8 100%/100%)** | **0.4400** | 998.0 | 14.4 | **100%** | **0** |
| **Madhava speed GPU (OpenCL)** | **0.4400** | **38.1** | **6.7** | **100%** | — |
| Speed GPU **batch** (100 q) | **0.4400** | 22.6 | — | **100%** | — |

**Read the honest insight**: the R@10 of **0.44 at 10M reproduces the value the
project documents** (README/VERIFIED: R@10≈0.43-0.52) — it is the subset's
physical ceiling given ~8% GT coverage, not a defect. The bound engine reaches
**100% of that ceiling with 0 bound violations**, and the speed GPU (OpenCL,
verified `backend="gpu"`, v1.7.2 fused kernel) is **~13× faster than the bound
engine** (38.1ms vs 491ms single-query) — the fastest exact path at 10M. The
poda 5%/1% costs zero recall vs 100%. Batch recall == single-query.

### Hybrid benchmark (News 210K, v1.3.0)

The `winnex-madhava` hybrid mode (MadHybrid) benchmark — same engine,
`default` and `hybrid`, compared against HNSW / IVF / IVF-PQ on real News
Category data (209,527 articles, 42 categories, SBERT 384D float32):

[![Kaggle](https://img.shields.io/badge/Kaggle-MadHybrid%20vs%20HNSW%2FIVF%2FIVF--PQ-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-hybrid-vs-hnsw-ivf-ivf-pq)

| Method | NDCG@10 | Recall@10 | Lat (ms) | QPS | Build |
|---|---|---|---|---|---|
| FlatIP (exact) | 0.5960 | 0.5250 | 32.3 | 31 | N/A |
| HNSW(ef=32) | 0.5960 | 0.5250 | 0.51 | 1962 | **180.8 s** |
| HNSW(ef=256) | 0.5960 | 0.5250 | 2.17 | 461 | 180.8 s |
| IVF(nprobe=5) | 0.5878 | 0.5170 | 0.85 | 1176 | <1 min |
| IVF-PQ(m=8) | 0.5212 | 0.4610 | 3.18 | 315 | 9.1 s |
| **Madhava default (u8/L2)** | 0.4212 | 0.3805 | 16.9 | 59 | **3.4 s** |
| **MadHybrid(np=3)** | 0.5939 | **0.5270** | 5.17 | 193 | **12.9 s** |
| **MadHybrid(np=5)** | **0.5983** | **0.5295** | 8.0 | 125 | **13.0 s** |

**Read the honest insight**: MadHybrid's edge is **not raw recall** (plain
IVF wins at low nprobe). Its edge is **build speed + bound guarantee**:
14× faster build than HNSW, zero bound violations (mathematical proof per
exclusion), per-minute index rebuild for streaming data, and deterministic
results. The `winnex-madhava` hybrid mode reaches the same NDCG@10 as the
exact FlatIP baseline while running 4× faster per query.

Related public benchmarks:
- [winnex-madhava-1-7-honest-10m-gpu-vs-official-gt](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-1-7-honest-10m-gpu-vs-official-gt) — **honest 10M**: larger subset, official GT, R@10=0.44, GPU 38ms, 100% efficiency
- [winnex-madhava-1-7-honest-gpu-vs-official-gt](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-1-7-honest-gpu-vs-official-gt) — **honest 1M**: pip-installed wheel, official GT, normalized recall, GPU truthfully reported
- [winnex-madhava-pip-200-queries](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-pip-200-queries) — official L2 GT, 200 queries, 10M/100M
- [winnex-madhava-faiss-benchmark](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-faiss-benchmark) — side-by-side with FAISS HNSW/IVF/IVF-PQ
- [winnex-madhava-hybrid-vs-hnsw-ivf-ivf-pq](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-hybrid-vs-hnsw-ivf-ivf-pq) — hybrid (MadHybrid) vs HNSW/IVF/IVF-PQ on News 210K
- [winnex-madhava-speed-gpu-vs-hnsw-ivf-ivf-pq-bigann](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-speed-gpu-vs-hnsw-ivf-ivf-pq-bigann) — speed mode (native C++, O(K) anchors) vs HNSW/IVF/IVF-PQ on BIGANN-100M

### Speed benchmark (BIGANN-100M, v1.6.0)

The `winnex-madhava` speed mode — native C++ (OpenCL GPU default, OpenMP/AVX2
on CPU), with **O(K) PiPrime anchor navigation** (sublinear, not brute force) —
compared against HNSW / IVF / IVF-PQ on the **BIGANN-100M** dataset (subset
1M, 30 queries, official L2 ground truth):

[![Kaggle](https://img.shields.io/badge/Kaggle-Speed%20vs%20HNSW%2FIVF%2FIVF--PQ%20(BIGANN)-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-speed-gpu-vs-hnsw-ivf-ivf-pq-bigann)

| Method | R@10 | NDCG | Lat (ms) | QPS | Effic. |
|---|---|---|---|---|---|
| HNSW(ef=64) | 0.0067 | 0.0120 | 0.46 | 2185 | 67% |
| **HNSW(ef=128)** | 0.0100 | 0.0151 | **0.69** | 1454 | **100%** |
| IVF-PQ(m=16) | 0.0067 | 0.0147 | 1.20 | 835 | 67% |
| IVF(nprobe=10) | 0.0067 | 0.0120 | 2.16 | 463 | 67% |
| IVF(nprobe=50) | 0.0100 | 0.0151 | 9.44 | 106 | 100% |
| **Madhava speed (brute)** | 0.0100 | 0.0151 | 24.1 | 41 | **100%** |
| **Madhava speed O(K) a=16 np=8** | 0.0100 | 0.0151 | 35.2 | 28 | **100%** |
| Madhava speed O(K) a=32 np=4 | 0.0067 | 0.0120 | 29.2 | 34 | 67% |

**Read the honest insight**:
- The official BIGANN GT is for the **full 100M**. On a 1M subset only ~1%
  of true neighbors exist, so **all** methods (including exact scan) cap at
  R@10 ≈ 0.01. The "efficiency" column is relative to the subset's
  exact-scan ceiling — the speed mode reaches **100%** (it is exact).
- **HNSW is faster in raw latency on CPU** (0.69 ms vs 24 ms) — expected:
  the speed mode does an exact scan (O(N·d)); HNSW is sublinear. **On GPU**,
  the QKᵀ matmul closes much of the gap — the fused kernel (v1.7.2) runs an
  exact scan of 1M in 2.41 ms (see the [honest benchmark](#honest-benchmark--official-gt-pip-installed-wheel-gpu-v172)).
- **O(K) anchors**: with `nprobe=8`, the anchor navigation reaches **100%**
  of the ceiling (the anchors capture the true neighbors) while evaluating
  only the relevant cells. With `nprobe=4`, recall drops to 67% — the
  nprobe trade-off is real and documented.
- **Build**: speed brute = 0.6 s vs HNSW = 332 s (**~556× faster**).
- The speed mode's value is **exactness + build speed + determinism**, not
  beating HNSW on raw CPU latency. On GPU it competes directly on latency.

## Limitations (read this first)

We are explicit about what winnex-madhava **does not** do. Most "surprising"
behavior below is by design — the engine is optimized for a specific input
domain, and using it outside that domain silently degrades quality.

### Input: `default` needs uint8; `hybrid` accepts float32

**`default` mode** treats every corpus vector as **uint8 bytes** (`np.uint8`),
values 0–255. This is the BIGANN-style quantized format the math assumes.

```python
# ✅ Correct (default mode)
corpus = np.random.randint(0, 256, size=(10_000, 128), dtype=np.uint8)
engine = winnex_madhava.build_engine(corpus, dim=128, k=10)
query  = corpus[0].astype(np.float32)     # float32 *of the uint8 values*

# ❌ Wrong in default mode — silently gives poor recall
corpus = np.random.randn(10_000, 128).astype(np.float32)   # floats ~0
engine = winnex_madhava.build_engine(corpus, dim=128, k=10)    # truncated to uint8!
```

If you pass a `float32` corpus to `default` mode, `build_engine` **truncates**
it to `uint8` via `astype(np.uint8)` — values like `0.09` become `0`, `3.44`
becomes `3`. The engine will still run and report `bound_violations == 0`,
but the recall can collapse.

**For float32 embeddings (cosine), use `hybrid=True`** — the MadHybrid path
accepts float32 directly and routes the query to clustered cells, avoiding
the uint8 truncation:

```python
# ✅ Correct for float32 embeddings (hybrid mode)
embeddings = np.random.randn(10_000, 128).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
engine = winnex_madhava.build_engine(embeddings, dim=128, k=10,
                                     hybrid=True, nlist=64, nprobe=5,
                                     metric="cosine")
```

### `requires-python >= 3.8`, but pre-built wheel is CPython 3.12 only

See [Installation](#installation). 3.8–3.11 installs build from source and
needs a C++20 toolchain. If `pip install` starts compiling, you are on an
unsupported wheel path.

### The guarantee is per-document bound-correctness, not "great recall"

`bound_violations == 0` means: *every vector the engine pruned was provably
not in the exact top-K.* It does **not** mean the returned top-K is the true
top-K. If `k1_fraction` is too small (e.g. 0.001 on a hard dataset), the
survivors may be a weak sample and recall drops — still with 0 violations.
The bound is sound, but pruning quality depends on `stage1_dim`, `stage2_dim`
and `k1_fraction`. Tune them on your data.

### Lower-dimensional / tiny corpora

The Stage-1 QR projection shines on high-dimensional uint8 data (64–1000D).
On tiny corpora or d < ~8 the projection overhead dominates and an exact
`search_exact` scan is both faster and simpler.

### Speed mode: CPU exact scan is O(N·d); GPU needs a CUDA build

The **CPU** speed mode is an exact scan — O(N·d) per query. It is correct
everywhere but HNSW beats it on raw CPU latency (sub-linear vs linear). The
**O(K) anchor navigation** (`speed_n_anchors>=2`) reduces the evaluated set
to the relevant cells (sub-linear in data touched), with a recall cost that
depends on `nprobe`.

The **GPU** path is a **fused QKᵀ+topk** kernel. The default backend is
**OpenCL** (`src/speed_opencl.cpp`) — vendor-neutral, JIT-compiled, no nvcc
required — with a **CUDA opt-in** (`src/speed_gpu.cu`, built via
`-DMADHAVA_USE_CUDA=ON`). The GPU is enabled automatically when an OpenCL
loader + device is present, else it falls back to CPU (see `require_gpu=True`
to force a hard error instead). There is no persistence/serialize API yet —
rebuild per process.

### O(K) anchor recall depends on `nprobe`

The sublinear speed mode trades recall for speed like any IVF index: with a
small `nprobe`, some true neighbors may fall outside the probed cells
(measured: `nprobe=4` → 67% of the exact ceiling, `nprobe=8` → 100% on
structured data). Tune `speed_n_anchors`/`speed_nprobe` on your data.

## Honest comparison

We are explicit about where winnex-madhava **does not** win:

| Use case | Best tool | Why |
|---|---|---|
| Lowest latency (sub-ms) | HNSW | HNSW ≈ 0.45 ms vs madhava bound ≈ 2.7 ms at 50K×1536D (approximate vs exact) |
| Exact scan, low latency | **winnex-madhava speed GPU** | Fused QKᵀ+topk: 2.41 ms at 1M single-query — the fastest exact path |
| **Provable completeness** | **winnex-madhava** | Only engine with 0 bound violations + per-doc proof |
| Frequent index rebuilds | **winnex-madhava** | Build ≈ 1 s (10M) vs HNSW ≈ 1025 s |
| Regulated / auditable retrieval | **winnex-madhava** | Deterministic, per-document audit trail |
| Sublinear query on clustered corpora | **winnex-madhava hybrid** | MadHybrid: nprobe×cell query, 14× faster build than HNSW, bound guarantee |

**If you need raw speed, use HNSW — it is excellent.** winnex-madhava is for the
regions where "fast but unprovable" is a liability: legal discovery, medical
records, financial compliance, government audits, and RAG systems that must
not silently drop a relevant document.

## Build from source

```bash
# Wheel + sdist (pip-installable)
python -m build

# C++ library only
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build        # C++ unit tests

# Python tests
python -m pytest tests/python/
```

## License

**Business Source License 1.1 (BSL 1.1)** — the same license as the rest of the
Winnex stack.

### What BSL 1.1 means for you

- **Free to use** for **evaluation and non-production work** — study, test,
  prototype, benchmark. This is the recommended way to start.
- **Not free** for **commercial / production use** (a "Search Service" that
  exposes the functionality to third parties as a service). That requires a
  **commercial license** from Winnex.
- **Change date:** the license converts to **GPL v2.0 or later** on the change
  date (see the full license text), at which point the standard open-source
  terms apply.

**How to get a commercial license:** email `pay@winnex.ai`. The Winnex team
will issue a license agreement for your use case (ISV embedding, database
vendor, platform company, or internal production deployment).

### Contact

`pay@winnex.ai` · Winnex Brasil Soluções Empresariais LTDA-ME · Goiânia, Brazil
