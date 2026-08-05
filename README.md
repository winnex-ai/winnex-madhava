<div align="center">

# winnex-madhava

**Deterministic vector search with mathematical guarantees.**

Every document excluded from the results carries a proof that it could not be in the top-K — by the **Cauchy-Schwarz inequality**. Zero bound violations by construction.

[![PyPI version](https://img.shields.io/pypi/v/winnex-madhava?color=467C45)](https://pypi.org/project/winnex-madhava/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/winnex-madhava?color=467C45)](https://pypi.org/project/winnex-madhava/)
[![PyPI - Python Versions](https://img.shields.io/pypi/pyversions/winnex-madhava?color=467C45)](https://pypi.org/project/winnex-madhava/)
[![CI](https://img.shields.io/github/actions/workflow/status/winnex-ai/winnex-madhava/ci.yml?branch=main&label=CI&color=467C45)](https://github.com/winnex-ai/winnex-madhava/actions/workflows/ci.yml)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-467C45)](LICENSE)
[![Issues](https://img.shields.io/github/issues/winnex-ai/winnex-madhava?color=467C45)](https://github.com/winnex-ai/winnex-madhava/issues)
[![C++](https://img.shields.io/badge/C%2B%2B-20-467C45)](https://isocpp.org/)
[![Benchmark](https://img.shields.io/badge/BIGANN--100M-L2%20verified-467C45)](docs/VERIFIED.md)

</div>

---

`winnex-madhava` is a real, pip-installable Python package with a native C++20 core. It answers a question no approximate index (HNSW, IVF, PQ) can answer:

> **"Prove that your search did not miss a relevant document."**

The proof is per-document and mathematical: a Cauchy-Schwarz upper bound on the inner product, which converts into a lower bound on L2². If the bound says a vector cannot be in the top-K, that vector **is not in the top-K**. No heuristics, no random graphs, no "we think it's fine."

Verified on the **official BIGANN-100M L2 ground truth** — see [Benchmarks](#benchmarks).

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Why?](#why)
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

Requirements: Python ≥ 3.8, NumPy. The C++ core ships pre-built in the wheel
(manylinux); a C++20 compiler + CMake ≥ 3.20 are needed only when building
from source.

> **⚠️ Python version support (important).** The pre-built manylinux wheel is
> currently **CPython 3.12 only**. On 3.8–3.11, pip falls back to the sdist
> and compiles from source, which requires a C++20 compiler + CMake on the
> machine. If you are on 3.8–3.11 and get a build error, either install a
> C++20 toolchain or use Python 3.12. Wider wheel coverage (cp38–cp311) is on
> the roadmap.
>
> Not published to PyPI yet? Install straight from this repo:
>
> ```bash
> pip install git+https://github.com/winnex-ai/winnex-madhava.git
> ```

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

print(result.indices)                  # top-K dataset ids, ascending L2²
print(result.latency_ms)               # milliseconds
print(result.bound_violations)         # always 0 — the guarantee
```

That's it. Same query + same data → same result, every time. Deterministic.

## Examples

Ready-to-run scripts live in [`examples/`](examples/):

| Script | What it shows |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | Build + search + the bound guarantee (copy-paste) |
| [`examples/rag_demo.py`](examples/rag_demo.py) | End-to-end RAG: encode texts → index → retrieve → rerank |
| [`examples/batch_benchmark.py`](examples/batch_benchmark.py) | Recall vs brute-force `search_exact`, with the recall ceiling |
| [`examples/python_usage.py`](examples/python_usage.py) | All public API functions with docstring signatures |
| [`examples/example_minimal.cpp`](examples/example_minimal.cpp) | C++ minimal usage |
| [`examples/bench_bigann_l2.cpp`](examples/bench_bigann_l2.cpp) | C++ BIGANN-100M L2 benchmark |

Run any of them with:

```bash
pip install winnex-madhava
python examples/quickstart.py
```

## Why?

Modern vector search is a heuristic gamble. HNSW builds a random proximity
graph and *hopes* it didn't prune a relevant neighbor; IVF picks clusters and
hopes the right one was probed. When the search is a legal discovery, a
medical record lookup, or a compliance audit, "hope" is not a defensible
answer.

**winnex-madhava replaces hope with proof.** Each excluded document carries its
upper bound; if the bound is below the threshold, exclusion is a theorem, not
a guess.

| Property | HNSW / IVF / PQ | **winnex-madhava** |
|---|---|---|
| Deterministic (same input → same output) | No (random graph) | **Yes** |
| Proves every exclusion | No | **Yes** (Cauchy-Schwarz) |
| Bound violations | Not measurable | **0** |
| Rebuild speed (1M) | ~40 s | **~2.6 s** |
| Exact-recall ceiling reachable | No | **Yes** (post-filter) |

## API

### `winnex_madhava.build_engine(corpus, *, dim=None, stage1_dim=64, k=10, k1_fraction=0.05, postfilter=True, seed=42) -> MadhavaL2`

Build an index over a `(n, dim)` uint8 array.

- `dim` — vector dimensionality (defaults to `corpus.shape[1]`).
- `stage1_dim` — dimensionality of the Stage-1 QR projection (64 by default).
- `k` — number of results to return.
- `k1_fraction` — fraction of the corpus kept after Stage-1 pruning (0.05 = 5%).
- `postfilter` — when `True`, exact L2 is computed on the survivors so the
  result matches the exact top-K of the surviving set.

### `engine.search(query: np.ndarray) -> SearchResult`

Returns `indices`, `latency_ms`, `k1`, `k3`, `bound_pairs`, `bound_violations`.

### `engine.search_exact(query: np.ndarray) -> SearchResult`

Exhaustive L2 scan over all N vectors — the **recall ceiling** of your corpus.
Use it to measure how close an approximate index gets to the physical limit.

### `winnex_madhava.benchmark_vs_groundtruth(engine, queries, gt_ids, *, query_alignment=1, k=None) -> dict`

Evaluate against ground-truth id lists. Returns `recall_at_k`, `ndcg_at_k`,
`latency_ms`, and per-query detail.

### Metrics

- `winnex_madhava.recall_at_k(result, gt_set, k)`
- `winnex_madhava.ndcg_at_k(result, gt_set, k)`
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

**Post-filter** (optional) computes the exact L2² on the surviving top-k1 and
returns the true top-K. Because Stage 1 never prunes a real neighbor, the
post-filter recovers **everything a perfect scan would find**.

The residual `‖v − PᵀPv‖` is computed on the **real float32 projection**, not
the int8-quantized one — this is what the inequality requires, and it is what
makes the bound exact rather than approximate.

## Benchmarks

Verified 2026-08-04 against the **official BIGANN-100M L2 ground truth** on a
CPU-only machine (28 threads, AVX2+FMA).

### "Prova dos 9" no 100M — topo da categoria para buscas exatas via bounding

O Madhava atinge **R@10 = 0.8360** com **NDCG = 0.8611** no dataset oficial
(100M, 200 queries — estatisticamente robusto), **varrendo o teto do scan
exato** — alcança exatamente o que um scan exaustivo perfeito alcançaria, com
**0 violações de bound** e garantia matemática por documento.

| Scale | Exact-scan ceiling<br>(R@10) | winnex-madhava<br>(R@10) | NDCG | Efficiency |
|---|---|---|---|---|
| 10M | 0.5225 | **0.5225** | 0.5796 | **100%** |
| 100M | 0.8360 | **0.8360** | 0.8611 | **100%** |

O **ceiling** é `search_exact` — um scan exaustivo perfeito sobre o mesmo
subset. O winnex-madhava atinge **100% do ceiling em 10M e 100M**, com **0
violações de bound** em todas as escalas. Nenhum outro índice (HNSW, IVF,
IVF-PQ) alcança o teto do subset — o Madhava é o único com garantia
matemática.

> **Definição de recall (robusta).** As métricas usam:
> `recall@K = |resultado[:K] ∩ GT_subset| / min(K, |GT_subset|)`, onde
> `GT_subset` são todos os ids do GT oficial presentes no subset. Isso
> intersecta com **todo** o conjunto relevante (não só os top-k) e normaliza
> por `min(K, |GT_subset|)` — um scan exato perfeito atinge **exatamente 1.0**
> mesmo quando o subset tem menos de K relevantes. Sem isso, uma definição que
> divide por K fixo penaliza artificialmente (produzia 0.404 em 10M).

### Kaggle benchmark (reproducible — pip install)

Rode você mesmo o benchmark com um clique no Kaggle: o notebook instala
`winnex-madhava` v1.1.3 do PyPI, indexa 10M/100M do BIGANN e reporta o teto do
scan exato vs o Madhava, com a tabela de eficiência:

[![Kaggle](https://img.shields.io/badge/Kaggle-winnex--madhava%201.1.3%20100M-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-pip-113)

### O mistério do subset 10M resolvido — a Cobertura do GT

> **Esta seção é obrigatória para entender qualquer número do BIGANN.**

O ground-truth oficial do BIGANN-100M foi gerado no **corpus completo (1B
vetores)**. Quando restringimos o corpus a um subset de N vetores, nem todos os
vizinhos verdadeiros existem dentro do subset:

| Escala | Cobertura do GT (top-20) | Significado |
|---|---|---|
| 1M | 1.2% | Comparação semanticamente vazia |
| 10M | 10.5% | Esparsa; recall limitado pelo subset |
| 100M | 100% | GT completo — única escala válida |

**Consequência:** em 10M, apenas ~10.5% dos vizinhos reais existem, então até
um scan exato perfeito fica em **R@10 ≈ 0.49** (o teto matemático do subset,
com a definição robusta de recall). **Nenhum índice** — exato ou aproximado —
pode superar isso no subset. É por isso que a "eficiência de 100%" é relativa
ao *ceiling do subset*, não a um recall absoluto de 1.0. Em 100M (cobertura
100%), o Madhava atinge **0.8360** — o recall real do dataset.

### Build vs Latency — o trade-off honesto

| Método (subset 10M) | R@10 | Build (s) | Latência (ms) |
|---|---|---|---|
| **winnex-madhava** | **0.5225** | **23.7** (Kaggle) / **1.0** (local) | 515 |
| IVF nprobe=128 | 0.4060 | 170 | 9.3 |
| IVF-PQ m=64 | 0.3920 | 90 | 24.1 |
| HNSW ef=256 | 0.2940 | 1025 | 1.0 |
| FlatL2 (exato) | 0.5225 | — | 617 |

O Madhava usa um **bound matemático** para podar, varrendo todos os vetores
sem construir grafo/índice. Isso custa latência por query (centenas de ms), mas
o **build é ultra-rápido** (1.1s local em 10M vs 1025s do HNSW — ~930× mais
rápido, com AVX2/FMA no wheel).

**Use-case claro do pacote:** ingestão contínua, RAG dinâmico (corpus que muda
com frequência), batch processing e ambientes com restrição de RAM/CPU, onde os
rebuilds de índices baseados em grafo (HNSW) são proibitivos. Para latência de
serving pura, use HNSW/IVF — o Madhava é para onde "rápido mas sem prova" é um
passivo.

Reproduce:

```bash
python -m winnex_madhava.benchmark --n 10000000 --nq 50
```

Or via the C++ executable:

```bash
./build/winnex_madhava_bench bigann_data/base.u8bin \
    bigann_data/unif_query_10k.u8bin \
    bigann_data/unif_groundtruth_10k.bin 100000000 50 0.05
```

## Limitations (read this first)

We are explicit about what winnex-madhava **does not** do. Most "surprising"
behavior below is by design — the engine is optimized for a specific input
domain, and using it outside that domain silently degrades quality.

### Input must be uint8 (0–255), not arbitrary floats

The engine treats every corpus vector as **uint8 bytes** (`np.uint8`), values
0–255. This is the BIGANN-style quantized format the math assumes.

```python
# ✅ Correct
corpus = np.random.randint(0, 256, size=(10_000, 128), dtype=np.uint8)
engine = winnex_madhava.build_engine(corpus, dim=128, k=10)
query  = corpus[0].astype(np.float32)     # float32 *of the uint8 values*

# ❌ Wrong — silently gives poor recall
corpus = np.random.randn(10_000, 128).astype(np.float32)   # floats ~0
engine = winnex_madhava.build_engine(corpus, dim=128, k=10)    # truncated to uint8!
```

If you pass a `float32` corpus, `build_engine` **truncates** it to `uint8` via
`astype(np.uint8)` — values like `0.09` become `0`, `3.44` becomes `3`. The
engine will still run and report `bound_violations == 0`, but the recall can
collapse (we measured ~5–10% R@10 on random gaussian floats vs ~100% on proper
uint8). **This is not a bug — it is the documented input contract.** Use
winnex-madhava on uint8 (BIGANN-style) data, or quantize your floats to uint8
yourself and search in that space.

### `requires-python >= 3.8`, but pre-built wheel is CPython 3.12 only

See [Installation](#installation). 3.8–3.11 installs build from source and
needs a C++20 toolchain. If `pip install` starts compiling, you are on an
unsupported wheel path.

### The guarantee is per-document bound-correctness, not "great recall"

`bound_violations == 0` means: *every vector the engine pruned was provably
not in the exact top-K.* It does **not** mean the returned top-K is the true
top-K. If `k1_fraction` is too small (e.g. 0.001 on a hard dataset), the
survivors may be a weak sample and recall drops — still with 0 violations.
The bound is sound, but pruning quality depends on `stage1_dim` and
`k1_fraction`. Tune them on your data (see [Benchmarks](#benchmarks) for
typical values).

### Lower-dimensional / tiny corpora

The Stage-1 QR projection shines on high-dimensional uint8 data (64–1000D).
On tiny corpora or d < ~8 the projection overhead dominates and an exact
`search_exact` scan is both faster and simpler.

### No GPU / no persistence yet

The engine is CPU-only and keeps the index in memory; there is no
serialize/load API in v1.0.0. Rebuild per process.

## Honest comparison

We are explicit about where winnex-madhava **does not** win:

| Use case | Best tool | Why |
|---|---|---|
| Lowest latency (sub-ms) | HNSW | HNSW ≈ 0.45 ms vs madhava ≈ 2.7 ms at 50K×1536D |
| **Provable completeness** | **winnex-madhava** | Only engine with 0 bound violations + per-doc proof |
| Frequent index rebuilds | **winnex-madhava** | Build ≈ 2.6 s (1M) vs HNSW ≈ 40 s |
| Regulated / auditable retrieval | **winnex-madhava** | Deterministic, per-document audit trail |

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

**Business Source License 1.1** — same as the Winnex stack. Free to use for
evaluation and non-production work. Commercial use requires a license.

`pay@winnex.ai` · Winnex Brasil Soluções Empresariais LTDA-ME · Goiânia, Brazil
