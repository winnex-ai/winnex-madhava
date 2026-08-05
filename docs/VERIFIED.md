# Verification — winnex-madhava L2 engine

**Status: VERIFIED** — reproduced on the official BIGANN-100M L2 ground truth.

This document is the evidence backing the README claims. Every number below
was produced by running the code in this repository (or the C++ benchmark it
ships), not by extrapolation.

---

## 1. BIGANN-100M L2 subset (official ground truth)

Verified 2026-08-04 on the official BIGANN-100M L2 ground truth, 200 queries,
CPU-only (28 threads, AVX2+FMA), robust recall definition
(`|result[:K] ∩ GT| / min(K, |GT|)`).

| Scale | `exact_scan` ceiling (R@10) | `winnex-madhava` (R@10) | NDCG | Bound vio. | Efficiency |
|-------|------------------------------|--------------------------|------|------------|------------|
| 10M | 0.5225 | **0.5225** | 0.5796 | 0 | **100%** |
| 100M | 0.8360 | **0.8360** | 0.8611 | 0 | **100%** |

**Efficiency vs ceiling: 100.0%** — the bound+post-filter recovers exactly the
recall of a perfect exhaustive scan over the same subset, with 0 bound
violations. Runs: [winnex-madhava-pip-200-queries](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-pip-200-queries).

### Why is the ceiling not 1.0?

The official BIGANN L2 ground truth was generated in the **full 1B space**. On
a subset, the exact top-K by L2² differs from the 1B ground truth, so even a
perfect exhaustive scan caps at R@10 ≈ 0.52 (10M) and 0.84 (100M). **No index —
exact or approximate — can score higher on this subset against this ground
truth.** winnex-madhava reaches **100% of that ceiling** at both scales.

---

## 1b. Streaming at 100M (v1.2.0)

Verified on Kaggle (notebook `winnex-madhava-stream-100m`), 4 CPUs, V3-style
cosine (`metric='cosine'`, `normalize_input=True`, `k2_max=2000`), corpus
memory-mapped (12.8 GB never loaded into RAM):

| Scale | Build (s) | Lat (ms) | R@10 | NDCG | RSS (GB) | Vio |
|---|---|---|---|---|---|---|
| 100K | 1.3 | 7.5 | 0.750 | 0.658 | 0.4 | 0 |
| 1M | 5.9 | 66 | 0.843 | 0.667 | 0.8 | 0 |
| 10M | 34.0 | 698 | 0.501 | 0.544 | 3.8 | 0 |
| **100M** | **342.6** | **7592** | **0.780** | **0.813** | **31.4** | **0** |

**100M indexed in 342.6 s** (~5.7 min) via mmap, 0 bound violations.

---

## 2. Reproduce it yourself

The package ships a benchmark CLI:

```bash
pip install winnex-madhava
python -m winnex_madhava.benchmark --n 10000000 --nq 50
```

Or via the C++ executable:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMADHAVA_BUILD_BENCHMARK=ON
cmake --build build -j
./build/winnex_madhava_bench bigann_data/base.u8bin \
    bigann_data/unif_query_10k.u8bin \
    bigann_data/unif_groundtruth_10k.bin 100000000 50 0.05
```

---

## 3. Unit tests (CI)

The test suite runs on every push via GitHub Actions (`ci.yml`):

**Python (`tests/python/test_python_api.py`)** — 9 tests:
- build + search returns the configured `k` results
- query equal to a corpus vector is its own top-1 (L2)
- `search_exact` matches a brute-force L2 scan
- cosine + cascade `[stage1, stage2]` + modulation: query is its own top-1
- `quant='none'` (float32) matches exact scan
- `recall_at_k` / `ndcg_at_k` metric helpers return 1.0 on a perfect list
- robust recall: |gt| < K → a perfect result still scores 1.0
- robust recall: intersects with the entire relevant set, not just top-K GT
- `benchmark_vs_groundtruth` integrates the GT mapping

**C++ (`tests/test_winnex_madhava.cpp`)**:
- Cauchy-Schwarz bound: 0 violations on a random 10K corpus
- Post-filter: recovers the exact-scan top-K
- Metric helpers: perfect list → R@10 = NDCG@10 = 1.0

---

## 4. The guarantee, stated precisely

`bound_violations == 0` means: *every vector the engine pruned was provably not
in the exact top-K* (by the Cauchy-Schwarz inequality). It is **not** a claim of
high recall on arbitrary data — recall depends on `stage1_dim` / `k1_fraction`
and the data's latent structure. The bound is always sound; the pruning quality
is tunable. See the README's [Limitations](../README.md#limitations-read-this-first).

---

*Verified by running the shipped code. Hardware/date noted per benchmark run.*
