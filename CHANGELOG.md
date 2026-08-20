# Changelog

All notable changes to `winnex-madhava` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.2] — 2026-08-20

### Added: `search_with_commitment` — the lightweight audit commitment path

Fixes the production audit-trail bottleneck: `search_audited` returned the
FULL O(N) certificate (measured **~2 MB per query** at d=1536 with a PCA
basis, because `max_audit_records` was ignored — every provably-excluded doc
got a record). For WORM-backed compliance flows (tracer-gov / tracer-med)
that payload is written to disk on every query — unsustainable at scale.

`search_with_commitment(query, k, max_sample=50)` returns a compact
`AuditCommitment` (~400–500 bytes regardless of corpus size, 99.98% smaller):

- `total_excluded_count` — how many docs the Cauchy-Schwarz bound PROVED
  outside the exact top-K (the raw mathematical fact).
- `global_threshold` — the exact score of the K-th result (the TRUE global
  threshold the motor decided with — mirrors the 1.9.1 witness fix).
- `sampled_records` — a DETERMINISTIC boundary-biased sample (up to
  `max_sample`) of excluded docs nearest the threshold, for spot-check audit.

**Memory: O(max_sample), NOT O(N)** — the motor never materializes the
excluded list. `max_sample` is honored exactly (verified: `max_sample=2` →
2 records for 15,540 exclusions).

**Consistency:** `total_excluded_count == search_audited.audit_excluded`
exactly (verified at d=1536, PCA basis, off-manifold query).

**Determinism:** the sample is seeded from the query hash + engine seed, so
the same query always yields the same sampled doc_ids (reproducible by an
auditor). All sampled docs satisfy `upper_bound < global_threshold`.

**Hybrid security model (responsibility split):** the commitment carries NO
internal hash and NO key. The Python compliance layer (`tracer-gov` /
`tracer-med` `core.commitment`) hashes the raw fields (hashlib) and signs them
with an **Ed25519** private key held OUTSIDE the C++ binary (env / KMS), then
stores the ~500-byte signed record in the WORM. This neutralizes the "lying
engine" attack: even a compromised C++ binary cannot forge past records —
it never holds the signing key.

## [1.9.1] — 2026-08-19

### Fix: the per-document audit certificate is now a WITNESS, not a judge

The 1.9.0 `search_audited` recomputed the Cauchy-Schwarz bounds AFTER the
search with a query residual derived from the raw query norm instead of the
motor's `qn_eff` (=1.0 for cosine+normalize). At d=1536 that diverged from
the motor and produced false "excluded" records — measured on the Kaggle
benchmark `winnex-madhava-1-9-0-honest`: **462/973 certificate violations on
arXiv d=1536** (GloVe/BIGANN were clean). A second, deeper bug: `pruned_by_bound`
used the post-filter **pool threshold** (heap[0] among the k1_fraction
candidates) instead of the **global K-th threshold** — so real top-K docs could
be marked "provably excluded" (measured: 9 of the top-10 in audit_ids).

Root-cause fixes (the audit now records the motor's exact decision, at the
moment it is made — it never recomputes a bound):

1. `search()` captures, at decision time, the per-doc `ub`, residual, and the
   **GLOBAL K-th threshold** (`audit_ids` / `audit_ubs` / `audit_residuals` /
   `audit_threshold` on `SearchResult`).
2. The pruning threshold is now `exact_score` of the K-th returned result —
   the true global threshold. No audit_id can be in the returned top-K.
   `audit_excluded == pruned_by_bound` exactly.
3. `search_audited` reads the captured witness — it never recomputes a bound.

**Validated** (all green): 32/32 generic scenarios (D∈{64,128,512,1536} ×
norm∈{0.3,1,5,50} × random/pca) with 0 certificate violations and exact
consistency; BIGANN real (d=128) recall 1.0, cert 0 viol; arXiv-simile
(d=1536, norm 56.6) cert 0 viol; 29/29 Python tests + ctest. New regression
test: `test_certificate_high_dim_nonunit_norm_query`.

## [1.9.0] — 2026-08-19

### Added: `search_audited` + `audit_json` — the mathematical proof per document

The motor now emits the **per-document Cauchy-Schwarz certificate** that the
winnex-audit-cpp spec and the tracer-gov/tracer-med compliance flows consume
(`GovAuditRecord`: `doc_id`, `true_cosine`, `upper_bound`, `threshold`,
`excluded`, `stage`).

- `search_audited(query, k, max_audit_records=500)` → the normal `SearchResult`
  plus a per-doc certificate. The math is 100% the motor's own (`ub_raw`,
  `residuals1`, `exact_score`) — nothing is reimplemented.
- `audit_json(query, k, max_audit_records=500)` → the certificate as JSON
  (the audit_json of winnex-audit-cpp).
- The certificate examines the `max_audit_records` docs nearest the top-K
  boundary + the top-K themselves (the tracer-gov default is 500).
- **Metric-correct**: cosine excludes when `UB < threshold`; L2 excludes when
  `L2²-lower-bound > threshold` (mirrors the motor's own `n_bound_pruned`).
- **Consistency proven**: with `max_audit_records >= N`, `audit_excluded`
  equals the motor's `pruned_by_bound` exactly.
- `search()` is unchanged — this is purely additive. 0 bound violations by
  construction, verified by 7 new Python tests (28 total, all passing).

### Added: explicit OpenCL loader config (no hardcoded vendor .so)

The speed-mode GPU backend now resolves the OpenCL loader **transparently and
configurably** — no hardcoded driver name:

- `build_engine(..., speed=True, speed_opencl_lib="<loader.so>")` (and
  `MadhavaSpeed(..., opencl_lib=...)`) let the caller pin the exact loader /
  driver `.so` (e.g. `"libOpenCL.so.1"`, `"libnvidia-opencl.so.1"`, or a full
  path). Empty (default) = the `WINNEX_OPENCL_LIB` env var, else the platform
  ICD loader.
- Every loader attempt is logged (`[Winnex Madhava] OpenCL loader resolved: ...`
  / `failed to load: ...`), and `gpu_reason()` reports the exact loader tried —
  the CPU fallback is always explainable, never a silent hardcoded cascade.
- Verified on this machine: the NVIDIA GPU (CUDA 12.9) was previously invisible
  to the engine because the generic ICD loader (`libOpenCL.so.1`) was not in
  the default `dlopen` path. With `speed_opencl_lib` pointing at the installed
  loader, the SpeedEngine enables the **GPU backend** (`OpenCL QK^T + device
  topk, JIT-compiled kernels`) and matches exact top-1.
- **Fix (per-engine loader, not global):** the OpenCL loader state was a
  process-global singleton — once any loader was loaded, a later `SpeedEngine`
  ignored its own `opencl_lib` config (even an invalid one with
  `require_gpu=True`). Now `load_opencl()` tracks which loader is loaded and
  reloads when a different one is requested, so each engine's config is
  authoritative. Verified: an invalid loader with `require_gpu=True` correctly
  raises after a valid loader was used in the same process.

## [1.8.8] — 2026-08-15

### Fix: prefilter clamp, residuals1 zeros at k=d, BIGANN space mismatch, drop internal solver

1. `pruned_by_prefilter` clamped to `max(0, N - exact_evals - pruned_by_bound)` —
   it could go negative because the survivors and the bound-proved set overlap.
2. `residuals1()`/`residuals2()` return an array of zeros when `e1_`/`e2_` is not
   populated (e.g. k >= d, projection complete) instead of an uninitialized array.
3. BIGANN space mismatch: with `normalize_input=true` and a float32 path, the
   streaming now normalizes v in-place before projecting, so
   `e(v) = sqrt(1 - ||Pv||^2)` is computed in the unit-norm space the PCA basis
   expects. Previously v held raw uint8-cast values, `pn1 >> 1` and e(v)
   collapsed to 0 (measured 0.0 -> 0.3307, the correct value).
4. `build_engine(basis="pca_corpus")` no longer runs the internal
   power-iteration PCA solver when the LAPACK base will be supplied via
   `set_basis` (the UB Width paper path). The internal solver was the ~100 s
   ingestion bottleneck and under-converged for the spectrum tail (13-36% of
   variance vs 94% for LAPACK). Measured: d=1536 pca_corpus build 103.9 s ->
   1.9 s, e(v) = 0.2414.

## [1.8.7] — 2026-08-15

### Docs: honest pruning breakdown + public Kaggle benchmark in the README

Added the `SearchResult.pruned_by_bound` / `pruned_by_prefilter` breakdown to
the README, with the public Kaggle benchmark on 3 real datasets (GloVe d=100,
BIGANN-100M d=128, arXiv OpenAI d=1536), package installed from PyPI:

- arXiv d=1536, random: pruned_by_bound = 0.0% (the wide bound proves nothing;
  the "95%" is the fixed k1_fraction cutoff — the truth is now exposed).
- arXiv d=1536, pca_corpus: pruned_by_bound = 80.5% (the tight bound proves
  80.5% outside top-10), recall 1.000, 0 violations.

Kernel: kleniopadilha/winnex-madhava-1-8-6-honest-breakdown.

## [1.8.6] — 2026-08-15

### Parametrizable `build_engine` — accepts any dataset dtype; LAPACK PCA basis

1. **Practical dtype routing.** `build_engine` now accepts float32 AND float64
   embeddings (both go through the float32 manifold via `build_float32`);
   only a genuine uint8 corpus (BIGANN) uses the `build_numpy` L2 path.
   Previously a float64 corpus was mis-detected as non-float and truncated to
   uint8, collapsing `e(v)` to 1.0.

2. **LAPACK PCA basis (the UB Width path).** `basis="pca_corpus"` computes the
   PCA basis with LAPACK (`numpy.eigh`) and supplies it via `set_basis` — per
   the WINNEX UB Width paper (DOI 10.5281/zenodo.21939495, §7). The earlier
   hand-rolled float32 power iteration under-converged for the spectrum tail
   (captured 13–36% of the variance vs 94% for LAPACK) and cost ~110 s at
   d=1536. LAPACK is exact, fast, and restores the bound's pruning power at
   high dimension.

**Measured at d=1536 (arXiv, 20K subset, 100 queries, top-10):**

| mode | build | e(v) | recall | pruned_by_bound |
|---|---|---|---|---|
| random | 0.6 s | 0.9265 | 1.000 | 0.0% |
| pca_corpus | 25.6 s | 0.2462 | 1.000 | 97.8% |

The random basis proves nothing (wide bound); the PCA-aligned basis proves
97.8% of the corpus is outside the top-10, at full recall.

## [1.8.5] — 2026-08-14

### Fix (P0): the UB Width engine was producing a degenerate basis — e(v)=1.0

Three independent bugs made `basis="pca_corpus"` inert at high dimension
(d = 1536): the residual `e(v) = √(1 − ‖Pv‖²)` stayed 1.0 (the bound never
pruned) and the reported recall was a fallback artifact of the exact
post-filter, not the bound. Root causes, each proved by measurement:

1. **The residual was computed over the wrong space.** `build_float32`
   populated `corpus_f32_` (the real float32 manifold) for the PCA basis, but
   the `build()` streaming pass evaluated `v` from the uint8 re-normalized
   `base_` — a space orthogonal to the basis (`‖P·v_u8‖²=0.001` vs
   `‖P·v_f32‖²=0.939`). Fixed: the residual now uses `corpus_f32_` when
   available.
2. **The covariance was centered, which removed the dominant direction.**
   For arXiv, `‖μ‖=0.87`, so centering collapsed the spectrum to λ₁=0.009
   (e(v)=0.86). The non-centered covariance `C = XᵀX/N` keeps the total
   energy (λ₁=0.76, e(v)=0.24) with the same 0-violation guarantee. Fixed:
   `build_pca_basis` no longer centers.
3. **The internal eigensolver was a fragile Jacobi** that under-converges at
   d=1536 in float32 (unreachable tolerance → null-subspace basis). Fixed:
   power iteration with deflation + MGS re-orthogonalization (the X-Factor's
   proven method), verified to recover λ₁=0.767 exactly.

Also fixed: `build_engine` no longer converts float32 embeddings to uint8 when
`basis="pca_corpus"` (that conversion destroyed the manifold). Validated:
`build_engine(corpus_f32, basis="pca_corpus")` now yields e(v)=0.2465,
recall@10=1.000, 0 violations, 95% pruning at d=1536.

### Docs: UB Width mode + public Kaggle benchmark in the README

Added the `basis="pca_corpus"` (UB Width) section to the README with the
public Kaggle benchmark on the real BIGANN-100M dataset
(`kleniopadilha/winnex-madhava-1-8-4-ub-width-bigann-v3`): recall@10 = 1.000,
0 bound violations, 95% pruning, for both UB Width and the random default at
d = 128, with an honest scope note (the UB Width advantage is at high
dimension, d = 1536 — see the WINNEX UB Width paper, DOI 10.5281/zenodo.21939495).

## [1.8.4] — 2026-08-14

### Fix (P0): `build_engine(basis="pca_corpus")` raised `NameError: BasisMode`

`build_engine` referenced `BasisMode` but the Python module did not import
it. `build_engine(corpus, basis="pca_corpus")` — the UB Width mode — crashed
with `NameError`. **Fixed**: `BasisMode` is now re-exported from the native
extension and added to `__all__`.

## [1.8.3] — 2026-08-14

### Fix (P0): `build_float32` truncated unit-norm embeddings to zero

`build_float32` converted float32 embeddings to uint8 with
`uint8 = clamp(v, 0, 255)`. For a unit-norm embedding (values in [-1, 1]) this
maps nearly all values to 0 — the engine then sees all-zero vectors and recall
collapses to 0. **Fixed**: the affine map `uint8 = (v + 1) * 127.5` preserves
the geometry.

### Fix (P0): `set_basis` desynchronized the basis and the cached projections

`set_basis` replaced the projection basis (`P1_`) but did not recompute the
per-vector projections (`pr1_f_`) that `ub_raw` uses. The bound then combined
the new basis (query side) with the old basis' projections (corpus side) — a
garbage bound, measured recall 0.05. **Fixed**: `set_basis` now recomputes the
projections and residuals consistently over the real float32 corpus.

## [1.8.2] — 2026-08-11

### Fix (P0): early_exit default for cosine — recall 1.0 → 0.10 in high dimension

The Python binding forced `early_exit=True` for cosine (auto). In high
dimensions (dim >= 384) the modulated bound does not order like the exact
score (the bound is ~19× wider than the real neighbour separation), so
early-exit stopped at the wrong candidate and recall dropped to 0.10.

**Fix:** the default is now `False` (safe): the exact post-filter runs, which
guarantees recall=1.0. Operators may enable early_exit explicitly when they
know the corpus/dimension is safe.

**Validated (E2E):** cosine dim=384, default config → recall@10 = 1.000
(was 0.10 with the old default).

## [1.8.1] — 2026-08-08

### Fix: streaming build allocates only the current batch (robust C++ transaction)

The build streaming pass allocated a **fixed CHUNK × dim** buffer
(500K × dim × 4 bytes — e.g. 2 GB for dim=1024) regardless of corpus size.
This OOM-crashed small corpora on memory-limited containers (the Maestro
backend has a 2 GB limit). Now it allocates only `nt × dim` per batch.

- Build of a 4-vector 1024d corpus: ~2 GB alloc → **~16 KB**.
- Enables dim=1024 (Qwen3-Embedding) inside the Maestro container.
- Keeps the streaming behavior for large corpora (chunked, memory-bounded).

## [1.8.0] — 2026-08-08

## [1.8.0] — 2026-08-08

### Motor para o Maestro — M4 (AVX2), M1 (batch), M2 (persistência)

Capacidades adicionadas para servir o Maestro (ERP semântico):

**M4 — AVX2/FMA nos hot loops (latência):**
- `dot_int8_scaled`: dot int8×scale×float vetorizado (16 int8/iter) no `ub_raw`.
- `l2_sq_avx`: L2² uint8×float vetorizado (16 uint8/iter) no `exact_score`.
- `exact_score` cosine vetorizado.
- **Medido:** `search_exact` 1M: 9.0ms → **3.87ms (2.3×)**. Corretude preservada
  (recall 1.0, 0 violações).

**M1 — `search_batch` no bound engine (batch RAG):**
- Processa N queries de uma vez, paralelizado cross-query (OpenMP, engine
  imutável → thread-safe). Retorna nq×k indices.
- **Medido:** 50 queries em 1M: serial 491ms → batch **86ms (5.7×)**.

**M2 — persistência do índice (`save_index`/`load_index`):**
- Serializa projeções int8 + scales + residuals + config (binário, mmap-friendly).
- O corpus bruto fica no disco (o chamador re-anexa o base).
- **Medido:** N=100K int8: save 6.8ms (20.9MB), load **4.5ms** vs rebuild ~200ms
  (**45× mais rápido**) — o ERP reinicia sem rebuild.

**Nota (bug pré-existente documentado):** o `early_exit=True` no caminho **L2**
degrada o recall (1.0 → 0.10). Não afeta produção (default L2 = off; cosine =
on e funciona). Correção em versão futura.

## [1.7.2] — 2026-08-08

### Speed GPU: fused QKᵀ+topk — single-query 20× mais rápido

Corrige o gargalo mais significativo do speed mode GPU: o single-query era mais
lento que o CPU (47.8ms vs 9.2ms no 1M). O kernel `qkt` original usava **1
work-group por query** (1 SM ativo de 32 → ~3% da GPU), **acesso não-coalescido**
(512B/item) e **materializava scores[N]** (2× tráfego de memória).

**Novo kernel `qkt_fused_topk`** (ver `docs/OTIMIZACAO_GPU_SINGLE_QUERY.md`):
- **M work-groups por query** (M=64) → todos os SMs ativos mesmo com nq=1.
- **Acesso coalescido**: work-items adjacentes leem vetores adjacentes do corpus.
- **Fusão QKᵀ+topk**: top-k local em registrador/local mem, sem `scores[N]`.
- **L2 correction aplicada no fused** — 4 kernels/query → 2.
- `topk_merge_scores` funde os M top-k locais → top-k global exato.

**Resultado (RTX 5060 Ti, 1M, L2, 100 queries):**

| N | Antes | Depois | Speedup |
|---|---|---|---|
| 100K | 5.05 ms | 0.53 ms | 9.5× |
| 500K | 24.05 ms | 1.39 ms | 17.3× |
| 1M | 47.77 ms | **2.40 ms** | **19.9×** |

- GPU vs CPU single-query (1M): **3.97× mais rápido** (antes 5× mais lento).
- Batch (1M): ~1.5-1.9 ms/query, ~600-640 QPS (estável).
- **Corretude preservada**: 20/20 L2 + 20/20 cosine vs brute-force numpy;
  100/100 queries uint8 vs scan exato; batch == single-query.

### 📊 Benchmark real — referência validada (Kaggle, P100)

**Correção de integridade (2026-08-08).** Auditoria rigorosa provou que o GT
oficial do dataset `shurangwu/bigann-100m` **não é utilizável com o seu base**:
o `base.u8bin` tem ordem de vetores diferente da ordem canônica do BIGANN, logo
os ids do GT apontam para vetores errados (verificado: 0/500 acertos no top-10
exato). Recalls medidos contra esse GT (incluindo os R@10 de 0.52/0.836/0.44
reportados anteriormente) **não são significativos** e devem ser descartados.

Novo notebook público (referência válida):
[winnex-madhava-1-7-real-benchmark-vs-hnsw-ivf-pq](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-1-7-real-benchmark-vs-hnsw-ivf-pq)
— usa o **scan exato local como teto** (recall vs vizinhos reais, independente
do GT), documenta a validação do GT em runtime, e compara com FAISS
HNSW/IVF/IVF-PQ no mesmo subset.

**Resultados válidos (subset 1M, 100 queries, Kaggle GPU P100):**

| Método | R@10 (vs teto exato) | Lat (ms) | Eficiência |
|---|---|---|---|
| Madhava bound (5%/1%) | 0.9960 | 45.8 | 100% (0 vio) |
| **Speed GPU (OpenCL v1.7.2)** | **1.0000** | **6.14** | **100%** |
| Batch GPU | 1.0000 | 3.09 | 100% |
| HNSW(ef=128) | 0.9760 | 0.56 | 98% |
| IVF(nlist=4000,np=50) | 0.9250 | 0.75 | 93% |
| IVF-PQ(nlist=4000) | 0.4780 | 0.23 | 48% |

**O Madhava recupera 99.6-100% do top-10 exato com 0 violações**, e o speed GPU
é exato (R@10=1.0) a 6.1ms — enquanto os baselines aproximados perdem recall
(48-98%).

## [1.7.1] — 2026-08-07

### Speed GPU: topk paralelo (divide-and-conquer) + QKᵀ com tiling

- **Topk paralelo**: substitui o insertion sort serial (1 work-group/query) por
  `topk_local` (M work-groups por query, cada um varre um chunk de N) + `topk_merge`
  (funde os top-k locais). O scan de N deixa de ser serial — M× mais paralelismo.
- **QKᵀ com tiling**: 1 work-group por query, query carregada em local memory e
  reutilizada (em vez de reler da memória global para cada vetor).
- Corretude preservada: **recall exato 50/50** (o merge dos top-k locais é
  matematicamente exato — nenhuma garantia do Madhava perdida).
- Observação honesta: o gargalo real do batch é o matmul QKᵀ (memory-bound, lê o
  corpus inteiro), não o topk. O roteamento O(K) por âncoras reduziria os bytes
  lidos mas é aproximado (trade-off recall×velocidade) — mantido como modo
  separado, não o default.

## [1.7.0] — 2026-08-07

### 🔊 OpenCL GPU backend (generic, vendor-neutral) + forced GPU

- **OpenCL is now the default GPU backend** (`src/speed_opencl.cpp`): the
  generic, vendor-neutral path that works on NVIDIA/AMD/Intel GPUs with **no
  offline compiler (nvcc) and no CUDA toolkit**. Kernels are JIT-compiled at
  runtime by the driver — the compute equivalent of how GLQuake used OpenGL
  instead of a proprietary toolkit, with a software fallback.
- CUDA (`src/speed_gpu.cu`) is now **opt-in** at build time via
  `-DMADHAVA_USE_CUDA=ON`; the default build uses OpenCL when the loader is
  present, else CPU.
- Backend selection is logged at configure time:
  `Madhava speed GPU backend: opencl|cuda|cpu`.
- Verified on an RTX 5060 Ti: QKᵀ + device topk matches brute-force exactly
  (20/20 top-1, top-k exact), and batch throughput (N=100k, 1000 queries) is
  1.77× faster than the CPU backend (74 ms vs 131 ms).
- New C++ test validates the GPU backend matches exact scan when `has_gpu()`.

### 🔊 Speed engine GPU/CPU transparency + forced GPU

- `SpeedEngine` (and `MadhavaSpeed` / `build_engine(speed=True)`) now log clearly
  to stderr when the GPU backend cannot be enabled and the engine falls back to
  CPU, with the exact reason (`build without CUDA` vs `no CUDA device found`).
- New `require_gpu=True` option: raises `RuntimeError` instead of silently
  degrading to the CPU backend when no usable GPU is present. Exposed at the
  C++ constructor, the `SpeedEngine` pybind, `MadhavaSpeed(..., require_gpu=True)`,
  and `build_engine(..., speed=True, require_gpu=True)`.
- New `SpeedEngine.backend_name()` (`"gpu"`/`"cpu"`) and `gpu_reason()` accessors
  for logs and diagnostics, mirrored in the Python bindings.
- Internals: the GPU enable hook is now the private method `SpeedEngine::enable_gpu`
  (no more free `_enable_gpu` that shadowed the CUDA definition), and the CPU
  stubs are guarded by `#ifndef MADHAVAS_HAS_CUDA` so a CUDA build links cleanly.

## [1.4.0] — 2026-08-07

### 🚀 Speed mode (GPU) — direct HNSW competitor

- `build_engine(..., speed=True)` returns a `MadhavaSpeed` GPU engine
- Uses the exact attention `Q Kᵀ` matmul as a batched exact scan
- float32 embeddings (cosine) or uint8 raw bytes (L2) with correct L2 score
- `search` (individual) and `search_batch` (throughput) with CUDA warmup
- optional-dependency: `pip install winnex-madhava[speed]` (torch)
- 5 correctness tests (vs brute-force), skipped without CUDA


## [1.3.1] — 2026-08-07

### 📝 README: hybrid benchmark results

- Document the News-210K MadHybrid benchmark table (real Kaggle results)
- Document hybrid/nlist/nprobe parameters + float32-vs-uint8 input contract
- Link the hybrid-vs-HNSW/IVF/IVF-PQ Kaggle notebook


## [1.3.0] — 2026-08-06

### Hybrid mode (MadHybrid)

`build_engine(..., hybrid=True, nlist=..., nprobe=...)` returns a clustered
MadHybrid wrapper: the same bound engine, partitioned into cells and queried
sublinearly. Supports **float32 embeddings** (cosine, pure-Python bound cells)
and **uint8 raw bytes** (L2, native C++ per cell). One motor, two modes.

- `MadHybrid` class + `_MadhavaCellPy` (per-cell bound cascade 32D→64D)
- `build_engine` gains `hybrid`, `nlist`, `nprobe` parameters
- Validated on News 210K / SBERT: recall@10 ≈ 1.0 at nprobe=3–8
- See README "Hybrid mode (MadHybrid)"


## [1.1.3] — 2026-08-05

### Métricas robustas (Recall@K / NDCG@K)

As métricas dividiam por `k=10` fixo e intersectavam apenas `gt[:k]`, o que
penalizava queries com poucos relevantes no subset (um scan exato perfeito podia
marcar < 1.0) e ignorava vizinhos relevantes além da posição k.

**Nova definição:**
```
recall@K = |result[:K] ∩ gt| / min(K, |gt|)
ndcg@K   = DCG / IDCG, com IDCG usando min(K, |gt|)
```

**Verificado (10M e 100M, GT oficial, Kaggle):**

> ⚠️ **Nota de correção (2026-08-08).** Estes números usaram o arquivo GT do
> dataset `shurangwu/bigann-100m`, que uma auditoria posterior provou ser
> **inválido para o base do mesmo dataset** (ordem de vetores diferente da
> canônica → ids do GT apontam para vetores errados). Ver
> [Benchmark real](#📊-benchmark-real--referência-validada-kaggle-p100) na
> v1.7.2. Os recalls absolutos abaixo **não são significativos**; a eficiência
> relativa (100% vs teto) permanece como propriedade do motor.

| Escala | R@10 | NDCG | Eficiência | Vio | Build |
|---|---|---|---|---|---|
| 10M | **0.4882** | 0.5393 | 100% | 0 | 27.7s |
| 100M | **0.7880** | 0.8208 | 100% | 0 | 227.5s |

Agora um scan exato perfeito atinge **exatamente 1.0** quando o GT tem poucos
relevantes, e os números são consistentes com runs anteriores (0.488/0.788).

### Benchmark Kaggle (pip install)

Novo notebook público: [winnex-madhava-pip-113](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-pip-113).

## [1.1.2] — 2026-08-05

### Corrige a definição de recall@K (padrão BIGANN)

O recall era calculado contra **todos** os ids do GT presentes no subset (até
100), inflando R@10 de 0.404 para 0.43 (10M) e de 0.354 para 0.788 (100M). O
padrão BIGANN usa apenas os **top-K ids oficiais** do GT.

**Valores corrigidos (verificados no Kaggle com o wheel v1.1.2):**

| Escala | R@10 | NDCG | Eficiência | Vio | Build |
|---|---|---|---|---|---|
| 10M | **0.4040** | 0.4755 | 100% | 0 | 24.5s |
| 100M | **0.3540** | 0.4344 | 100% | 0 | 219.9s |

O Madhava atinge **exatamente o teto do scan exato** em todas as escalas, com
0 violações de bound e garantia matemática por documento.

### Benchmark Kaggle (pip install)

Novo notebook público: [winnex-madhava-pip-112](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-pip-112).

## [1.1.1] — 2026-08-05

### 🚀 Build ultra-rápido (AVX2/FMA)

O wheel era compilado sem `-mavx2 -mfma`, fazendo o C++ cair no path scalar —
build 10M em ~52s. Com as flags SIMD: **build 10M = 1.0s** (medido no PyPI).

## [1.1.0] — 2026-08-04

### 🎯 Alinhamento com a stack Winnex AI

O motor agora reflete a arquitetura completa da stack Winnex (v17, Madhava-Sec,
HMC v7), tornando-se **parametrizável** com valores padrão:

- **Métrica**: `cosine` (padrão — embeddings normalizados, como a stack) ou
  `l2` (uint8 cru, BIGANN-style).
- **Cascata de 2 estágios**: `[stage1_dim, stage2_dim]` — bound largo B1 (64D)
  para poda inicial, bound justo B2 (128D) para poda final. Padrão `[64, 128]`.
- **Modulação error-backprop**: `modulation=True` — o ranking é feito por
  `B1 + α·(B2−B1)` com `α = sigmoid((e1−e2)/μ)`, mas a **poda usa SEMPRE o bound
  garantido** (B2). Este é o invariante FIX(1) do Madhava-Sec: modulação apenas
  para ranking, nunca para corte.
- **Quantização**: `int8` (padrão, memory-light — só armazena int8, viável em
  100M) ou `none` (float32 exato).
- **Normalização**: `normalize_input=True` para cosine.
- **Parâmetros novos**: `k2_fraction`, `k2_min`, `modulation`, `normalize_input`.

### Correções

- **Leitor de ground-truth corrigido** (C++ `read_bigann_groundtruth` já estava
  correto; os scripts de benchmark Python agora pulam os `dists` float32
  intercalados no formato BIGANN `[ids][dists]` por query).
- **Poda com bound garantido**: a poda sempre usa o bound mais justo disponível
  (B2), nunca o score modulado — garantindo 0 violações por construção.
- **Memória otimizada**: o caminho int8 aloca apenas projeções int8 (sem cópia
  float32), reduzindo o footprint em ~4× e tornando 100M viável em RAM restrita.

### 📊 Validação contra o GT oficial (BIGANN-100M)

- **Prova dos 9 no 100M**: R@10 = **0.7880**, NDCG = **0.8208**, varrendo o
  teto do scan exato, 0 violações.
- **Eficiência de 100%** vs o teto do scan exato em 10M (0.4300) e 100M
  (0.7880).
- **Cobertura do GT documentada**: 1M=1.2%, 10M=10.5%, 100M=100% — explicando
  por que o teto do subset 10M é 0.43.
- **Contraste Build vs Latency**: build 3.7s (10M) vs HNSW 1025s (~280×),
  posicionando o pacote para ingestão contínua / RAG dinâmico / batch.

### 📦 Testes

- 7 testes Python (incluindo novos: `test_cosine_cascade_self_is_top1`,
  `test_quant_none_matches_exact`, `test_self_is_top1_l2`).
- Testes C++ unitários (bound 0 violações, post-filter, métricas).

## [1.0.1] — 2026-08-04

### Adicionado

- README com badge de PyPI, downloads, CI e benchmark.
- Seção de limitações honestas (uint8 only, wheel cp312, bound-correctness).
- `docs/VERIFIED.md` com a verificação local.

## [1.0.0] — 2026-08-04

### Adicionado

- Motor C++20 com bound Cauchy-Schwarz e post-filter L2 exato.
- Módulo Python (pybind11) com `build_engine`, `search`, `search_exact`.
- Benchmark contra o GT oficial BIGANN-100M (subset 10M).
- CI (GitHub Actions) e publish via Trusted Publisher para o PyPI.## [1.6.0] — 2026-08-07

### 🧭 O(K) anchor navigation in speed mode (sublinear)

- PiPrime anchors (SVD power-iteration + Gram-Schmidt) partition the corpus
  into Voronoi cells
- Queries route to the nprobe most-similar anchor cells (O(K·d) navigation)
- QKᵀ runs only over the selected cells — sublinear, not brute force
- `build_engine(speed=True, speed_n_anchors=16, speed_nprobe=4)`
- recall 1.0 vs brute-force; 1.6× faster at 1M (CPU); scales with N



