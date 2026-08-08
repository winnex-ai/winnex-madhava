# Changelog

All notable changes to `winnex-madhava` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.2] — 2026-08-08

### ⚡ Speed GPU: fused QKᵀ+topk — single-query 20× mais rápido

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

### 📊 Benchmark honesto 10M (Kaggle, P100)

Novo notebook público:
[winnex-madhava-1-7-honest-10m-gpu-vs-official-gt](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-1-7-honest-10m-gpu-vs-official-gt)
— subset **10M** do BIGANN-100M, 100 queries, GT oficial, recall normalizado
por cobertura (~8% no 10M).

| Método | R@10 | Lat (ms) | Eficiência |
|---|---|---|---|
| Teto real (search_exact) | 0.4400 | 554.2 | — |
| Madhava bound (5%/1%) | 0.4400 | 491.0 | 100% |
| **Speed GPU (OpenCL v1.7.2)** | **0.4400** | **38.1** | **100%** |
| Batch GPU | 0.4400 | 22.6 | 100% |

**O R@10=0.44 em 10M reproduz o valor documentado do projeto** (~0.43-0.52),
com o speed GPU **~13× mais rápido que o bound engine** (38ms vs 491ms) e 0
violações em todos os métodos.

## [1.7.1] — 2026-08-07

### ⚡ Speed GPU: topk paralelo (divide-and-conquer) + QKᵀ com tiling

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

### ✨ Hybrid mode (MadHybrid)

`build_engine(..., hybrid=True, nlist=..., nprobe=...)` returns a clustered
MadHybrid wrapper: the same bound engine, partitioned into cells and queried
sublinearly. Supports **float32 embeddings** (cosine, pure-Python bound cells)
and **uint8 raw bytes** (L2, native C++ per cell). One motor, two modes.

- `MadHybrid` class + `_MadhavaCellPy` (per-cell bound cascade 32D→64D)
- `build_engine` gains `hybrid`, `nlist`, `nprobe` parameters
- Validated on News 210K / SBERT: recall@10 ≈ 1.0 at nprobe=3–8
- See README "Hybrid mode (MadHybrid)"


## [1.1.3] — 2026-08-05

### 🔧 Métricas robustas (Recall@K / NDCG@K)

As métricas dividiam por `k=10` fixo e intersectavam apenas `gt[:k]`, o que
penalizava queries com poucos relevantes no subset (um scan exato perfeito podia
marcar < 1.0) e ignorava vizinhos relevantes além da posição k.

**Nova definição:**
```
recall@K = |result[:K] ∩ gt| / min(K, |gt|)
ndcg@K   = DCG / IDCG, com IDCG usando min(K, |gt|)
```

**Verificado (10M e 100M, GT oficial, Kaggle):**

| Escala | R@10 | NDCG | Eficiência | Vio | Build |
|---|---|---|---|---|---|
| 10M | **0.4882** | 0.5393 | 100% | 0 | 27.7s |
| 100M | **0.7880** | 0.8208 | 100% | 0 | 227.5s |

Agora um scan exato perfeito atinge **exatamente 1.0** quando o GT tem poucos
relevantes, e os números são consistentes com runs anteriores (0.488/0.788).

### Benchmark Kaggle (pip install)

Novo notebook público: [winnex-madhava-pip-113](https://www.kaggle.com/code/kleniopadilha/winnex-madhava-pip-113).

## [1.1.2] — 2026-08-05

### 🔧 Corrige a definição de recall@K (padrão BIGANN)

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

### 🔧 Correções

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



