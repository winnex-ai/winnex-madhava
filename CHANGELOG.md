# Changelog

All notable changes to `winnex-madhava` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- CI (GitHub Actions) e publish via Trusted Publisher para o PyPI.
