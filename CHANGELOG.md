# Changelog

All notable changes to `winnex-madhava` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
