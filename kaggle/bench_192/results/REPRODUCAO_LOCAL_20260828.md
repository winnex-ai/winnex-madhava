# Reprodução Local — winnex-madhava 1.9.2 AuditCommitment Benchmark
**Data:** 2026-08-28
**Objetivo:** gerar output auditável do benchmark `winnex-madhava-1-9-2-honest` fora do Kaggle, pois o output de execução na plataforma exige login. O código é idêntico ao do Kaggle (`bench_192/winnex-madhava-1-9-2-honest.py`), adaptado apenas para caminhos locais.

---

## 1. Protocolo (idêntico ao Kaggle)

- **Pacote:** `winnex-madhava==1.9.2` instalado do PyPI (venv dedicado, versão verificada).
- **Dados brutos** (sem pré-processamento numpy):
  - GloVe (`rtatman/glove`) d=100 — `glove.6B.100d.txt` (20.000 vetores)
  - BIGANN (`shurangwu/bigann-100m`) d=128 uint8 — `base.u8bin` (20.000 vetores)
  - arXiv (`tomtum/openai-arxiv-embeddings`) d=1536 — extraído de `ml-arxiv-embeddings.json` → `vectors.dat` raw float32 n×1536 (20.000 vetores, mesmos do dataset Kaggle)
- **N=19.900** (train), **NQ=100** (held-out), **K=10**, métrica cosine, `normalize_input=True`.
- **GT:** o próprio `search_exact` do motor na mesma query (ceiling válido) — não há reordenação nem GT externo.
- **Bases:** `random` e `pca_corpus`, `stage1_dim=min(192, dim)`, `stage2_dim=0`.

## 2. Resultados (métricas do motor)

### GloVe d=100 (N=19.900)

| Basis | recall@10 | bound_viol | pruned_bound | pruned_prefilter | survivor | lat(ms) | build(s) |
|---|---|---|---|---|---|---|---|
| random | **1.0000** | **0** | 99.9% | 0.0% | 5.0% | 25.7 | 0.0 |
| pca_corpus | **1.0000** | **0** | 72.4% | 22.6% | 5.0% | 21.2 | 0.2 |

### BIGANN d=128 (N=19.900)

| Basis | recall@10 | bound_viol | pruned_bound | pruned_prefilter | survivor | lat(ms) | build(s) |
|---|---|---|---|---|---|---|---|
| random | **1.0000** | **0** | 99.9% | 0.0% | 5.0% | 22.9 | 0.0 |
| pca_corpus | **1.0000** | **0** | 99.9% | 0.0% | 5.0% | 21.9 | 0.1 |

### arXiv d=1536 (N=19.900)

| Basis | recall@10 | bound_viol | pruned_bound | pruned_prefilter | survivor | lat(ms) | build(s) |
|---|---|---|---|---|---|---|---|
| random | **0.9940** | **0** | 0.0% | 95.0% | 5.0% | 26.5 | 0.6 |
| pca_corpus | **1.0000** | **0** | 87.1% | 8.5% | 5.0% | 23.0 | 23.0 |

> Nota: o 0.994 no arXiv-random (vs 0.996 no site, que reporta o v1.8.8) é esperado — os números variam ligeiramente entre versões (1.9.2 vs 1.8.8) e a extração local dos vetores é independente. O comportamento qualitativo é idêntico: em d=1536 o basis random mal faz bound (0.0% bound-pruned, 95% prefilter), e pca_corpus restaura o pruning (87.1%) com recall 1.0000 — exatamente o que o site documenta.

## 3. Validação do AuditCommitment (1.9.2) — 100/100 em todos os critérios

Para **cada** query (100 por dataset × 2 bases = 600 validações):

| Critério | GloVe | BIGANN | arXiv | Claim do site |
|---|---|---|---|---|
| `sample_bounded` (≤ max_sample=50) | 100/100 | 100/100 | 100/100 | ✅ "max_sample is HONORED" |
| `exact_count_match` (== search_audited) | 100/100 | 100/100 | 100/100 | ✅ "total_excluded == audit_excluded (exact)" |
| `deterministic` (mesma query → mesmo sample) | 100/100 | 100/100 | 100/100 | ✅ determinismo |
| `genuine_exclusions` (UB < threshold) | 100/100 | 100/100 | 100/100 | ✅ exclusão genuína |
| `bound_violations` | 0 | 0 | 0 | ✅ 0 violações |

**Payload do commitment** (o claim "WORM-friendly ~KB, not ~MB"):
- GloVe: máx 3.888 B (~3.9 KB) — com 1.988.995 excluded records
- BIGANN: máx 3.826 B (~3.8 KB) — com 1.988.997 excluded records
- arXiv (pca): máx 3.776 B — com 1.734.004 excluded records
- arXiv (random): 243 B (0 excluded — prefilter faz o trabalho)

**Confirma o CHANGELOG 1.9.2:** o commitment é um registro leve de ~KB (não os ~2 MB/query do `search_audited` em d=1536).

## 4. Conclusão

Os claims técnicos do site winnex.ai sobre a versão 1.9.x são **reproduzidos e auditáveis**:

1. **0 bound violations** em GloVe, BIGANN e arXiv — ✅ confirmado em 600 queries.
2. **AuditCommitment válido**: bounded, exato, determinístico e genuíno — ✅ 100/100 em todos os critérios, em todos os datasets.
3. **Payload ~KB** (não ~MB) — ✅ confirmado (~3.8 KB com quase 2M excluded).
4. **Comportamento em d=1536 documentado corretamente**: random basis não faz bound (0.0%), pca_corpus restaura (87.1%) — ✅ exatamente o que o site descreve.

**Artefatos:** este diretório contém os JSONs de resultado por dataset, o `summary.json` consolidado e o log completo da execução.

## 5. Como reproduzir

```bash
python3 -m venv /tmp/wm192_venv
/tmp/wm192_venv/bin/pip install "winnex-madhava==1.9.2" numpy
/tmp/wm192_venv/bin/python run_192_local.py /tmp/results_192 \
  /tmp/glove_unz/glove.6B.100d.txt \
  /home/wnnx_user/zenodo/bigann_data/base.u8bin \
  /tmp/arxiv_vectors_20k.dat
```
