#!/usr/bin/env python3
"""
build_notebook.py — Gera o notebook Kaggle do benchmark HONESTO 10M do winnex-madhava.

Baseado no benchmark honesto 1M (kaggle/bench_honesto), mas com subset MAIOR:
N_SUBSET=10_000_000 — onde a cobertura do GT oficial é ~8-10% (vs ~1% no 1M),
dando um recall bruto mais alto e uma validação mais significativa contra o GT.

O Madhava e instalado via `pip install winnex-madhava` (nao por codigo-fonte).
Metodologia (criterio tecnico rigoroso, validada localmente na RTX 5060 Ti):

  1. METRICA PRIMARIA: Recall@10 vs GT OFICIAL BIGANN-100M (alinhamento direto
     GT[i]<->Q[i], nao um ceiling artificial).
  2. Recall normalizado por cobertura do GT no subset: um scan exato perfeito
     atinge 1.0 mesmo com cobertura parcial.
  3. Teto real = search_exact L2 no subset (limite fisico do dataset).
  4. Eficiencia = recall_motor / recall_teto, sempre no MESMO subset.
  5. Speed GPU = exact scan QK^T (OpenCL). NAO e sublinear — sem âncoras ativas
     no GPU. Nota de honestidade: reportamos apenas o exact scan real.
  6. Cobertura GT reportada explicitamente.

Gerenciamento de memoria (10M): o corpus float32 pesa ~5.1GB. As fases liberam
X (uint8) antes de criar Xf (float32) para nao estourar a RAM do Kaggle.

Hardware Kaggle: GPU (P100 16GB ou similar), dataset shurangwu/bigann-100m.
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "winnex_madhava_honest_10m_benchmark.ipynb")

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "source": src, "metadata": {}})

def code(src):
    cells.append({"cell_type": "code", "source": src, "metadata": {},
                  "outputs": [], "execution_count": None})

md(["""# winnex-madhava 1.7.2 — Benchmark HONESTO 10M vs GT oficial BIGANN-100M

**O Madhava é instalado via `pip install winnex-madhava`** — o wheel real do PyPI,
com o núcleo C++20 nativo (bound Cauchy-Schwarz + speed mode OpenCL GPU).

**Este notebook usa um subset MAIOR: N=10M** (vs 1M no benchmark anterior), onde
a cobertura do GT oficial é ~8-10% — um recall bruto mais alto e uma validação
mais significativa contra o ground truth.

**Metodologia (critério técnico rigoroso):**
- **Recall@10 vs GT OFICIAL BIGANN-100M L2** (alinhamento `GT[i]↔Q[i]`, não um
  ceiling artificial recalculado).
- **Recall normalizado por cobertura do GT no subset**: `|result[:K] ∩ (GT[:K]∩subset)| / min(K, |GT[:K]∩subset|)`.
  Um scan exato perfeito atinge 1.0 mesmo quando parte dos vizinhos verdadeiros
  está fora do subset.
- **Teto real** = `search_exact` L2 no subset (limite físico — verificado no
  código C++: scan linear puro).
- **Eficiência** = recall_motor / recall_teto, sempre no **mesmo subset**.
- **Speed GPU = exact scan QKᵀ** (OpenCL). **NÃO é sublinear** — sem âncoras
  ativas no GPU. Reportamos apenas o exact scan real.
- **Cobertura GT** reportada explicitamente: em subset <100M o recall bruto é
  limitado pela fração de vizinhos verdadeiros presentes no subset.
"""])

code(["""# 1. Instala o winnex-madhava do PyPI (o wheel real, com C++ nativo).
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'winnex-madhava'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'numpy'])
print('installed winnex-madhava')
"""])

code(["""# 2. Garante o loader OpenCL (dlopen target: libOpenCL.so.1).
# O wheel dlopens libOpenCL em runtime; se faltar, instala via apt.
import os, subprocess, sys, ctypes

def opencl_loader_present():
    try:
        ctypes.cdll.LoadLibrary("libOpenCL.so.1")
        return True
    except OSError:
        return False

if not opencl_loader_present():
    print("libOpenCL.so.1 missing — installing ocl-icd-libopencl1 ...")
    r = subprocess.run(
        ["apt-get", "update", "-qq"] +
        ["apt-get", "install", "-y", "-qq", "ocl-icd-libopencl1"],
        capture_output=True, text=True)
    print("apt install exit:", r.returncode)
    if not opencl_loader_present():
        print("WARNING: OpenCL loader still missing — will fall back to CPU.")

print("libOpenCL present:", opencl_loader_present())
"""])

code(["""import json, os, time, gc, warnings
import numpy as np
warnings.filterwarnings('ignore')
import winnex_madhava
print('winnex_madhava', winnex_madhava.__version__)
print('CPU threads:', os.cpu_count())
"""])

code(["""# 3. Localiza o dataset BIGANN (recursivo, lida com mount aninhado).
def find_file(name):
    for r, d, files in os.walk('/kaggle/input/'):
        if name in files:
            return os.path.join(r, name)
    return None

base_path = find_file('base.u8bin')
qpath = find_file('unif_query_10k.u8bin')
gtpath = find_file('unif_groundtruth_10k.bin')
print('base :', base_path)
print('queries:', qpath)
print('gt   :', gtpath)

DIM, K = 128, 10
HAS_BIGANN = all([base_path, qpath, gtpath])
print('BIGANN dataset found:', HAS_BIGANN)
"""])

code(["""# 4. Carrega queries + GT OFICIAL (alinhamento direto GT[i] <-> Q[i]).
def read_queries(path, nq):
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
        q = np.frombuffer(f.read(int(hdr[0]) * int(hdr[1])), dtype=np.uint8).reshape(int(hdr[0]), int(hdr[1]))
    return q[:nq].astype(np.float32)

def read_gt(path, nq):
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
        gt = np.frombuffer(f.read(4 * int(hdr[1]) * int(hdr[0])), dtype=np.int32).reshape(int(hdr[0]), int(hdr[1]))
    return gt[:nq]

# Subset MAIOR: 10M (cobertura GT ~8-10%). Usa 100 queries para o custo de
# busca exata ser viavel no Kaggle (10M x 100 = 1.28G de distancia em float32).
N_SUBSET = int(os.environ.get('SPEED_N', 10_000_000))
NQ = 100
Q = read_queries(qpath, NQ)
GT = read_gt(gtpath, NQ)
print(f'Queries: {Q.shape}, GT: {GT.shape}')
"""])

code(["""# 5. Metricas HONESTAS.
# recall normalizado por cobertura: scan exato perfeito = 1.0 mesmo com
# cobertura parcial do GT no subset.
def recall_at_k(pred, gt_row, k, subset_n):
    gt_in = [g for g in gt_row[:k] if g < subset_n]
    if not gt_in:
        return 1.0  # sem vizinhos alcancaveis — trata como perfeito
    return sum(1 for i in pred[:k] if int(i) in gt_in) / min(k, len(gt_in))

def ndcg_at_k(pred, gt_row, k, subset_n):
    gt_in = [g for g in gt_row[:k] if g < subset_n]
    if not gt_in:
        return 1.0
    rel = {int(g): 1 for g in gt_in}
    dcg = sum((rel.get(int(i), 0)) / np.log2(j + 2) for j, i in enumerate(pred[:k]))
    idcg = sum(1.0 / np.log2(j + 2) for j in range(min(k, len(gt_in))))
    return dcg / idcg if idcg > 0 else 0.0

def gt_coverage(gt_row, subset_n, k):
    return np.mean([g < subset_n for g in gt_row[:k]])

# Cobertura do GT no subset — O RECALL BRUTO ESTA LIMITADO A ISSO.
cov = float(np.mean([gt_coverage(GT[i], N_SUBSET, K) for i in range(NQ)]))
print(f'Cobertura GT top-{K} no subset {N_SUBSET:,}: {cov:.3f}')
print(f'=> recall bruto vs GT oficial limitado a ~{cov:.0%} mesmo para scan exato.')
"""])

code(["""# 6. Teto real: search_exact L2 no subset (limite fisico).
base = np.memmap(base_path, dtype=np.uint8, mode='r', shape=(100_000_000, DIM))
X = np.ascontiguousarray(base[:N_SUBSET])  # uint8 (10M x 128) = 1.28GB
del base; gc.collect()

t0 = time.time()
eng_exact = winnex_madhava.build_engine(X, dim=DIM, k=K, metric='l2', postfilter=True)
t_build_exact = time.time() - t0

recs_ex, lats_ex = [], []
for qi in range(NQ):
    tq = time.time()
    idx = eng_exact.search_exact(Q[qi]).indices
    lats_ex.append((time.time() - tq) * 1000)
    recs_ex.append(recall_at_k(idx, GT[qi], K, N_SUBSET))
r_ex = float(np.mean(recs_ex)); l_ex = float(np.mean(lats_ex))
print(f'Teto real (search_exact): R@10={r_ex:.4f} | lat={l_ex:.2f}ms | build={t_build_exact:.2f}s')
print(f'=> este é o limite físico: nenhum método pode superá-lo no subset.')
del eng_exact; gc.collect()
"""])

code(["""# 7. Bound engine (MadhavaL2) — o produto principal.
# Config agressiva (5%/1%) E generosa (100%) para medir o custo real da poda.
# Em 10M, k1_fraction=0.05 -> 500K sobreviventes, k2_fraction=0.01 -> 100K.
t0 = time.time()
eng_b = winnex_madhava.build_engine(X, dim=DIM, k=K, metric='l2',
                                    k1_fraction=0.05, k2_fraction=0.01, postfilter=True)
t_build_b = time.time() - t0
recs_b, lats_b, vio_b = [], [], 0
for qi in range(NQ):
    tq = time.time()
    r = eng_b.search(Q[qi])
    lats_b.append((time.time() - tq) * 1000)
    recs_b.append(recall_at_k(r.indices, GT[qi], K, N_SUBSET))
    vio_b += r.bound_violations
r_b = float(np.mean(recs_b)); l_b = float(np.mean(lats_b))
print(f'Bound 5%/1%:  R@10={r_b:.4f} | lat={l_b:.2f}ms | build={t_build_b:.2f}s | vio={vio_b} | eff={100*r_b/r_ex:.1f}%')

# Variante generosa: k1_fraction=1.0 (poda minima) — mede o custo da poda.
t0 = time.time()
eng_b2 = winnex_madhava.build_engine(X, dim=DIM, k=K, metric='l2',
                                     k1_fraction=1.0, k2_fraction=1.0, postfilter=True)
t_build_b2 = time.time() - t0
recs_b2, lats_b2 = [], []
for qi in range(NQ):
    tq = time.time()
    r = eng_b2.search(Q[qi])
    lats_b2.append((time.time() - tq) * 1000)
    recs_b2.append(recall_at_k(r.indices, GT[qi], K, N_SUBSET))
r_b2 = float(np.mean(recs_b2)); l_b2 = float(np.mean(lats_b2))
print(f'Bound 100%:  R@10={r_b2:.4f} | lat={l_b2:.2f}ms | build={t_build_b2:.2f}s | eff={100*r_b2/r_ex:.1f}%')
print(f'=> custo de recall da poda 5%/1% vs 100%: {r_b - r_b2:+.4f}')
del eng_b, eng_b2, X; gc.collect()
"""])

code(["""# 8. Speed engine GPU (OpenCL) — exact scan QK^T.
# NOTA DE HONESTIDADE: o GPU executa o QK^T completo (exact scan). NAO e
# sublinear — sem âncoras ativas no GPU. Reportamos apenas o exact scan real.
# Em 10M, o corpus float32 pesa ~5.1GB — cabe na VRAM do P100 (16GB).
base = np.memmap(base_path, dtype=np.uint8, mode='r', shape=(100_000_000, DIM))  # re-abre
Xf = np.ascontiguousarray(base[:N_SUBSET].astype(np.float32))  # 10M x 128 f32 = 5.1GB
del base; gc.collect()
t0 = time.time()
eng_g = winnex_madhava._native.SpeedEngine(Xf, DIM, 1, 0, 4, True)  # L2, require_gpu
t_build_g = time.time() - t0
print(f'backend={eng_g.backend_name()} | has_gpu={eng_g.has_gpu()} | reason={eng_g.gpu_reason()!r}')
assert eng_g.has_gpu(), 'GPU nao ativa — OpenCL falhou'
eng_g.search(Q[0], K)  # warmup
recs_g, lats_g = [], []
for qi in range(NQ):
    tq = time.time()
    r = eng_g.search(Q[qi], K)
    lats_g.append((time.time() - tq) * 1000)
    recs_g.append(recall_at_k(r['indices'], GT[qi], K, N_SUBSET))
r_g = float(np.mean(recs_g)); l_g = float(np.mean(lats_g))
print(f'Speed GPU:  R@10={r_g:.4f} | lat={l_g:.2f}ms | build={t_build_g:.2f}s | eff={100*r_g/r_ex:.1f}%')
print(f'=> exact scan: deve atingir 100% do teto (todos os vizinhos).')
"""])

code(["""# 9. Throughput batch GPU — com avaliacao de QUALIDADE (nao so throughput).
eng_g.search_batch(np.ascontiguousarray(Q[:8]), 8, K)  # warmup
t0 = time.time()
rb = eng_g.search_batch(np.ascontiguousarray(Q[:NQ]), NQ, K)
dt_b = (time.time() - t0) * 1000
rb_idx = rb['indices']
batch_r = [recall_at_k(rb_idx[qi*K:(qi+1)*K], GT[qi], K, N_SUBSET) for qi in range(NQ)]
print(f'Batch {NQ} queries: {dt_b:.1f}ms total | {dt_b/NQ:.3f}ms/query | {NQ/(dt_b/1000):.1f} QPS')
print(f'Batch R@10 = {np.mean(batch_r):.4f} (vs single {r_g:.4f})')
# batch == single-query?
match = all(int(rb_idx[qi*K]) == eng_g.search(Q[qi], K)['indices'][0] for qi in range(min(10, NQ)))
print(f'Batch == single-query (top-1 dos 10 primeiros): {match}')
"""])

code(["""# 10. Salva o relatorio.
report = {
    'package': 'winnex-madhava',
    'version': winnex_madhava.__version__,
    'methodology': ('RECALL vs GT OFICIAL BIGANN-100M L2 (alinhamento direto), '
                    'recall normalizado por cobertura do subset; '
                    'teto = search_exact L2 no subset (limite fisico); '
                    'speed GPU = exact scan QK^T, nao sublinear (honestidade).'),
    'reference': 'GT OFICIAL BIGANN-100M (nao ceiling artificial)',
    'N_subset': N_SUBSET, 'NQ': NQ, 'dim': DIM, 'k': K,
    'gt_coverage_topk': cov,
    'ceiling_search_exact': {'recall_at_10': r_ex, 'latency_ms': l_ex, 'build_s': t_build_exact},
    'methods': [
        {'method': 'MadhavaL2 bound (int8 5%/1%)', 'recall_at_10': r_b, 'latency_ms': l_b,
         'build_s': t_build_b, 'efficiency_pct': 100*r_b/r_ex, 'bound_violations': int(vio_b)},
        {'method': 'MadhavaL2 bound (int8 100%/100%)', 'recall_at_10': r_b2, 'latency_ms': l_b2,
         'build_s': t_build_b2, 'efficiency_pct': 100*r_b2/r_ex},
        {'method': 'Madhava speed GPU (OpenCL exact scan)', 'recall_at_10': r_g, 'latency_ms': l_g,
         'build_s': t_build_g, 'efficiency_pct': 100*r_g/r_ex,
         'backend': eng_g.backend_name(), 'has_gpu': bool(eng_g.has_gpu()),
         'batch_ms_per_query': float(dt_b/NQ), 'batch_recall_at_10': float(np.mean(batch_r)),
         'honesty_note': 'exact scan QK^T full corpus; NOT sublinear'},
    ],
}
print(json.dumps(report, indent=2))
with open('/kaggle/working/report.json', 'w') as f:
    json.dump(report, f, indent=2)
print('report saved to /kaggle/working/report.json')
"""])

md(["""## Summary

- **O Madhava foi instalado via `pip install winnex-madhava`** — o wheel real do PyPI.
- **Subset MAIOR: 10M** — cobertura GT ~8-10% (vs ~1% no 1M), recall bruto mais alto.
- **Métrica primária**: Recall@10 vs **GT oficial BIGANN-100M** (não ceiling artificial).
- **Recall normalizado por cobertura**: um scan exato atinge 1.0 mesmo quando parte
  dos vizinhos verdadeiros está fora do subset.
- **Teto real** = `search_exact` (scan linear puro, verificado no código C++).
- **Eficiência** = recall_motor / recall_teto, no mesmo subset.
- **Speed GPU** = exact scan QKᵀ (OpenCL). **Não é sublinear** — nota de honestidade.
- `bound_violations == 0` é a garantia Cauchy-Schwarz por documento.
"""])

nb = {
    "cells": cells,
    "metadata": {
        "kaggle": {
            "accelerator": "GPU",
            "language": "python",
            "kernelType": "notebook",
            "isPrivate": True,
            "datasetSources": ["shurangwu/bigann-100m"],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Notebook gerado: {OUT} ({len(cells)} celulas)")
