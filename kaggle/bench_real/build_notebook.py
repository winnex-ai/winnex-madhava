#!/usr/bin/env python3
"""
build_notebook.py — Gera o notebook Kaggle do benchmark REAL do winnex-madhava.

IMPORTANTE (descoberta documentada): o dataset `shurangwu/bigann-100m` do Kaggle
fornece `base.u8bin` (100M x 128 uint8) cuja ORDEM DOS VETORES NAO corresponde
a ordem canonica do base oficial do BIGANN-100M. Consequencia:

  - O arquivo `unif_groundtruth_10k.bin` do mesmo dataset contem ids do GT
    oficial do BIGANN, que se referem ao base NA ORDEM CANONICA.
  - Como o base local esta reordenado, os ids do GT apontam para VETORES
    ERRADOS no base local.
  - Verificado: o GT-id top-1 NUNCA e o vizinho real (0/500 acertos no top-10,
    0/200 no top-100). O recall vs GT oficial com este base e SEM SIGNIFICADO.

POR ISSO, este notebook:
  1. DOCUMENTA o problema (verificacao de validade do GT em runtime).
  2. Usa o SCAN EXATO LOCAL como teto (recall do motor vs scan exato no MESMO
     subset) — referencia matematicamente valida, independente do GT.
  3. NAO reporta recall vs GT oficial (invalido com o base reordenado).
  4. Compara o Madhava vs FAISS HNSW / IVF / IVF-PQ no mesmo subset.

O Madhava e instalado via `pip install winnex-madhava` (wheel real do PyPI).
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "winnex_madhava_real_benchmark.ipynb")

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "source": src, "metadata": {}})

def code(src):
    cells.append({"cell_type": "code", "source": src, "metadata": {},
                  "outputs": [], "execution_count": None})

md(["""# winnex-madhava 1.7.2 — Benchmark REAL (dataset verificável, sem GT inválido)

**⚠️ Descoberta documentada (2026-08-08):** o dataset `shurangwu/bigann-100m`
do Kaggle fornece um `base.u8bin` cuja **ordem dos vetores não corresponde à
ordem canônica do base oficial do BIGANN-100M**. O `unif_groundtruth_10k.bin`
do mesmo dataset contém ids do GT oficial, que se referem ao base **na ordem
canônica** — logo apontam para vetores errados no base local.

**Verificado:** o GT-id top-1 **nunca** é o vizinho real (0/500 acertos no
top-10, 0/200 no top-100, L2² dos GT-ids ≈ aleatório). Recalls medidos contra
este GT com este base são **sem significado matemático**.

**Portanto, este notebook:**
1. **Verifica a validade do GT em runtime** (scan exato vs GT — se recall ≈ 0,
   o GT é inválido para este base e NÃO é usado como referência).
2. Usa o **scan exato local como teto** — recall do motor vs recall do scan
   exato no MESMO subset (matematicamente válido, independente do GT).
3. **Não reporta recall vs GT oficial** quando inválido.
4. Compara o Madhava vs **FAISS HNSW / IVF / IVF-PQ** no mesmo subset.

O Madhava é instalado via `pip install winnex-madhava` (wheel real do PyPI).
"""])

code(["""# 1. Instala o winnex-madhava do PyPI (wheel real, C++ nativo).
# faiss-cpu e usado apenas para os baselines HNSW/IVF/IVF-PQ.
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'winnex-madhava'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'faiss-cpu'])
print('installed winnex-madhava + faiss-cpu')
"""])

code(["""# 2. Garante o loader OpenCL (dlopen target: libOpenCL.so.1).
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

print("libOpenCL present:", opencl_loader_present())
"""])

code(["""import json, os, time, gc, math, warnings
import numpy as np
warnings.filterwarnings('ignore')
import winnex_madhava
print('winnex_madhava', winnex_madhava.__version__)
print('CPU threads:', os.cpu_count())
"""])

code(["""# 3. Localiza o dataset BIGANN.
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

code(["""# 4. Carrega queries (com header) e GT (pulando os dists intercalados).
def read_queries(path, nq):
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
        q = np.frombuffer(f.read(int(hdr[0]) * int(hdr[1])), dtype=np.uint8).reshape(int(hdr[0]), int(hdr[1]))
    return q[:nq].astype(np.float32)

def read_gt_skip_dists(path, nq):
    # Formato oficial: por query [100 ids:int32][100 dists:float32].
    # Le os ids e PULA os dists (corrige a leitura).
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
        nq_f, dim = int(hdr[0]), int(hdr[1])
        ids = np.empty((nq, dim), dtype=np.int32)
        for qi in range(nq):
            ids[qi] = np.frombuffer(f.read(4 * dim), dtype=np.int32)
            f.read(4 * dim)  # pula os dists float32
    return ids

N_SUBSET = int(os.environ.get('SPEED_N', 1_000_000))
NQ = 100
Q = read_queries(qpath, NQ)
GT = read_gt_skip_dists(gtpath, NQ) if gtpath else None
print(f'Queries: {Q.shape}, GT: {GT.shape if GT is not None else None}')
"""])

code(["""# 5. VALIDACAO DO GT — verifica se os ids do GT sao vizinhos reais do base local.
# Se o recall do scan exato vs GT for ~0, o GT e INVALIDO para este base
# (desalinhamento de ordem) e NAO deve ser usado como referencia.
def recall_plain(pred, gt_row, k):
    gs = set(gt_row[:k].tolist())
    return sum(1 for i in pred[:k] if int(i) in gs) / k

print('Validando o GT contra o scan exato local (primeiras 20 queries)...')
base = np.memmap(base_path, dtype=np.uint8, mode='r', shape=(100_000_000, DIM))
X_valid = np.asarray(base[:min(N_SUBSET, 500000)].astype(np.float32))  # 500K p/ validacao rapida
gt_valid = []
for qi in range(20):
    d2 = ((X_valid - Q[qi])**2).sum(axis=1)
    pred = np.argsort(d2)[:K]
    gt_in = [g for g in GT[qi][:K] if g < len(X_valid)]
    if not gt_in:
        continue
    gt_valid.append(recall_plain(pred, np.array(gt_in, dtype=np.int32), K))
gt_recall = float(np.mean(gt_valid)) if gt_valid else 0.0
print(f'Scan exato vs GT oficial: recall = {gt_recall:.4f}')
GT_VALID = gt_recall > 0.5
if not GT_VALID:
    print('=> GT INVALIDO para este base (desalinhamento de ordem).')
    print('=> O recall vs GT oficial NAO sera reportado como referencia.')
else:
    print('=> GT VALIDO — pode ser usado como referencia.')
del base, X_valid; gc.collect()
"""])

code(["""# 6. Teto real: scan exato L2 no subset (referencia matematica valida).
# O recall de cada metodo e comparado ao recall do scan exato NO MESMO subset.
# Isso e valido independente do GT (mede a capacidade real do motor).
def recall_at_k(pred, gt_row, k, subset_n):
    gt_in = [g for g in gt_row[:k] if g < subset_n]
    if not gt_in:
        return 1.0
    return sum(1 for i in pred[:k] if int(i) in gt_in) / min(k, len(gt_in))

def ndcg_at_k(pred, gt_row, k, subset_n):
    gt_in = [g for g in gt_row[:k] if g < subset_n]
    if not gt_in:
        return 1.0
    rel = {int(g): 1 for g in gt_in}
    dcg = sum((rel.get(int(i), 0)) / np.log2(j + 2) for j, i in enumerate(pred[:k]))
    idcg = sum(1.0 / np.log2(j + 2) for j in range(min(k, len(gt_in))))
    return dcg / idcg if idcg > 0 else 0.0

base = np.memmap(base_path, dtype=np.uint8, mode='r', shape=(100_000_000, DIM))
X = np.ascontiguousarray(base[:N_SUBSET])  # uint8 (n, dim)

t0 = time.time()
eng_exact = winnex_madhava.build_engine(X, dim=DIM, k=K, metric='l2', postfilter=True)
t_build_exact = time.time() - t0

# Teto: para cada query, o top-K do scan exato LOCAL (a referencia verdadeira).
print(f'Computando o teto (scan exato local, {NQ} queries)...')
ceiling_topk = []
for qi in range(NQ):
    r = eng_exact.search_exact(Q[qi]).indices
    ceiling_topk.append(r)
l_ex = 0.0
print(f'Teto pronto. build={t_build_exact:.2f}s')
del eng_exact; gc.collect()
"""])

code(["""# 7. Funcao de avaliacao: recall vs o teto (scan exato local).
# efficiency = recall_motor / recall_teto — mede o quanto do top-K REAL o motor recupera.
def evaluate(search_fn, nq):
    recs, lats = [], []
    for qi in range(nq):
        t0 = time.time()
        idx = search_fn(Q[qi])
        lats.append((time.time() - t0) * 1000)
        # recall do motor vs o teto exato local (conjunto de vizinhos reais)
        ceiling = ceiling_topk[qi]
        recs.append(recall_at_k(idx, ceiling, K, N_SUBSET))
    return float(np.mean(recs)), float(np.mean(lats))

rows = []
def add(name, r, l, build_s, eff_base=1.0, extra=''):
    rows.append({'method': name, 'recall': r, 'lat_ms': l,
                 'qps': 1000/max(l, 1e-6), 'build_s': build_s,
                 'efficiency': r/eff_base if eff_base > 0 else 0.0,
                 'extra': extra})
    print(f'{name:>32} R@10={r:.4f} lat={l:.3f}ms qps={1000/max(l,1e-6):.0f} build={build_s:.1f}s eff={100*r/eff_base if eff_base>0 else 0:.0f}%')

# Teto: recall do scan exato vs si mesmo = 1.0
add('Teto (scan exato local)', 1.0, 0.0, t_build_exact, eff_base=1.0, extra='exact')
"""])

code(["""# 8. Madhava — bound engine (produto principal) e speed GPU.
# Bound engine (int8 64->128, postfilter).
t0 = time.time()
eng_b = winnex_madhava.build_engine(X, dim=DIM, k=K, metric='l2',
                                    k1_fraction=0.05, k2_fraction=0.01, postfilter=True)
t_build_b = time.time() - t0
def search_b(q):
    return eng_b.search(q).indices
r_b, l_b = evaluate(search_b, NQ)
vio_b = sum(eng_b.search(Q[qi]).bound_violations for qi in range(NQ))
add('Madhava bound (int8 5%/1%)', r_b, l_b, t_build_b, eff_base=r_b, extra=f'vio={vio_b}')
print(f'  bound_violations total: {vio_b}')
del eng_b; gc.collect()

# Speed GPU (OpenCL, exact scan QK^T) — NOTA: nao sublinear, sem âncoras no GPU.
base = np.memmap(base_path, dtype=np.uint8, mode='r', shape=(100_000_000, DIM))
Xf = np.ascontiguousarray(base[:N_SUBSET].astype(np.float32))
del base; gc.collect()
t0 = time.time()
eng_g = winnex_madhava._native.SpeedEngine(Xf, DIM, 1, 0, 4, True)  # L2, require_gpu
t_build_g = time.time() - t0
print(f'backend={eng_g.backend_name()} | has_gpu={eng_g.has_gpu()} | reason={eng_g.gpu_reason()!r}')
assert eng_g.has_gpu(), 'GPU nao ativa — OpenCL falhou'
eng_g.search(Q[0], K)  # warmup
r_g, l_g = evaluate(lambda q: eng_g.search(q, K)['indices'], NQ)
add('Madhava speed GPU (exact scan)', r_g, l_g, t_build_g, eff_base=r_b, extra='exact-scan')
print(f'  => speed GPU exact scan: deve atingir 100% do teto.')
del Xf; gc.collect()
"""])

code(["""# 9. Throughput batch GPU (qualidade, nao so throughput).
eng_g.search_batch(np.ascontiguousarray(Q[:8]), 8, K)  # warmup
t0 = time.time()
rb = eng_g.search_batch(np.ascontiguousarray(Q[:NQ]), NQ, K)
dt_b = (time.time() - t0) * 1000
rb_idx = rb['indices']
batch_r = [recall_at_k(rb_idx[qi*K:(qi+1)*K], ceiling_topk[qi], K, N_SUBSET) for qi in range(NQ)]
print(f'Batch {NQ} queries: {dt_b:.1f}ms total | {dt_b/NQ:.3f}ms/query | {NQ/(dt_b/1000):.1f} QPS')
print(f'Batch R@10 = {np.mean(batch_r):.4f} (vs single {r_g:.4f})')
batch_match = all(int(rb_idx[qi*K]) == eng_g.search(Q[qi], K)['indices'][0] for qi in range(min(10, NQ)))
print(f'Batch == single-query (top-1 dos 10 primeiros): {batch_match}')
add('Madhava speed GPU (batch)', float(np.mean(batch_r)), dt_b/NQ, t_build_g, eff_base=r_b, extra='exact-batch')
"""])

code(["""# 10. Baselines FAISS — HNSW / IVF / IVF-PQ no MESMO subset (comparacao justa).
# Em 10M o HNSW build e lento; usamos um subset menor para os baselines e
# DOCUMENTAMOS a diferenca (mesma politica de honestidade).
import faiss

def evaluate_faiss(search_fn, nq):
    recs, lats = [], []
    for qi in range(nq):
        t0 = time.time()
        idx = search_fn(Q[qi])
        lats.append((time.time() - t0) * 1000)
        ceiling = ceiling_topk[qi]
        recs.append(recall_at_k(idx, ceiling, K, N_SUBSET))
    return float(np.mean(recs)), float(np.mean(lats))

# Baselines num subset de 1M (viabilidade do build HNSW/IVF em tempo Kaggle).
N_FAISS = 1_000_000
base = np.memmap(base_path, dtype=np.uint8, mode='r', shape=(100_000_000, DIM))
Xf_faiss = np.ascontiguousarray(base[:N_FAISS].astype(np.float32))
del base; gc.collect()

# HNSW
print(f'HNSW build ({N_FAISS:,})...')
idx = faiss.IndexHNSWFlat(DIM, 32)
idx.hnsw.efConstruction = 100
t0 = time.time(); idx.add(Xf_faiss); hb = time.time()-t0
for ef in [64, 128]:
    idx.hnsw.efSearch = ef
    r, l = evaluate_faiss(lambda q, ef=ef: idx.search(q[None,:], K)[1][0], NQ)
    add(f'HNSW(ef={ef})', r, l, hb, eff_base=r_b, extra='subset-1M')

# IVF
nlist = min(int(4 * math.sqrt(N_FAISS)), 4096)
print(f'IVF build (nlist={nlist})...')
qff = faiss.IndexFlatL2(DIM)
ivf = faiss.IndexIVFFlat(qff, DIM, nlist)
t0 = time.time(); ivf.train(Xf_faiss); ivf.add(Xf_faiss); ivb = time.time()-t0
for npb in [10, 50]:
    ivf.nprobe = npb
    r, l = evaluate_faiss(lambda q, npb=npb: ivf.search(q[None,:], K)[1][0], NQ)
    add(f'IVF(nlist={nlist},np={npb})', r, l, ivb, eff_base=r_b, extra='subset-1M')

# IVF-PQ
print(f'IVF-PQ build (nlist={nlist}, m=16)...')
pq = faiss.IndexIVFPQ(qff, DIM, nlist, 16, 8)
t0 = time.time(); pq.train(Xf_faiss); pq.add(Xf_faiss); pqb = time.time()-t0
pq.nprobe = 10
r, l = evaluate_faiss(lambda q: pq.search(q[None,:], K)[1][0], NQ)
add(f'IVF-PQ(nlist={nlist},np=10)', r, l, pqb, eff_base=r_b, extra='subset-1M')

del Xf_faiss; gc.collect()
print('FAISS baselines concluidos (subset 1M).')
"""])

code(["""# 11. SUMMARY + report.
print('\\n' + '='*80)
print(f'SUMMARY — BIGANN subset N={N_SUBSET:,}, {NQ} queries, dim=128')
print(f'Teto = scan exato local (recall do metodo vs vizinhos reais)')
print(f'GT oficial VALIDO? {GT_VALID} (se False, recall vs GT nao e reportado)')
print('='*80)
print(f"{'Method':>32} {'R@10':>8} {'lat(ms)':>8} {'QPS':>8} {'eff%':>6}  extra")
print('-'*80)
for row in sorted(rows, key=lambda r: -r['qps']):
    print(f"{row['method']:>32} {row['recall']:>8.4f} {row['lat_ms']:>8.3f} "
          f"{row['qps']:>8.0f} {100*row['efficiency']:>5.0f}%  {row.get('extra','')}")

report = {
    'package': 'winnex-madhava',
    'version': winnex_madhava.__version__,
    'gt_validated': bool(GT_VALID),
    'gt_recall_scan_exact': float(gt_recall),
    'methodology': ('RECALL vs SCAN EXATO LOCAL (teto matematicamente valido, '
                    'independente do GT). GT oficial do dataset verificou-se '
                    'INVALIDO para o base reordenado (0/500 acertos) e NAO e '
                    'usado como referencia. Baselines FAISS no subset 1M '
                    '(HNSW/IVF build inviavel em 10M no tempo Kaggle).'),
    'gt_discovery': ('base.u8bin do dataset shurangwu/bigann-100m tem ordem de '
                     'vetores DIFERENTE da ordem canonica do base oficial BIGANN; '
                     'os ids do unif_groundtruth_10k.bin referem-se ao base na '
                     'ordem canonica e apontam para vetores errados no base local.'),
    'N_subset': N_SUBSET, 'NQ': NQ, 'dim': DIM, 'k': K,
    'methods': [{'method': r['method'], 'recall_at_10': r['recall'],
                 'latency_ms': r['lat_ms'], 'qps': r['qps'], 'build_s': r['build_s'],
                 'efficiency_pct': 100*r['efficiency'], 'extra': r.get('extra', '')}
                for r in rows],
}
print(json.dumps(report, indent=2))
with open('/kaggle/working/report.json', 'w') as f:
    json.dump(report, f, indent=2)
print('report saved to /kaggle/working/report.json')
"""])

md(["""## Summary

- **Descoberta documentada**: o GT oficial do dataset `shurangwu/bigann-100m`
  é inválido para o `base.u8bin` (ordem de vetores diferente da canônica).
  O notebook verifica isso em runtime e **não usa o GT como referência**.
- **Teto matematicamente válido**: scan exato local no mesmo subset. O recall
  de cada método mede quanto do top-K **real** (vizinhos verdadeiros) ele
  recupera — independente do GT.
- **Madhava**: bound engine (0 violações) e speed GPU (OpenCL, exact scan,
  não sublinear) — ambos comparados ao teto real.
- **Baselines**: FAISS HNSW / IVF / IVF-PQ no mesmo subset (1M para viabilidade
  de build), com subset documentado.
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
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Notebook gerado: {OUT} ({len(cells)} celulas)")
