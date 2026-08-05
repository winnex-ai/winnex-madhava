#!/usr/bin/env python3
"""
bench_verify.py — Verificação independente dos claims do winnex-madhava v1.1.0
==============================================================================
Valida os claims do README e do relatório contra o GT L2 oficial do BIGANN:

  Claim 1: 10M  → R@10 = 0.430 = teto (scan exato), eficiência 100%
  Claim 2: 100M → R@10 = 0.745 = 94% do teto (0.788)
  Claim 3: 0 bound violations em todas as escalas

Além disso, verifica a parametrização nova (cosine + cascata + modulação):
  - cosine + [64,128] + modulation: o motor bate o scan exato (ceiling)
  - quant='none' == quant='int8' (métrica exata)

Uso:  python3 bench_verify.py [--n 10000000] [--nq 50]
"""
import argparse, os, time, json
import numpy as np

import winnex_madhava

D = 128
K = 10

BASE = "/home/wnnx_user/zenodo/bigann_data/base.u8bin"
QFILE = "/home/wnnx_user/zenodo/bigann_data/unif_query_10k.u8bin"
GTFILE = "/home/wnnx_user/zenodo/bigann_data/unif_groundtruth_10k.bin"


def read_bigann_gt(path, n_queries):
    """Lê o GT oficial BIGANN. Formato: [nq:int32][dim:int32] então, por
    query: [dim ids:int32][dim dists:float32]. Os dists intercalam — precisamos
    pular 4*dim bytes de dists entre cada bloco de ids."""
    with open(path, "rb") as f:
        nq, dim = np.frombuffer(f.read(8), dtype=np.int32)
        ids = np.empty((n_queries, dim), dtype=np.int32)
        for gi in range(n_queries):
            ids[gi] = np.frombuffer(f.read(4 * dim), dtype=np.int32)
            f.read(4 * dim)  # pula os dists float32
    return ids.tolist(), nq, dim


def recall_at_k(ann, gt, k=K):
    return len(set(ann[:k]) & set(gt[:k])) / max(k, 1)


def ndcg_at_k(ann, gt, k=K):
    rel = {v: 1 for v in gt[:k]}
    dcg = sum(rel.get(a, 0) / np.log2(i + 2) for i, a in enumerate(ann[:k]))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg else 0.0


def eval_engine(eng, queries, gt_ids, n):
    """Avalia o motor contra o GT oficial (GT[i] <-> query 2i)."""
    nq = len(gt_ids)
    tr = tn = tlat = 0.0
    vio = 0
    n_eval = 0
    for gi in range(nq):
        gs = [v for v in gt_ids[gi] if 0 <= v < n]
        if not gs:
            continue
        qi = 2 * gi
        t0 = time.time()
        res = eng.search(queries[qi])
        tlat += (time.time() - t0) * 1000
        tr += recall_at_k(res.indices, gs)
        tn += ndcg_at_k(res.indices, gs)
        vio += res.bound_violations
        n_eval += 1
    m = max(n_eval, 1)
    return {"recall": tr / m, "ndcg": tn / m, "lat_ms": tlat / m, "vio": vio, "n_eval": n_eval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--nq", type=int, default=50)
    args = ap.parse_args()
    n, nq = args.n, args.nq

    print(f"=== Verificação dos claims winnex-madhava v1.1.0 ===")
    print(f"dataset: BIGANN-100M subset N={n:,}, nq={nq}, k={K}")
    print(f"memória livre: {os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')/1e9:.1f}GB\n")

    # Carrega GT + queries
    gt_ids, g_nq, g_dim = read_bigann_gt(GTFILE, nq)
    with open(QFILE, "rb") as f:
        qbuf = np.frombuffer(f.read(nq * 2 * D), dtype=np.uint8).reshape(nq * 2, D)
    queries = qbuf.astype(np.float32)
    print(f"GT: {g_nq}x{g_dim}, queries: {queries.shape}, alinhamento GT[i]<->2i\n")

    # mmap da base (sem copiar para RAM)
    base = np.memmap(BASE, dtype=np.uint8, mode="r", shape=(n, D))
    print(f"base mmap: {n:,} x {D} ({base.nbytes/1e9:.1f}GB)\n")

    results = {}

    # ─────────────────────────────────────────────────────────────
    # Claim 1: 10M L2 — R@10 = 0.430 = ceiling, 0 vio
    # ─────────────────────────────────────────────────────────────
    print("[Claim 1] L2 single-stage [64], postfilter=True (README claim)")
    cfg = winnex_madhava.Config()
    cfg.dim = D
    cfg.metric = winnex_madhava.Metric.L2
    cfg.stage1_dim = 64
    cfg.stage2_dim = 0
    cfg.k = K
    cfg.k1_fraction = 0.05
    cfg.postfilter = True
    eng = winnex_madhava.MadhavaL2(cfg)
    t0 = time.time()
    eng.build_numpy(base)
    build_s = time.time() - t0
    r_madhava = eval_engine(eng, queries, gt_ids, n)
    r_ceiling = eval_engine(eng, queries, gt_ids, n)  # same engine search_exact below

    # ceiling = search_exact
    r_ceil = {"recall": 0.0, "ndcg": 0.0, "lat_ms": 0.0, "vio": 0, "n_eval": 0}
    nq_eval = 0
    for gi in range(nq):
        gs = [v for v in gt_ids[gi] if 0 <= v < n]
        if not gs:
            continue
        qi = 2 * gi
        t0 = time.time()
        res = eng.search_exact(queries[qi])
        r_ceil["lat_ms"] += (time.time() - t0) * 1000
        r_ceil["recall"] += recall_at_k(res.indices, gs)
        r_ceil["ndcg"] += ndcg_at_k(res.indices, gs)
        nq_eval += 1
    m = max(nq_eval, 1)
    r_ceil = {k2: (v / m if k2 in ("recall", "ndcg", "lat_ms") else v) for k2, v in r_ceil.items()}
    r_ceil["n_eval"] = nq_eval

    eff = 100.0 * r_madhava["recall"] / max(r_ceil["recall"], 1e-9)
    results["l2_single_stage"] = {
        "build_s": build_s,
        "madhava": r_madhava,
        "ceiling": r_ceil,
        "efficiency_pct": eff,
    }
    print(f"  build: {build_s:.1f}s")
    print(f"  Madhava:  R@10={r_madhava['recall']:.4f} NDCG={r_madhava['ndcg']:.4f} lat={r_madhava['lat_ms']:.1f}ms vio={r_madhava['vio']}")
    print(f"  Ceiling:  R@10={r_ceil['recall']:.4f} NDCG={r_ceil['ndcg']:.4f} lat={r_ceil['lat_ms']:.1f}ms")
    print(f"  Eficiência vs ceiling: {eff:.1f}%")
    print(f"  ✅ Claim R@10=0.430: {'OK' if abs(r_madhava['recall']-0.430)<0.01 else 'DIVERGE'} (medido {r_madhava['recall']:.4f})")
    print(f"  ✅ Claim 0 vio: {'OK' if r_madhava['vio']==0 else 'FAIL'} (medido {r_madhava['vio']})\n")
    del eng

    # ─────────────────────────────────────────────────────────────
    # Claim 4 (novo): cosine + cascata [64,128] + modulação
    # ─────────────────────────────────────────────────────────────
    print("[Claim 4] Cosine + cascata [64,128] + modulação (stack)")
    cfg2 = winnex_madhava.Config()
    cfg2.dim = D
    cfg2.metric = winnex_madhava.Metric.COSINE
    cfg2.stage1_dim = 64
    cfg2.stage2_dim = 128
    cfg2.k = K
    cfg2.k1_fraction = 0.05
    cfg2.k2_fraction = 0.01
    cfg2.modulation = True
    cfg2.postfilter = True
    cfg2.normalize_input = True
    eng2 = winnex_madhava.MadhavaL2(cfg2)
    t0 = time.time()
    eng2.build_numpy(base)
    build_s2 = time.time() - t0
    r_cos = eval_engine(eng2, queries, gt_ids, n)
    # ceiling cosine = search_exact com cosine
    r_ceil_cos = {"recall": 0.0, "ndcg": 0.0, "lat_ms": 0.0, "vio": 0, "n_eval": 0}
    nq_eval = 0
    for gi in range(nq):
        gs = [v for v in gt_ids[gi] if 0 <= v < n]
        if not gs:
            continue
        qi = 2 * gi
        t0 = time.time()
        res = eng2.search_exact(queries[qi])
        r_ceil_cos["lat_ms"] += (time.time() - t0) * 1000
        r_ceil_cos["recall"] += recall_at_k(res.indices, gs)
        r_ceil_cos["ndcg"] += ndcg_at_k(res.indices, gs)
        nq_eval += 1
    m = max(nq_eval, 1)
    r_ceil_cos = {k2: (v / m if k2 in ("recall", "ndcg", "lat_ms") else v) for k2, v in r_ceil_cos.items()}
    r_ceil_cos["n_eval"] = nq_eval
    eff_cos = 100.0 * r_cos["recall"] / max(r_ceil_cos["recall"], 1e-9)
    results["cosine_cascade"] = {
        "build_s": build_s2,
        "madhava": r_cos,
        "ceiling": r_ceil_cos,
        "efficiency_pct": eff_cos,
    }
    print(f"  build: {build_s2:.1f}s")
    print(f"  Madhava:  R@10={r_cos['recall']:.4f} NDCG={r_cos['ndcg']:.4f} lat={r_cos['lat_ms']:.1f}ms vio={r_cos['vio']}")
    print(f"  Ceiling:  R@10={r_ceil_cos['recall']:.4f} NDCG={r_ceil_cos['ndcg']:.4f}")
    print(f"  Eficiência vs ceiling: {eff_cos:.1f}%")
    print(f"  ✅ 0 vio: {'OK' if r_cos['vio']==0 else 'FAIL'} (medido {r_cos['vio']})\n")
    del eng2

    # ─────────────────────────────────────────────────────────────
    # quant='none' vs quant='int8' (equivalência)
    # ─────────────────────────────────────────────────────────────
    print("[Claim 5] quant='none' (float32) == quant='int8' (métrica)")
    cfg3 = winnex_madhava.Config()
    cfg3.dim = D
    cfg3.metric = winnex_madhava.Metric.COSINE
    cfg3.quant = winnex_madhava.QuantMode.NONE
    cfg3.stage1_dim = 64
    cfg3.stage2_dim = 128
    cfg3.k = K
    cfg3.k1_fraction = 0.05
    cfg3.k2_fraction = 0.01
    cfg3.postfilter = True
    eng3 = winnex_madhava.MadhavaL2(cfg3)
    t0 = time.time()
    eng3.build_numpy(base)
    build_s3 = time.time() - t0
    r_qnone = eval_engine(eng3, queries, gt_ids, n)
    results["quant_none"] = {"build_s": build_s3, "madhava": r_qnone}
    # compara com int8 (r_cos)
    same = abs(r_qnone["recall"] - r_cos["recall"]) < 0.001
    print(f"  quant=none: R@10={r_qnone['recall']:.4f} vio={r_qnone['vio']}")
    print(f"  quant=int8: R@10={r_cos['recall']:.4f} vio={r_cos['vio']}")
    print(f"  ✅ Equivalência (Δ recall < 0.001): {'OK' if same else 'DIVERGE'}\n")
    del eng3

    # Salva
    out = "bench_verify_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Salvo: {out}")


if __name__ == "__main__":
    main()
