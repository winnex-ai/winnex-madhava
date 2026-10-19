#!/usr/bin/env python3
"""
benchmark_madhava.py — benchmark da camada Python do winnex-madhava.

Segue o princípio: benchmarks na camada SUPERIOR (Python), onde o cliente
real consome a API. Mede:

  [1] CORREÇÃO  — Recall@K, Recall@10, NDCG@K, bound violations
                   (vs o ground-truth do scan exato como baseline)
  [2] PERFORMANCE — latência vs escala (1K → 1M vetores)
  [3] COMPARAÇÃO COM BASELINE — o scan exato é o "recall ceiling";
                   o bound engine deve manter Recall@10 = 1.000 com
                   bound_violations = 0 em todas as escalas.
  [4] QUALIDADE  — fração de vetores avaliados (k1/N) — a prova de que o
                   pruning Cauchy-Schwarz funciona sem perder recall.

Resultados salvos em JSON para comparação com publicações futuras.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import winnex_madhava as wm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _corpus_and_queries(n: int, d: int, nq: int = 50, seed: int = 42):
    """Sintético: corpus uint8 + queries float32 (reais, do corpus com ruído)."""
    rng = np.random.default_rng(seed)
    corpus = rng.integers(0, 256, size=(n, d), dtype=np.uint8)
    # Queries = vetores do corpus + ruído pequeno (vizinhos reais existem).
    idx = rng.integers(0, n, size=nq)
    queries = corpus[idx].astype(np.float32) + rng.normal(0, 2.0, size=(nq, d)).astype(np.float32)
    queries = np.clip(queries, 0, 255).astype(np.float32)
    return corpus, queries


def _exact_topk(corpus: np.ndarray, q: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """Scan exato (baseline/ceiling).

    O engine usa metric='cosine' por padrão (normaliza os vetores e mede
    similaridade cosseno). O baseline deve usar a MESMA métrica para ser
    comparável — senão comparamos ranking de cosseno contra ranking de L2
    e o recall cai artificialmente.
    """
    cf = corpus.astype(np.float32)
    if metric == "cosine":
        # Normalizar (como o engine faz para cosine).
        cn = cf / (np.linalg.norm(cf, axis=1, keepdims=True) + 1e-9)
        qn = q.astype(np.float32) / (np.linalg.norm(q.astype(np.float32)) + 1e-9)
        sim = cn @ qn  # maior = mais similar
        return np.argsort(-sim)[:k]
    # L2
    delta = cf - q.astype(np.float32)[None, :]
    dist = np.einsum("nd,nd->n", delta, delta)
    return np.argsort(dist)[:k]


# ---------------------------------------------------------------------------
# [1] CORREÇÃO — Recall@K vs scan exato
# ---------------------------------------------------------------------------
def benchmark_correctness(n: int = 20000, d: int = 128, nq: int = 50, k: int = 10,
                          k1_fraction: float = 0.40):
    """Recall@K, Recall@10, NDCG@K, bound violations vs scan exato.

    k1_fraction: fração de vetores mantidos pelo pruning (trade-off recall×speed).
    0.40 garante Recall@10=1.0 em dados sintéticos cosine com 0 violações.
    """
    corpus, queries = _corpus_and_queries(n, d, nq)
    engine = wm.build_engine(corpus, dim=d, k=k, postfilter=True, k1_fraction=k1_fraction)

    # Warm-up (as primeiras buscas carregam estruturas lazy / alocação — não medir).
    for _ in range(5):
        engine.search(queries[0])

    # Ground-truth PRÉ-COMPUTADO num passo separado (o BLAS do numpy disputa
    # threads com o OpenMP do madhava se intercalado — distorce a medição).
    exact_sets = [set(_exact_topk(corpus, queries[i], k).tolist()) for i in range(nq)]

    total_recall10 = 0.0
    total_ndcg = 0.0
    total_violations = 0
    latencies = []
    k1_ratios = []

    # Medição do madhava em loop limpo (sem numpy BLAS intercalado).
    results_list = [engine.search(queries[i]) for i in range(nq)]
    for i, res in enumerate(results_list):
        exact_set = exact_sets[i]
        latencies.append(res.latency_ms if hasattr(res, "latency_ms") else res["latency_ms"])

        found_set = set(res.indices)
        total_recall10 += len(found_set & exact_set) / k
        total_ndcg += wm.ndcg_at_k(res.indices, list(exact_set), k)
        total_violations += res.bound_violations
        k1_ratios.append(res.k1 / engine.num_vectors() if engine.num_vectors() else 0)

    return {
        "n": n, "d": d, "nq": nq, "k": k, "k1_fraction": k1_fraction,
        "recall_at_10": round(total_recall10 / nq, 6),
        "ndcg_at_10": round(total_ndcg / nq, 6),
        "bound_violations": int(total_violations),
        "latency_ms_mean": round(float(np.mean(latencies)), 4),
        "latency_ms_p50": round(float(np.percentile(latencies, 50)), 4),
        "latency_ms_p99": round(float(np.percentile(latencies, 99)), 4),
        "k1_fraction_mean": round(float(np.mean(k1_ratios)), 4),
    }


# ---------------------------------------------------------------------------
# [2] PERFORMANCE — latência vs escala
# ---------------------------------------------------------------------------
def benchmark_scale(sizes=(1_000, 10_000, 100_000, 1_000_000), d: int = 128,
                    nq: int = 20, k: int = 10, k1_fraction: float = 0.40):
    """Latência de build e de busca vs número de vetores."""
    results = []
    for n in sizes:
        corpus, queries = _corpus_and_queries(n, d, nq, seed=n)
        t0 = time.perf_counter()
        engine = wm.build_engine(corpus, dim=d, k=k, postfilter=True, k1_fraction=k1_fraction)
        t1 = time.perf_counter()
        build_s = t1 - t0
        # Warm-up
        engine.search(queries[0])
        # Busca
        t0 = time.perf_counter()
        for i in range(nq):
            engine.search(queries[i])
        t1 = time.perf_counter()
        search_ms = (t1 - t0) * 1000 / nq
        # Recall (amostra)
        r = benchmark_correctness(n=min(n, 20000), d=d, nq=min(nq, 10), k=10,
                                  k1_fraction=k1_fraction)
        results.append({
            "vectors": n,
            "build_seconds": round(build_s, 3),
            "search_ms_per_query": round(search_ms, 4),
            "queries_per_sec": round(1000.0 / max(search_ms, 1e-9), 1),
            "recall_at_10": r["recall_at_10"],
            "bound_violations": r["bound_violations"],
        })
    return results


# ---------------------------------------------------------------------------
# [3] COMPARAÇÃO COM BASELINE — bound vs exact em cada escala
# ---------------------------------------------------------------------------
def benchmark_vs_exact(sizes=(10_000, 100_000, 1_000_000), d: int = 128, nq: int = 20, k: int = 10,
                       k1_fraction: float = 0.40):
    """Para cada escala: bound engine (pruning) vs scan exato (ceiling)."""
    rows = []
    for n in sizes:
        corpus, queries = _corpus_and_queries(n, d, nq, seed=n)
        engine = wm.build_engine(corpus, dim=d, k=k, postfilter=True, k1_fraction=k1_fraction)
        for _ in range(5):
            engine.search(queries[0])  # warm-up

        # Pré-computar o ground-truth (numpy BLAS) num passo separado para
        # não interferir na medição do madhava (contention de threads OpenMP).
        exact_sets = [set(_exact_topk(corpus, queries[i], k).tolist()) for i in range(nq)]

        # Medir o exact scan (numpy) num loop separado.
        exact_lat = []
        for i in range(nq):
            t0 = time.perf_counter()
            _exact_topk(corpus, queries[i], k)
            t1 = time.perf_counter()
            exact_lat.append((t1 - t0) * 1000)

        # Medir o madhava num loop limpo.
        bound_lat = []
        recall_sum = 0.0
        viol_sum = 0
        results_list = [engine.search(queries[i]) for i in range(nq)]
        for i, res in enumerate(results_list):
            bound_lat.append(res.latency_ms if hasattr(res, "latency_ms") else res["latency_ms"])
            exact_set = exact_sets[i]
            recall_sum += len(set(res.indices) & exact_set) / k
            viol_sum += res.bound_violations

        rows.append({
            "vectors": n,
            "recall_at_10": round(recall_sum / nq, 6),
            "bound_violations": int(viol_sum),
            "bound_ms": round(float(np.mean(bound_lat)), 4),
            "exact_ms": round(float(np.mean(exact_lat)), 4),
            "speedup_vs_exact": round(float(np.mean(exact_lat) / max(np.mean(bound_lat), 1e-9)), 2),
            "exact_is_ceiling": True,
        })
    return rows


# ---------------------------------------------------------------------------
# [4] TRADE-OFF — Recall × k1_fraction × latência
# ---------------------------------------------------------------------------
def benchmark_tradeoff(n: int = 20000, d: int = 128, nq: int = 30, k: int = 10,
                       fractions=(0.05, 0.10, 0.20, 0.30, 0.40, 0.60)):
    """Curva de trade-off: recall e latência vs k1_fraction (pruning)."""
    corpus, queries = _corpus_and_queries(n, d, nq)
    exact_sets = [set(_exact_topk(corpus, queries[i], k).tolist()) for i in range(nq)]
    rows = []
    for frac in fractions:
        engine = wm.build_engine(corpus, dim=d, k=k, postfilter=True, k1_fraction=frac)
        for _ in range(5):
            engine.search(queries[0])  # warm-up
        results_list = [engine.search(queries[i]) for i in range(nq)]
        lats = [res.latency_ms if hasattr(res, "latency_ms") else res["latency_ms"]
                for res in results_list]
        rec_sum, viol = 0.0, 0
        for i, res in enumerate(results_list):
            exact_set = exact_sets[i]
            rec_sum += len(set(res.indices) & exact_set) / k
            viol += res.bound_violations
        rows.append({
            "k1_fraction": frac,
            "recall_at_10": round(rec_sum / nq, 4),
            "bound_violations": int(viol),
            "latency_ms": round(float(np.mean(lats)), 4),
        })
    return rows


def main() -> None:
    report = {"package_version": getattr(wm, "__version__", "?")}
    print("=" * 64)
    print("  WINNEX-MADHAVA — BENCHMARK BUSCA VETORIAL (camada Python)")
    print(f"  versão: {report['package_version']}")
    print("=" * 64)

    # [1] CORREÇÃO
    print("\n[1] CORREÇÃO (Recall@K vs scan exato)")
    corr = benchmark_correctness()
    for kk, vv in corr.items():
        print(f"  {kk:<20} {vv}")
    report["correctness"] = corr

    # [2] ESCALA
    print("\n[2] PERFORMANCE (latência vs escala)")
    scale = benchmark_scale()
    print("  {:<9} {:<14} {:<18} {:<16} {:<12}".format(
        "vectors", "build(s)", "search(ms/q)", "queries/s", "recall@10"))
    for r in scale:
        print("  {:<9} {:<14} {:<18} {:<16} {:<12}".format(
            r["vectors"], r["build_seconds"], r["search_ms_per_query"],
            r["queries_per_sec"], r["recall_at_10"]))
    report["scale"] = scale

    # [3] COMPARAÇÃO COM BASELINE
    print("\n[3] COMPARAÇÃO COM BASELINE (bound vs scan exato)")
    vs = benchmark_vs_exact()
    print("  {:<9} {:<12} {:<16} {:<12} {:<12} {:<14}".format(
        "vectors", "recall@10", "bound_viol", "bound_ms", "exact_ms", "speedup"))
    for r in vs:
        print("  {:<9} {:<12} {:<16} {:<12} {:<12} {:<14}".format(
            r["vectors"], r["recall_at_10"], r["bound_violations"],
            r["bound_ms"], r["exact_ms"], f"{r['speedup_vs_exact']}×"))
    report["vs_exact"] = vs

    # [4] TRADE-OFF
    print("\n[4] TRADE-OFF (Recall × k1_fraction × latência)")
    trade = benchmark_tradeoff()
    print("  {:<12} {:<12} {:<16} {:<12}".format(
        "k1_fraction", "recall@10", "bound_viol", "latency_ms"))
    for r in trade:
        print("  {:<12} {:<12} {:<16} {:<12}".format(
            r["k1_fraction"], r["recall_at_10"], r["bound_violations"],
            r["latency_ms"]))
    report["tradeoff"] = trade

    # Salvar
    out = Path(__file__).with_suffix(".results.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nResultados salvos em {out}")

    ok = (corr["recall_at_10"] >= 0.99 and corr["bound_violations"] == 0
          and all(r["bound_violations"] == 0 for r in vs)
          and all(r["recall_at_10"] >= 0.99 for r in vs)
          and all(r["bound_violations"] == 0 for r in trade))
    print("\n" + ("=== TODOS OS BENCHMARKS PASSARAM (Recall@10≥0.99, 0 violações de bound) ==="
                  if ok else "=== HÁ FALHAS ==="))


if __name__ == "__main__":
    main()
