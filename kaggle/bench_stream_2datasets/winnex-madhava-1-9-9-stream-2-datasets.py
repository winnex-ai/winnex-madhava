#!/usr/bin/env python3
"""
winnex-madhava 1.9.9 + winnex-ai-normalize 1.2.0 — STREAMING 2 DATASETS
=======================================================================
Benchmark final com 2 datasets grandes em STREAMING (memmap — o corpus
NUNCA é carregado na RAM). Motor + normalize do PyPI (limpo).

Datasets (raw, grandes, mmap):
  1. BIGANN-100M  (shurangwu/bigann-100m)  — base.u8bin, uint8, 100M×128,
     12.8 GB em disco; o motor processa em blocos de 500K, mantém só as
     projeções int8 (~19 GB) na RAM. `k2_max` limita o post-filter exato.
  2. arXiv OpenAI (tomtum/openai-arxiv-embeddings) — vectors.dat, float32,
     d=1536, ~44 GB; mmap raw float32. Embeddings reais de alta dimensão.

Fluxo (produção, normalize enxuto):
  pip install winnex-madhava==1.9.9 + winnex-ai-normalize==1.2.0
    → np.memmap(corpus)  →  build_engine (streaming, int8)  →  search
    → recall vs search_exact (ceiling válido), 0 violações, RSS (RAM) medida,
      pruning breakdown, AuditCommitment.

Protocolo honesto (herdado):
  - recall vs o próprio search_exact (ceiling válido, independente de GT externo).
  - RAM medida via ru_maxrss (pico real do processo).
  - Limitações declaradas: recall self-consistente (não vs rótulos semânticos);
    auto-executado (a reprodutibilidade no Kaggle é a mitigação).
"""
import numpy as np
import subprocess, sys, time, os, json, resource

print("=" * 72, flush=True)
print("winnex-madhava 1.9.9 + winnex-ai-normalize 1.2.0 — STREAMING 2 DATASETS", flush=True)
print("(BIGANN-100M uint8 + arXiv d=1536 — mmap, corpus nunca na RAM)", flush=True)
print("=" * 72, flush=True)

# ---- 1. Install from PyPI (limpo) ----
print("\n[1] Installing from PyPI...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-madhava==1.9.9"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-ai-normalize==1.2.0"])
import winnex_madhava as wm
import winnex_ai_normalize
from winnex_ai_normalize.core.quality import build_quality_engine
print(f"    winnex-madhava {wm.__version__} + winnex-ai-normalize {winnex_ai_normalize.__version__} installed from PyPI", flush=True)
assert wm.__version__ == "1.9.9"
assert winnex_ai_normalize.__version__ == "1.2.0"


def find_file(name):
    for r, d, files in os.walk("/kaggle/input/"):
        if name in files:
            return os.path.join(r, name)
    return None


def measure_rss_gb():
    """Pico de RAM do processo (ru_maxrss)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # MB -> GB


def measure(eng, Qn, K=10):
    """Recall vs search_exact, 0 violações, pruning breakdown, latência."""
    viol = 0; lat = 0.0; rec = 0.0; pb = 0; pp = 0; surv = 0
    for j in range(len(Qn)):
        t0 = time.time()
        r = eng.search(Qn[j])
        lat += (time.time() - t0) * 1000
        viol += r.bound_violations
        surv += r.k3
        pb += getattr(r, "pruned_by_bound", 0) or 0
        pp += getattr(r, "pruned_by_prefilter", 0) or 0
        r_ex = eng.search_exact(Qn[j])
        rec += sum(1 for i in r.indices if i in r_ex.indices) / K
    nq = len(Qn)
    return (rec / nq, viol, surv / nq, lat / nq, pb / nq, pp / nq)


def validate_commitment(eng, Qn, K=10, max_sample=50):
    n_q = len(Qn)
    n_b = 0; n_m = 0; n_d = 0; n_g = 0; total_excl = 0; max_payload = 0
    for j in range(n_q):
        c = eng.search_with_commitment(Qn[j], k=K, max_sample=max_sample)
        c2 = eng.search_with_commitment(Qn[j], k=K, max_sample=max_sample)
        ar = eng.search_audited(Qn[j], k=K, max_audit_records=max_sample)
        n_s = len(c["sampled_records"])
        total_excl += c["total_excluded_count"]
        payload = len(json.dumps(c).encode())
        max_payload = max(max_payload, payload)
        if n_s <= max_sample: n_b += 1
        if c["total_excluded_count"] == ar["audit_excluded"]: n_m += 1
        if [s["doc_id"] for s in c["sampled_records"]] == \
           [s["doc_id"] for s in c2["sampled_records"]]: n_d += 1
        thr = c["global_threshold"]
        if all(s["upper_bound"] < thr for s in c["sampled_records"]): n_g += 1
    return {"queries": n_q, "sample_bounded": n_b, "exact_count_match": n_m,
            "deterministic": n_d, "genuine_exclusions": n_g,
            "total_excluded": total_excl, "max_commitment_payload_bytes": max_payload}


def run_streaming(name, path, dtype, n_total, dim, preset=None,
                  n_use=None, NQ=20, K=10, metric="cosine"):
    """Streaming: memmap o corpus, NUNCA carrega na RAM, build + search.

    O corpus é mmap'ed (nunca em RAM). O motor processa em blocos de 500K
    (int8 projections na RAM, raw corpus no disco). RSS mede o pico real.
    """
    if n_use is None:
        n_use = n_total
    print(f"\n--- {name} (d={dim}, N_use={n_use}, corpus_total={n_total}, "
          f"file={os.path.basename(path)}) ---", flush=True)

    # memmap — o corpus NUNCA é carregado em RAM (streaming)
    if dtype == np.uint8:
        mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n_total, dim))
    else:
        mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n_total, dim))

    # queries = as últimas NQ do SUBSET indexado (n_use) como hold-out —
    # FIX (2026-08-31): as queries devem vir de DENTRO do n_use indexado,
    # senão (como na primeira versão) o recall vs search_exact dava 0 para o
    # BIGANN (queries do fim do arquivo de 100M, fora dos 5M indexados).
    assert n_use > NQ, "n_use deve ser maior que NQ (hold-out)"
    Q_raw = np.ascontiguousarray(mm[n_use - NQ:n_use][:NQ])
    if dtype == np.uint8:
        Qn = Q_raw.astype(np.float32)
    else:
        Qn = Q_raw
    Qn = Qn / np.maximum(np.linalg.norm(Qn, axis=1, keepdims=True), 1e-12)
    # corpus = os primeiros (n_use - NQ) (memmap, streaming) — as NQ últimas
    # viram hold-out e NÃO entram no índice.
    X = mm[:n_use - NQ]
    del Q_raw

    # RSS antes do build (baseline)
    rss0 = measure_rss_gb()

    # ---- Build (streaming) ----
    t0 = time.time()
    if dtype == np.uint8:
        # BIGANN raw bytes: metric l2, quant int8, k2_max limita o post-filter
        eng = wm.build_engine(X, dim=dim, metric="l2", quant="int8",
                              stage1_dim=64, stage2_dim=128, k=K,
                              k1_fraction=0.05, k2_fraction=0.01, k2_max=2000,
                              normalize_input=False, postfilter=True)
    else:
        # arXiv float32: cosine, presets por dataset do normalize
        if preset:
            eng, rep = build_quality_engine(
                X, dim=dim, k=K, return_report=True, dataset=preset,
                engine_kwargs=dict(metric="cosine", normalize_input=True))
        else:
            eng, rep = build_quality_engine(
                X, dim=dim, k=K, return_report=True,
                engine_kwargs=dict(metric="cosine", normalize_input=True))
    build_s = time.time() - t0
    rss_build = measure_rss_gb()

    # ---- Search (as NQ queries) ----
    rec, viol, surv, lat, pb, pp = measure(eng, Qn, K=K)
    commit = validate_commitment(eng, Qn, K=K, max_sample=50)
    rss_total = measure_rss_gb()

    # Queries para o search_exact (ceiling) — um subset para não pagar N scans exatos
    n_exact = min(NQ, 10)
    rec_exact, viol_exact, _, _, _, _ = measure(eng, Qn[:n_exact], K=K)
    del eng

    # N indexado real = n_use - NQ (as NQ últimas são hold-out)
    n_indexed = n_use - NQ
    print(f"    build={build_s:.1f}s  recall@10={rec:.4f}  viol={viol}  "
          f"lat={lat:.1f}ms  RSS_build={rss_build - rss0:.1f}GB  RSS_total={rss_total:.1f}GB", flush=True)
    print(f"    pb%={pb/max(n_indexed,1)*100:.1f}  pp%={pp/max(n_indexed,1)*100:.1f}  "
          f"survivor%={surv/max(n_indexed,1)*100:.1f}  "
          f"recall_exact_subset={rec_exact:.4f}", flush=True)
    print(f"    [commit] bounded={commit['sample_bounded']}/{commit['queries']}  "
          f"count={commit['exact_count_match']}/{commit['queries']}  "
          f"det={commit['deterministic']}/{commit['queries']}  "
          f"genuine={commit['genuine_exclusions']}/{commit['queries']}", flush=True)

    return {
        "package": "winnex-madhava", "version": wm.__version__,
        "normalize": f"winnex-ai-normalize {winnex_ai_normalize.__version__}",
        "dataset": name, "dim": int(dim), "N_use": int(n_indexed), "N_total": int(n_total),
        "dtype": str(dtype), "file_size_gb": round(os.path.getsize(path) / 1e9, 1),
        "metric": metric, "streaming": "mmap (corpus nunca na RAM)",
        "gt": "the motor's own search_exact on the same query (valid ceiling)",
        "results": {
            "recall@10": round(rec, 4), "bound_violations": int(viol),
            "pruned_by_bound_pct": round(pb / max(n_indexed, 1) * 100, 1),
            "pruned_by_prefilter_pct": round(pp / max(n_indexed, 1) * 100, 1),
            "survivor_pct": round(surv / max(n_indexed, 1) * 100, 1),
            "latency_ms": round(lat, 1), "build_s": round(build_s, 1),
            "ram_baseline_gb": round(rss0, 2), "ram_after_build_gb": round(rss_build, 2),
            "ram_total_gb": round(rss_total, 2),
            "corpus_on_disk_gb": round(os.path.getsize(path) / 1e9, 1),
            "commitment": commit,
        },
    }


# ---- 2. Segfault fix test (1.9.9) ----
print("\n[2] Segfault-fix test: build_engine(float32, pca_corpus, DEFAULT quant)...", flush=True)
try:
    rng = np.random.default_rng(7)
    N, D, K = 2000, 384, 10
    comp = rng.standard_normal((24, D)).astype(np.float32)
    coef = rng.standard_normal((N, 24)).astype(np.float32)
    X = (coef @ comp).astype(np.float32) * 0.35
    eng = wm.build_engine(X[:-100], dim=D, metric="cosine", basis="pca_corpus",
                          stage1_dim=min(192, D), stage2_dim=0, k=K, normalize_input=True)
    r = eng.search(X[-1])
    print(f"    OK — no segfault. recall@10="
          f"{sum(1 for i in r.indices if i in eng.search_exact(X[-1]).indices)/K}, "
          f"viol={r.bound_violations}", flush=True)
    segfault_fix = "PASS"
except Exception as e:
    print(f"    FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)
    segfault_fix = "FAIL"


# ---- 3. Stream the two datasets ----
out_all = {}
datasets_meta = {}

# 3.1 BIGANN-100M (uint8, 100M×128, streaming)
bigann = find_file("base.u8bin")
if bigann:
    sz = os.path.getsize(bigann)
    n_total = sz // 128
    print(f"[3.1] BIGANN base.u8bin: {sz/1e9:.1f} GB → {n_total}×128 uint8", flush=True)
    # Streaming real: memmap de 100M, usa um subset maior (ex.: 5M) para o benchmark
    n_use = min(5_000_000, n_total)
    datasets_meta["bigann_100m"] = {"file": "base.u8bin", "size_gb": round(sz / 1e9, 1)}
    out_all["bigann_5m_stream"] = run_streaming(
        "shurangwu/bigann-100m (streaming)", bigann, np.uint8, n_total, 128,
        n_use=n_use, NQ=20, K=10, metric="l2")
else:
    print("[3.1] BIGANN base.u8bin not found", flush=True)

# 3.2 arXiv OpenAI (float32, 44 GB, d=1536, streaming)
arxiv = find_file("vectors.dat")
if arxiv:
    sz = os.path.getsize(arxiv)
    if sz % (1536 * 4) == 0:
        n_total = sz // (1536 * 4)
        print(f"[3.2] arXiv vectors.dat: {sz/1e9:.1f} GB → {n_total}×1536 float32", flush=True)
        n_use = min(200_000, n_total)
        datasets_meta["arxiv_1536"] = {"file": "vectors.dat", "size_gb": round(sz / 1e9, 1)}
        out_all["arxiv_stream"] = run_streaming(
            "tomtum/openai-arxiv-embeddings (streaming)", arxiv, np.float32, n_total, 1536,
            n_use=n_use, NQ=20, K=10, metric="cosine", preset="arxiv")
    else:
        print(f"[3.2] arXiv vectors.dat size {sz} não é múltiplo de 1536*4", flush=True)
else:
    print("[3.2] arXiv vectors.dat not found", flush=True)

# ---- 4. Emit JSONs ----
os.makedirs("/kaggle/working/results", exist_ok=True)
for name, data in out_all.items():
    path = f"/kaggle/working/results/{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {path}", flush=True)

with open("/kaggle/working/results/summary.json", "w") as f:
    json.dump({
        "package": "winnex-madhava", "version": wm.__version__,
        "normalize": winnex_ai_normalize.__version__,
        "segfault_fix_float32_pca_default_quant": segfault_fix,
        "datasets": {k: v["results"] for k, v in out_all.items()},
        "datasets_meta": datasets_meta,
        "protocol": "honest — pip install winnex-madhava==1.9.9 + "
                    "winnex-ai-normalize==1.2.0 from PyPI; 2 datasets em "
                    "STREAMING (np.memmap, corpus nunca na RAM); recall vs "
                    "search_exact (ceiling válido); RSS medida; AuditCommitment "
                    "validado; sem GT externo (limitação declarada).",
    }, f, indent=2)

print("\n" + "=" * 72)
print("ALL RESULTS")
print(json.dumps({k: v["results"] for k, v in out_all.items()}, indent=2))
print("=" * 72)
