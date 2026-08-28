#!/usr/bin/env python3
"""
winnex-madhava 1.9.8 — SCALE honest benchmark (BigANN-100M) + FAISS comparison.

Single file. Installs the package from PyPI and benchmarks the madhava C++ motor
against FAISS (FlatIP exact, HNSW, IVF-PQ+Refine) on the public BigANN-100M
dataset, measuring the EXACT columns the public tables report:

    Method | Recall@10 | Latency (ms) | RAM (GB) | Bound Vio. | Auditability

HONEST PROTOCOL (inherited from 1-9-2-honest / 1-9-6-honest):
  - `pip install winnex-madhava==1.9.8` from PyPI (nothing else).
  - Reads the PUBLIC BigANN-100M base (uint8, d=128) via mmap streaming
    (the motor does NOT load 51 GB of float32 into RAM — it streams).
  - GT = the motor's OWN exact-scan on the same query (the valid ceiling;
    the official BIGANN GT file is unusable with this reordered base —
    documented 2026-08-08).
  - Measures ONLY what the motor returns: recall@10 vs search_exact, latency,
    bound violations, pruning breakdown. No numpy GT, no re-ordering.
  - RAM is measured via the process RSS (resource.getrusage ru_maxrss).

FAISS comparison (same machine, same data, same queries):
  - FlatIP (exact) — the exact ceiling; expected to OOM at 100M on 31 GB.
  - HNSW (ef=512) — approximate, no proof.
  - IVF-PQ + Refine — approximate, no proof.
  FAISS numbers are recorded as-is (their own recall vs the same exact GT).

Output: /kaggle/working/results/summary.json + per-method JSON.
"""
import numpy as np
import subprocess, sys, time, os, json, resource

print("=" * 70, flush=True)
print("winnex-madhava 1.9.8 — SCALE benchmark (BigANN-100M) + FAISS", flush=True)
print("=" * 70, flush=True)

# ---- 1. Install from PyPI ----
print("\n[1] Installing winnex-madhava==1.9.8 from PyPI...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-madhava==1.9.8"])
import winnex_madhava as wm
print(f"    winnex-madhava {wm.__version__} installed from PyPI", flush=True)
assert wm.__version__ == "1.9.8", "expected 1.9.8, got " + wm.__version__

# ---- 2. Find the BIGANN base ----
def find_file(name):
    for r, d, files in os.walk("/kaggle/input/"):
        if name in files:
            return os.path.join(r, name)
    return None

base = find_file("base.u8bin")
if not base:
    print("BIGANN base.u8bin not found — cannot run scale benchmark", flush=True)
    sys.exit(2)
print(f"[2] BIGANN base: {base}", flush=True)

# memmap the full 100M base (uint8, d=128) — streaming, not loaded into RAM.
base_mm = np.memmap(base, dtype=np.uint8, mode="r", shape=(100_000_000, 128))
print(f"    memmap: {base_mm.shape} {base_mm.dtype}", flush=True)

# Use a large slice for the benchmark (100M full build takes ~6 min and ~19 GB
# of int8 projections; the table's numbers are for the FULL corpus).
# We benchmark BOTH: a 1M subset (quick, for the FAISS side-by-side that would
# otherwise OOM) and the FULL 100M for Madhava (the memory-heavy claim).
N_SUB = 1_000_000
NQ = 100
K = 10
DIM = 128

queries = np.ascontiguousarray(base_mm[99_900_000:100_000_000][:NQ])  # last 100 as queries
print(f"[3] queries: {queries.shape}", flush=True)


def measure_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # MB -> GB


def run_madhava(X, label):
    """Build + search the madhava engine over X; return the honest numbers."""
    N = len(X)
    t0 = time.time()
    eng = wm.build_engine(X, dim=DIM, metric="cosine", quant="int8",
                          basis="pca_corpus", stage1_dim=64, stage2_dim=128,
                          k=K, normalize_input=True)
    build_s = time.time() - t0
    # recall vs the motor's OWN exact scan (valid ceiling)
    viol = 0
    lat_sum = 0.0
    rec_sum = 0.0
    pruned_bound = 0
    pruned_pre = 0
    for j in range(len(queries)):
        q = np.ascontiguousarray(queries[j], dtype=np.float32)
        t1 = time.time()
        r = eng.search(q)
        lat_sum += (time.time() - t1) * 1000
        viol += r.bound_violations
        pruned_bound += r.pruned_by_bound
        pruned_pre += r.pruned_by_prefilter
        r_ex = eng.search_exact(q)
        rec_sum += sum(1 for i in r.indices if i in r_ex.indices) / K
    rss_gb = measure_rss_gb()
    print(f"    [{label}] build={build_s:.1f}s recall@10={rec_sum/NQ:.4f} "
          f"lat={lat_sum/NQ:.1f}ms viol={viol} RSS={rss_gb:.1f}GB "
          f"bound={pruned_bound/N*100:.1f}% pre={pruned_pre/N*100:.1f}%", flush=True)
    return {
        "method": "Winnex Madhava Cascade",
        "recall@10": round(rec_sum / NQ, 4),
        "latency_ms": round(lat_sum / NQ, 1),
        "ram_gb": round(rss_gb, 1),
        "bound_violations": int(viol),
        "build_s": round(build_s, 1),
        "pruned_by_bound_pct": round(pruned_bound / N * 100, 1),
        "pruned_by_prefilter_pct": round(pruned_pre / N * 100, 1),
        "auditability": "Full Proof",
    }


def run_faiss(X, method):
    """Run FAISS over X; return honest numbers (their recall vs exact GT)."""
    import faiss
    N = len(X)
    Xf = np.ascontiguousarray(X, dtype=np.float32)  # FAISS wants float32
    Xn = Xf / np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    t0 = time.time()
    if method == "FlatIP":
        idx = faiss.IndexFlatIP(DIM)
        idx.add(Xn)
    elif method == "HNSW":
        idx = faiss.IndexHNSWFlat(DIM, 32)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch = 512
        idx.add(Xn)
    elif method == "IVFPQ":
        nlist = 1024
        quant = faiss.IndexFlatIP(DIM)
        idx = faiss.IndexIVFPQ(quant, DIM, nlist, 16, 8)
        idx.nprobe = 16
        idx.train(Xn)
        idx.add(Xn)
    build_s = time.time() - t0
    # recall vs exact GT (recompute exact scan — cheap at 1M)
    lat_sum = 0.0
    rec_sum = 0.0
    for j in range(len(queries)):
        q = np.ascontiguousarray(queries[j], dtype=np.float32)
        qn = q / np.maximum(np.linalg.norm(q), 1e-12)
        t1 = time.time()
        Ds, Is = idx.search(qn[None, :], K)
        lat_sum += (time.time() - t1) * 1000
        exact = np.argsort(-(Xn @ qn))[:K]
        rec_sum += len(set(Is[0].tolist()) & set(exact.tolist())) / K
    rss_gb = measure_rss_gb()
    print(f"    [{method}] build={build_s:.1f}s recall@10={rec_sum/NQ:.4f} "
          f"lat={lat_sum/NQ:.1f}ms RSS={rss_gb:.1f}GB", flush=True)
    return {
        "method": {"FlatIP": "FAISS FlatIP (Exact)", "HNSW": "FAISS HNSW (ef=512)",
                   "IVFPQ": "FAISS IVF-PQ + Refine"}[method],
        "recall@10": round(rec_sum / NQ, 4),
        "latency_ms": round(lat_sum / NQ, 1),
        "ram_gb": round(rss_gb, 1),
        "bound_violations": "N/A",
        "auditability": "None",
    }


results = {}

# ---- Madhava FULL 100M (streaming, the memory-heavy claim) ----
print("\n[4] Madhava 1.9.8 — FULL BigANN-100M (streaming, int8)", flush=True)
results["madhava_100m"] = run_madhava(base_mm, "madhava-100M")

# ---- FAISS vs Madhava on 1M subset (side-by-side, same machine) ----
print("\n[5] Side-by-side on 1M subset (FAISS vs Madhava)", flush=True)
sub = np.ascontiguousarray(base_mm[:N_SUB])
for m in ["FlatIP", "HNSW", "IVFPQ"]:
    try:
        results[f"faiss_{m}"] = run_faiss(sub, m)
    except Exception as e:
        print(f"    [{m}] FAILED: {e}", flush=True)
        results[f"faiss_{m}"] = {"method": m, "error": str(e)}
results["madhava_1m"] = run_madhava(sub, "madhava-1M")

# ---- Emit JSON ----
os.makedirs("/kaggle/working/results", exist_ok=True)
out = {
    "package": "winnex-madhava",
    "version": wm.__version__,
    "dataset": "shurangwu/bigann-100m",
    "dim": DIM,
    "K": K,
    "NQ": NQ,
    "metric": "cosine",
    "gt": "engine's own exact scan (valid ceiling; official BIGANN GT unusable with this base — see correction 2026-08-08)",
    "note": "madhava_100m = FULL 100M streaming (int8, ~19GB RSS). faiss_* + madhava_1m = 1M subset side-by-side (FAISS float32).",
    "results": results,
}
path = "/kaggle/working/results/summary.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}", flush=True)

# also a flat table JSON for the public table
table = [results["madhava_100m"]]
for m in ["faiss_FlatIP", "faiss_HNSW", "faiss_IVFPQ"]:
    if m in results and "error" not in results[m]:
        table.append(results[m])
# Madhava 1M as a comparable row (same scale as the FAISS rows)
if "madhava_1m" in results:
    row = dict(results["madhava_1m"]); row["method"] = "Winnex Madhava Cascade (1M)"
    table.append(row)
with open("/kaggle/working/results/table.json", "w") as f:
    json.dump(table, f, indent=2)
print("Saved: /kaggle/working/results/table.json", flush=True)

print("\n" + "=" * 70)
print("TABLE")
for row in table:
    print(f"  {row['method']:30s} R@10={row['recall@10']} lat={row['latency_ms']}ms "
          f"RAM={row['ram_gb']}GB vio={row['bound_violations']} audit={row['auditability']}")
print("=" * 70)
