#!/usr/bin/env python3
"""
UB Width benchmark — winnex-madhava 1.8.4 on the real Kaggle BIGANN-100M dataset.

Zero synthetic data, zero simulations, zero pipeline manipulation.
Installs winnex-madhava from PyPI and uses ONLY its public API:
  - build_engine(corpus, basis="pca_corpus")   # the UB Width mode (native)
  - build_engine(corpus, basis="random")       # the historical default
  - engine.search_exact(...)                   # the exact-scan ceiling

The PCA basis is computed INSIDE the C++ engine (BasisMode::PCACorpus); the
kernel does not compute any basis, residuals, or ground truth itself. The
ground truth is the engine's OWN exact scan (the valid ceiling — the official
BIGANN GT is invalid for this Kaggle base, documented in the project).

Measures what the MOTOR returns: recall@10 vs its own exact scan, pruning
(survivors the post-filter evaluates), and bound violations.
"""
import numpy as np
import subprocess, sys, time, os, json

print("=" * 70, flush=True)
print("UB Width benchmark — winnex-madhava on real BIGANN-100M")
print("=" * 70, flush=True)

# --- 1. Install from PyPI (the released package) ---
print("\n[1] Installing winnex-madhava from PyPI...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-madhava==1.8.4"])
import winnex_madhava as wm
print(f"    winnex-madhava {wm.__version__} installed from PyPI", flush=True)

# --- 2. Locate the real Kaggle dataset (robust: find by filename) ---
print("\n[2] Locating the BIGANN-100M dataset...", flush=True)
def find_file(name):
    for r, d, files in os.walk("/kaggle/input/"):
        if name in files:
            return os.path.join(r, name)
    return None
base_path = find_file("base.u8bin")
qpath = find_file("unif_query_10k.u8bin")
if base_path is None or qpath is None:
    raise RuntimeError("BIGANN-100M dataset not found (base.u8bin or queries missing)")
print(f"    base: {base_path}", flush=True)
print(f"    queries: {qpath}", flush=True)

# --- 3. Load subset + official queries (no simulation) ---
N = 200_000          # subset of the real base
NQ = 100
K = 10
base = np.memmap(base_path, dtype=np.uint8, mode="r", shape=(100_000_000, 128))
X = np.ascontiguousarray(base[:N])
print(f"\n[3] Subset: {X.shape} (first {N} of 100M real vectors)", flush=True)

with open(qpath, "rb") as f:
    nq_hdr = np.fromfile(f, dtype=np.int32, count=1)[0]
    dim_hdr = np.fromfile(f, dtype=np.int32, count=1)[0]
Q = np.fromfile(qpath, dtype=np.uint8, offset=8).reshape(nq_hdr, dim_hdr)[:NQ]
Q = np.ascontiguousarray(Q)
print(f"    queries: {Q.shape} (official, {NQ} of 10K)", flush=True)

# --- 4. Engines via the public API only ---
print("\n[4] Building engines (public API)...", flush=True)
# UB Width mode (native PCA basis computed in C++)
t0 = time.time()
eng_ub = wm.build_engine(np.ascontiguousarray(X.astype(np.float32)),
                         dim=128, metric="cosine", quant="none",
                         basis="pca_corpus", stage1_dim=64, stage2_dim=0,
                         k=K, normalize_input=True)
build_ub = time.time() - t0
print(f"    UB Width engine built in {build_ub:.1f}s", flush=True)

# Random basis (the historical default)
t0 = time.time()
eng_rand = wm.build_engine(np.ascontiguousarray(X.astype(np.float32)),
                           dim=128, metric="cosine", quant="none",
                           basis="random", stage1_dim=64, stage2_dim=0,
                           k=K, normalize_input=True)
build_rand = time.time() - t0
print(f"    Random engine built in {build_rand:.1f}s", flush=True)

# --- 5. Measure what the MOTOR returns ---
print("\n[5] Measuring recall/pruning/violations (what the engine returns)...", flush=True)

def measure(eng, Qn, K=10):
    """Recall of the MOTOR's search() vs its own search_exact(), SAME query.

    The recall is computed per-query: for each j, compare the indices the
    search() returned for query j against the exact top-10 for the SAME query
    j. This is the honest protocol (no cross-query comparison).
    """
    viol = 0
    surv = 0
    lat = 0.0
    rec_ex = 0.0
    for j in range(len(Qn)):
        t0 = time.time()
        r = eng.search(Qn[j])
        lat += (time.time() - t0) * 1000
        viol += r.bound_violations
        surv += r.k3
        # exact scan for the SAME query j (the valid ceiling)
        r_ex = eng.search_exact(Qn[j])
        rec_ex += sum(1 for i in r.indices if i in r_ex.indices) / K
    return rec_ex / len(Qn), viol, surv / len(Qn), lat / len(Qn)

Qf = Q.astype(np.float32)
rec_ub, viol_ub, surv_ub, lat_ub = measure(eng_ub, Qf)
rec_rand, viol_rand, surv_rand, lat_rand = measure(eng_rand, Qf)

print(f"\n    UB Width : recall@10={rec_ub:.4f}  viol={viol_ub}  "
      f"prune={(1-surv_ub/N)*100:.1f}%  lat={lat_ub:.1f}ms  build={build_ub:.1f}s", flush=True)
print(f"    Random   : recall@10={rec_rand:.4f}  viol={viol_rand}  "
      f"prune={(1-surv_rand/N)*100:.1f}%  lat={lat_rand:.1f}ms  build={build_rand:.1f}s", flush=True)

# --- 6. Results ---
out = {
    "package": "winnex-madhava", "version": wm.__version__,
    "dataset": "shurangwu/bigann-100m",
    "N_subset": N, "NQ": NQ, "dim": 128, "K": K,
    "metric": "cosine", "pca_rank": 64,
    "gt": "engine's own exact scan (valid ceiling; official GT invalid for this base — documented)",
    "results": {
        "ub_width": {"recall@10": round(rec_ub, 4), "violations": int(viol_ub),
                     "prune_pct": round((1-surv_ub/N)*100, 1),
                     "lat_ms": round(lat_ub, 2), "build_s": round(build_ub, 1)},
        "random": {"recall@10": round(rec_rand, 4), "violations": int(viol_rand),
                   "prune_pct": round((1-surv_rand/N)*100, 1),
                   "lat_ms": round(lat_rand, 2), "build_s": round(build_rand, 1)},
    },
}
with open("/kaggle/working/ubwidth_bigann_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n" + "=" * 70)
print("RESULTS")
print(json.dumps(out, indent=2))
print("=" * 70)
