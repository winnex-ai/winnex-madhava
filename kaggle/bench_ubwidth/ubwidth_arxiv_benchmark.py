#!/usr/bin/env python3
"""
UB Width benchmark — winnex-madhava 1.8.6 on real d=1536 arXiv embeddings.

HONEST PROTOCOL — the motor C++ is the ONLY thing being measured.
  - Installs winnex-madhava from PyPI (the released package).
  - Reads real embeddings (tomtum/openai-arxiv-embeddings, d=1536).
  - Passes the RAW data to the C++ engine (build_engine). The engine does
    its own L2 normalization (normalize_input=True); the kernel does NOT
    pre-process the vectors with numpy.
  - Measures ONLY what the MOTOR returns: recall@10 vs the engine's own
    exact scan (same query), pruning (the motor's k3), and bound violations.
  - No numpy ground-truth, no re-ordering, no pipeline assembly.

The ground truth is the engine's OWN search_exact on the same query — the
valid ceiling. The kernel compares what the bound-search returns against it.
"""
import numpy as np
import subprocess, sys, time, os, json

print("=" * 70, flush=True)
print("UB Width benchmark — winnex-madhava (motor C++) on real d=1536 embeddings")
print("=" * 70, flush=True)

# --- 1. Install from PyPI (the released package) ---
print("\n[1] Installing winnex-madhava from PyPI...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-madhava==1.8.6"])
import winnex_madhava as wm
print(f"    winnex-madhava {wm.__version__} installed from PyPI", flush=True)

# --- 2. Locate the dataset ---
print("\n[2] Locating the arXiv embeddings dataset...", flush=True)
def find_file(name):
    for r, d, files in os.walk("/kaggle/input/"):
        if name in files:
            return os.path.join(r, name)
    return None
vec_path = find_file("vectors.dat")
if vec_path is None:
    # fallback: any .npy with embeddings
    npy_candidates = []
    for r, d, files in os.walk("/kaggle/input/"):
        for f in files:
            if f.endswith(".npy"):
                npy_candidates.append(os.path.join(r, f))
    if npy_candidates:
        vec_path = npy_candidates[0]
    else:
        raise RuntimeError("no vectors.dat or .npy embeddings found")
print(f"    vectors: {vec_path}", flush=True)

# --- 3. Load embeddings (RAW — no normalization, the motor handles it) ---
print("\n[3] Loading embeddings (raw, no numpy normalization)...", flush=True)
D = 1536           # OpenAI Ada-002 dimension
N_MAX = 200_000    # subset size

if vec_path.endswith(".npy"):
    X = np.load(vec_path, mmap_mode="r")
    X = np.ascontiguousarray(X[:N_MAX])
    D = X.shape[1]
    print(f"    .npy: shape={X.shape} dtype={X.dtype}", flush=True)
else:
    # vectors.dat is RAW float32 n×dim (NO header). Verify from file size:
    #   size / (4·dim) must be an integer. If not, dim is wrong.
    sz = os.path.getsize(vec_path)
    exact = sz / (D * 4)
    if not exact.is_integer():
        raise RuntimeError(f"file size {sz} not divisible by 4*dim={D*4}: "
                           f"not raw float32 n×{D} (format assumption wrong)")
    n_use = min(N_MAX, sz // (D * 4))
    X = np.memmap(vec_path, dtype=np.float32, mode="r", offset=0, shape=(n_use, D))
    X = np.ascontiguousarray(X[:n_use])
print(f"    subset: {X.shape} float32 (d={D}, raw, no offset)", flush=True)

# Queries: held-out vectors from the SAME distribution (last 100). Passed RAW.
N = len(X)
NQ = 100
K = 10
Q = X[-NQ:].copy()
Xc = X[:-NQ].copy()
N = len(Xc)
del X
print(f"    corpus: {Xc.shape}, queries: {Q.shape} (held-out tail, raw)", flush=True)

# --- 4. Build engines via the public API (the motor C++ computes everything) ---
print("\n[4] Building engines (public API — the motor does the normalization) ...", flush=True)
t0 = time.time()
eng_ub = wm.build_engine(Xc, dim=D, metric="cosine", quant="none",
                         basis="pca_corpus", stage1_dim=192, stage2_dim=0,
                         k=K, normalize_input=True)
build_ub = time.time() - t0
print(f"    UB Width engine built in {build_ub:.1f}s", flush=True)

t0 = time.time()
eng_rand = wm.build_engine(Xc, dim=D, metric="cosine", quant="none",
                           basis="random", stage1_dim=192, stage2_dim=0,
                           k=K, normalize_input=True)
build_rand = time.time() - t0
print(f"    Random engine built in {build_rand:.1f}s", flush=True)

# --- 5. Measure ONLY what the MOTOR returns (per-query) ---
print("\n[5] Measuring what the motor returns (per-query, vs its own exact scan)...", flush=True)

def measure(eng, Qn, K=10):
    """Recall of the motor's search() vs its OWN search_exact(), SAME query."""
    viol = 0
    surv = 0
    lat = 0.0
    rec = 0.0
    for j in range(len(Qn)):
        t0 = time.time()
        r = eng.search(Qn[j])
        lat += (time.time() - t0) * 1000
        viol += r.bound_violations
        surv += r.k3
        r_ex = eng.search_exact(Qn[j])
        rec += sum(1 for i in r.indices if i in r_ex.indices) / K
    # mean Cauchy-Schwarz residual reported by the motor (the bound width)
    resid = eng.residuals1()
    resid1 = float(np.mean(resid)) if len(resid) else None
    return rec / len(Qn), viol, surv / len(Qn), lat / len(Qn), resid1

rec_ub, viol_ub, surv_ub, lat_ub, e_ub = measure(eng_ub, Q)
rec_rand, viol_rand, surv_rand, lat_rand, e_rand = measure(eng_rand, Q)

print(f"\n    UB Width : recall@10={rec_ub:.4f}  viol={viol_ub}  "
      f"prune={(1-surv_ub/N)*100:.1f}%  lat={lat_ub:.1f}ms  build={build_ub:.1f}s  "
      f"residual e(v)={e_ub:.4f}", flush=True)
print(f"    Random   : recall@10={rec_rand:.4f}  viol={viol_rand}  "
      f"prune={(1-surv_rand/N)*100:.1f}%  lat={lat_rand:.1f}ms  build={build_rand:.1f}s  "
      f"residual e(v)={e_rand:.4f}", flush=True)

# --- 6. Results (only what the motor returned — no numpy GT, no re-ordering) ---
out = {
    "package": "winnex-madhava", "version": wm.__version__,
    "dataset": "tomtum/openai-arxiv-embeddings",
    "N_subset": N, "NQ": NQ, "dim": D, "K": K,
    "metric": "cosine", "pca_rank": 192,
    "gt": "the motor's own search_exact on the same query (valid ceiling)",
    "note": "raw data passed to the motor; no numpy normalization or re-ordering",
    "results": {
        "ub_width": {"recall@10": round(rec_ub, 4), "violations": int(viol_ub),
                     "prune_pct": round((1-surv_ub/N)*100, 1),
                     "lat_ms": round(lat_ub, 2), "build_s": round(build_ub, 1),
                     "residual_e_v": round(e_ub, 4)},
        "random": {"recall@10": round(rec_rand, 4), "violations": int(viol_rand),
                   "prune_pct": round((1-surv_rand/N)*100, 1),
                   "lat_ms": round(lat_rand, 2), "build_s": round(build_rand, 1),
                   "residual_e_v": round(e_rand, 4)},
    },
}
with open("/kaggle/working/ubwidth_arxiv_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n" + "=" * 70)
print("RESULTS")
print(json.dumps(out, indent=2))
print("=" * 70)
