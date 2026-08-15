#!/usr/bin/env python3
"""
winnex-madhava 1.8.6 benchmark — installs from PyPI, tests on 3 public Kaggle datasets.

HONEST PROTOCOL — the motor C++ is the ONLY thing being measured.
  - Installs winnex-madhava from PyPI (the released package, ==1.8.6).
  - Reads REAL public Kaggle datasets across 3 dimensions/domains:
      1. GloVe (rtatman/glove)              d=100  — word vectors
      2. BIGANN-100M (shurangwu/bigann-100m) d=128 — uint8 raw (L2)
      3. arXiv OpenAI (tomtum/openai-arxiv-embeddings) d=1536 — embeddings
  - Passes the RAW data to the C++ engine (build_engine). The engine does
    its own L2 normalization (normalize_input=True); the kernel does NOT
    pre-process with numpy.
  - Measures ONLY what the MOTOR returns: recall@10 vs the engine's own
    exact scan (same query), pruning (the motor's k3), and bound violations.
  - No numpy ground-truth, no re-ordering, no pipeline assembly.

The ground truth is the engine's OWN search_exact on the same query — the
valid ceiling. Compares what the bound-search returns against it.
"""
import numpy as np
import subprocess, sys, time, os, json

print("=" * 70, flush=True)
print("winnex-madhava (motor C++) benchmark — PyPI install, 3 public datasets")
print("=" * 70, flush=True)

# --- 1. Install from PyPI (the released package) ---
print("\n[1] Installing winnex-madhava from PyPI...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-madhava==1.8.6"])
import winnex_madhava as wm
print(f"    winnex-madhava {wm.__version__} installed from PyPI", flush=True)

# --- 2. Locate the datasets ---
print("\n[2] Locating the public Kaggle datasets...", flush=True)
def find_file(name):
    for r, d, files in os.walk("/kaggle/input/"):
        if name in files:
            return os.path.join(r, name)
    return None

datasets = {}
# 1. GloVe (d=100): glove.6B.100d.txt
glove_txt = find_file("glove.6B.100d.txt")
if glove_txt:
    datasets["glove_d100"] = {"kind": "glove", "path": glove_txt, "dim": 100}
# 2. BIGANN (d=128): base.u8bin
bigann = find_file("base.u8bin")
if bigann:
    datasets["bigann_d128"] = {"kind": "bigann", "path": bigann, "dim": 128}
# 3. arXiv OpenAI (d=1536): vectors.dat
vectors = find_file("vectors.dat")
if vectors:
    datasets["arxiv_d1536"] = {"kind": "vectors", "path": vectors, "dim": 1536}

if not datasets:
    raise RuntimeError("no public datasets found in /kaggle/input/")
print(f"    datasets found: {list(datasets.keys())}", flush=True)

# --- 3. Load + benchmark each dataset (RAW data, the motor does everything) ---
print("\n[3] Benchmarking the motor on each dataset...", flush=True)

def load_glove(path, n_max=50000, dim=100):
    """Load GloVe word vectors (first n_max rows) as float32."""
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= n_max:
                break
            parts = line.strip().split()
            if len(parts) >= dim + 1:
                rows.append([float(x) for x in parts[1:dim+1]])
    return np.asarray(rows, dtype=np.float32)

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
    resid = eng.residuals1()
    resid1 = float(np.mean(resid)) if len(resid) else None
    return rec / len(Qn), viol, surv / len(Qn), lat / len(Qn), resid1

all_results = {}
for name, ds in datasets.items():
    print(f"\n--- Dataset: {name} (d={ds['dim']}) ---", flush=True)
    # load raw
    if ds["kind"] == "glove":
        X = load_glove(ds["path"], n_max=20000, dim=ds["dim"])
        D = ds["dim"]
        print(f"    loaded {X.shape} (GloVe raw)", flush=True)
    elif ds["kind"] == "bigann":
        base = np.memmap(ds["path"], dtype=np.uint8, mode="r",
                         shape=(100_000_000, ds["dim"]))
        X = np.ascontiguousarray(base[:20000])
        D = ds["dim"]
        print(f"    loaded {X.shape} (BIGANN uint8)", flush=True)
    else:  # vectors.dat
        sz = os.path.getsize(ds["path"])
        exact = sz / (ds["dim"] * 4)
        if not exact.is_integer():
            print(f"    SKIP {name}: not raw float32 n x {ds['dim']}", flush=True)
            continue
        n_use = min(20000, sz // (ds["dim"] * 4))
        X = np.ascontiguousarray(np.memmap(ds["path"], dtype=np.float32,
                                           mode="r", offset=0,
                                           shape=(n_use, ds["dim"])))
        D = ds["dim"]
        print(f"    loaded {X.shape} (raw float32)", flush=True)

    N = len(X)
    NQ = 100
    K = 10
    # held-out queries from the same distribution (last 100)
    Q = X[-NQ:].copy()
    Xc = X[:-NQ].copy()
    N = len(Xc)

    # build + measure both modes
    res = {}
    for basis in ["random", "pca_corpus"]:
        t0 = time.time()
        eng = wm.build_engine(Xc, dim=D, metric="cosine", quant="none",
                              basis=basis, stage1_dim=min(192, D), stage2_dim=0,
                              k=K, normalize_input=True)
        tb = time.time() - t0
        rec, viol, surv, lat, e = measure(eng, Q)
        res[basis] = {"recall@10": round(rec, 4), "violations": int(viol),
                      "prune_pct": round((1 - surv / N) * 100, 1),
                      "lat_ms": round(lat, 2), "build_s": round(tb, 1),
                      "residual_e_v": round(e, 4) if e else None}
        print(f"    {basis:12s}: recall={rec:.4f} viol={viol} "
              f"prune={(1-surv/N)*100:.1f}% build={tb:.1f}s e(v)={e if e else 0:.4f}", flush=True)
    all_results[name] = {"dim": D, "N": N, "results": res}

# --- 4. Save ---
out = {
    "package": "winnex-madhava", "version": wm.__version__,
    "installed_from": "PyPI",
    "gt": "the motor's own search_exact on the same query (valid ceiling)",
    "note": "raw data passed to the motor; no numpy normalization or re-ordering",
    "datasets": all_results,
}
with open("/kaggle/working/ubwidth_multi_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n" + "=" * 70)
print("RESULTS")
print(json.dumps(out, indent=2))
print("=" * 70)
