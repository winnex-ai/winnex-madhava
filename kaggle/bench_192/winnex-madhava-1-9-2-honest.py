#!/usr/bin/env python3
"""
winnex-madhava 1.9.2 benchmark — single file. Installs the package from PyPI
and tests the madhava C++ motor on public Kaggle datasets. Emits one
transparent JSON per dataset.

HONEST PROTOCOL (inherited from 1-9-1-honest):
  - `pip install winnex-madhava==1.9.2` from PyPI (nothing else).
  - Reads public Kaggle datasets (raw, no numpy pre-processing):
      1. GloVe  (rtatman/glove)               d=100
      2. BIGANN (shurangwu/bigann-100m)       d=128 (uint8)
      3. arXiv  (tomtum/openai-arxiv-embeddings) d=1536
  - Passes RAW data to the motor (build_engine). The motor normalizes and
    computes e(v) = sqrt(1 - ||Pv||^2) in its own manifold.
  - Measures ONLY what the motor returns: recall@10 vs the motor's own
    search_exact (same query), pruning (k3), bound violations, residual.
  - No numpy ground-truth, no re-ordering, no pipeline assembly.

NEW IN 1.9.2 (the AuditCommitment — production audit trail):
  - `search_with_commitment(q, k, max_sample)` returns a LIGHTWEIGHT
    commitment (total_excluded_count + global_threshold + a deterministic
    boundary sample) instead of the full O(N) certificate.
  - We validate:
      * max_sample is HONORED (the 1.9.1 search_audited returned ALL excluded
        docs regardless of max_audit_records — the ~2 MB/query problem);
      * total_excluded_count == search_audited.audit_excluded (exact);
      * determinism: same query -> same sampled doc_ids;
      * every sampled doc has upper_bound < global_threshold (genuine);
      * payload size (the WORM-friendly ~KB record, not ~MB).
  - This is the record the tracer-gov / tracer-med compliance flows store in
    the WORM (signed with Ed25519 by winnex-tracer-core).
"""
import numpy as np
import subprocess, sys, time, os, json

print("=" * 70, flush=True)
print("winnex-madhava 1.9.2 — PyPI install + AuditCommitment validation")
print("=" * 70, flush=True)

# ---- 1. Install from PyPI ----
print("\n[1] Installing winnex-madhava==1.9.2 from PyPI...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-madhava==1.9.2"])
import winnex_madhava as wm
print(f"    winnex-madhava {wm.__version__} installed from PyPI", flush=True)
assert wm.__version__ == "1.9.2", "expected 1.9.2, got " + wm.__version__
assert hasattr(wm.MadhavaL2, "search_with_commitment"), \
    "1.9.2 must expose search_with_commitment"


def find_file(name):
    for r, d, files in os.walk("/kaggle/input/"):
        if name in files:
            return os.path.join(r, name)
    return None


def measure(eng, Qn, K=10):
    """Recall of the motor's search() vs its OWN search_exact(), same query.

    Also captures the HONEST pruning breakdown the motor reports.
    """
    viol = 0; surv = 0; lat = 0.0; rec = 0.0; pb = 0; pp = 0
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
    resid = eng.residuals1()
    resid1 = float(np.mean(resid)) if len(resid) else None
    return (rec / len(Qn), viol, surv / len(Qn), lat / len(Qn), resid1,
            pb / len(Qn), pp / len(Qn))


def validate_commitment(eng, Qn, K=10, max_sample=50):
    """Validate the AuditCommitment (NEW in 1.9.2).

    For each query:
      1. search_with_commitment returns total_excluded + a boundary sample.
      2. max_sample is HONORED: len(sampled_records) <= max_sample.
      3. Exact consistency: total_excluded_count == search_audited's
         audit_excluded (the same global threshold).
      4. Determinism: two calls yield the same sampled doc_ids.
      5. Genuine exclusion: every sampled doc has upper_bound < threshold.
    Returns aggregate counts.
    """
    n_q = len(Qn)
    n_sample_bounded = 0          # queries where len(sample) <= max_sample
    n_exact_match = 0             # count == search_audited.audit_excluded
    n_deterministic = 0           # same sample on repeated call
    n_genuine = 0                 # all sampled ub < threshold
    total_excluded = 0
    max_payload = 0
    sample_rec = None

    for j in range(n_q):
        c = eng.search_with_commitment(Qn[j], k=K, max_sample=max_sample)
        c2 = eng.search_with_commitment(Qn[j], k=K, max_sample=max_sample)
        ar = eng.search_audited(Qn[j], k=K, max_audit_records=max_sample)

        n_s = len(c["sampled_records"])
        total_excluded += c["total_excluded_count"]
        payload = len(json.dumps(c).encode())
        max_payload = max(max_payload, payload)

        if n_s <= max_sample:
            n_sample_bounded += 1
        if c["total_excluded_count"] == ar["audit_excluded"]:
            n_exact_match += 1
        if [s["doc_id"] for s in c["sampled_records"]] == \
           [s["doc_id"] for s in c2["sampled_records"]]:
            n_deterministic += 1
        thr = c["global_threshold"]
        if all(s["upper_bound"] < thr for s in c["sampled_records"]):
            n_genuine += 1

        if sample_rec is None and c["sampled_records"]:
            sample_rec = c["sampled_records"][0]

    return {
        "queries": n_q,
        "sample_bounded": n_sample_bounded,
        "exact_count_match": n_exact_match,
        "deterministic": n_deterministic,
        "genuine_exclusions": n_genuine,
        "total_excluded": total_excluded,
        "max_commitment_payload_bytes": max_payload,
        "sample_record": sample_rec,
    }


def run_dataset(name, X, dim):
    NQ, K = 100, 10
    Q = X[-NQ:].copy()
    Xc = X[:-NQ].copy()
    N = len(Xc)
    print(f"\n--- {name} (d={dim}, N={N}) ---", flush=True)
    results = {}
    for basis in ["random", "pca_corpus"]:
        t0 = time.time()
        eng = wm.build_engine(Xc, dim=dim, metric="cosine", quant="none",
                              basis=basis, stage1_dim=min(192, dim),
                              stage2_dim=0, k=K, normalize_input=True)
        tb = time.time() - t0
        rec, viol, surv, lat, e, pb, pp = measure(eng, Q)
        commit = validate_commitment(eng, Q, K=K, max_sample=50)
        results[basis] = {
            "recall@10": round(rec, 4),
            "bound_violations": int(viol),
            "pruned_by_bound_pct": round(pb / N * 100, 1),
            "pruned_by_prefilter_pct": round(pp / N * 100, 1),
            "survivor_pct": round(surv / N * 100, 1),
            "latency_ms": round(lat, 2),
            "build_s": round(tb, 1),
            "residual_e_v": round(e, 4) if e else None,
            "commitment": commit,
        }
        print(f"    {basis:12s}: recall@10={rec:.4f}  viol={viol}  "
              f"build={tb:.1f}s  e(v)={e if e else 0:.4f}", flush=True)
        print(f"      [commit] sample_bounded={commit['sample_bounded']}/{commit['queries']}  "
              f"count_match={commit['exact_count_match']}/{commit['queries']}  "
              f"deterministic={commit['deterministic']}/{commit['queries']}  "
              f"genuine={commit['genuine_exclusions']}/{commit['queries']}  "
              f"max_payload={commit['max_commitment_payload_bytes']}B", flush=True)
    return {
        "package": "winnex-madhava",
        "version": wm.__version__,
        "installed_from": "PyPI",
        "dataset": name,
        "dim": int(dim),
        "N": int(N),
        "NQ": NQ,
        "K": K,
        "metric": "cosine",
        "gt": "the motor's own search_exact on the same query (valid ceiling)",
        "note": "raw data passed to the motor; no numpy normalization or re-ordering",
        "results": results,
    }


# ---- 2. Load each public dataset (raw) and benchmark ----
out_all = {}

# GloVe (d=100)
glove_txt = find_file("glove.6B.100d.txt")
if glove_txt:
    rows = []
    with open(glove_txt, "r") as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            parts = line.strip().split()
            if len(parts) >= 101:
                rows.append([float(x) for x in parts[1:101]])
    X = np.asarray(rows, dtype=np.float32)
    print(f"[2] GloVe loaded: {X.shape}", flush=True)
    out_all["glove_d100"] = run_dataset("rtatman/glove", X, 100)
else:
    print("[2] GloVe dataset not found", flush=True)

# BIGANN (d=128, uint8)
bigann = find_file("base.u8bin")
if bigann:
    base = np.memmap(bigann, dtype=np.uint8, mode="r", shape=(100_000_000, 128))
    X = np.ascontiguousarray(base[:20000])
    print(f"[2] BIGANN loaded: {X.shape}", flush=True)
    out_all["bigann_d128"] = run_dataset("shurangwu/bigann-100m", X, 128)
else:
    print("[2] BIGANN dataset not found", flush=True)

# arXiv OpenAI (d=1536)
vectors = find_file("vectors.dat")
if vectors:
    sz = os.path.getsize(vectors)
    exact = sz / (1536 * 4)
    if exact.is_integer():
        n_use = min(20000, sz // (1536 * 4))
        X = np.ascontiguousarray(np.memmap(vectors, dtype=np.float32, mode="r",
                                           offset=0, shape=(n_use, 1536)))
        print(f"[2] arXiv loaded: {X.shape}", flush=True)
        out_all["arxiv_d1536"] = run_dataset("tomtum/openai-arxiv-embeddings", X, 1536)
    else:
        print(f"[2] arXiv vectors.dat not raw float32 n x 1536 (size {sz})", flush=True)
else:
    print("[2] arXiv dataset not found", flush=True)

# ---- 3. Emit one transparent JSON per dataset ----
os.makedirs("/kaggle/working/results", exist_ok=True)
for name, data in out_all.items():
    path = f"/kaggle/working/results/{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {path}", flush=True)

with open("/kaggle/working/results/summary.json", "w") as f:
    json.dump({"package": "winnex-madhava", "version": wm.__version__,
               "datasets": {k: v["results"] for k, v in out_all.items()}}, f, indent=2)

print("\n" + "=" * 70)
print("ALL RESULTS")
print(json.dumps(out_all, indent=2))
print("=" * 70)
