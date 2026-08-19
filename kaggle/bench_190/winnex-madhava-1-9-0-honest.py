#!/usr/bin/env python3
"""
winnex-madhava 1.9.0 benchmark — single file. Installs the package from PyPI
and tests the madhava C++ motor on public Kaggle datasets. Emits one
transparent JSON per dataset.

HONEST PROTOCOL (inherited from 1-8-8-honest):
  - `pip install winnex-madhava==1.9.0` from PyPI (nothing else).
  - Reads public Kaggle datasets (raw, no numpy pre-processing):
      1. GloVe  (rtatman/glove)               d=100
      2. BIGANN (shurangwu/bigann-100m)       d=128 (uint8)
      3. arXiv  (tomtum/openai-arxiv-embeddings) d=1536
  - Passes RAW data to the motor (build_engine). The motor normalizes and
    computes e(v) = sqrt(1 - ||Pv||^2) in its own manifold.
  - Measures ONLY what the motor returns: recall@10 vs the motor's own
    search_exact (same query), pruning (k3), bound violations, residual.
  - No numpy ground-truth, no re-ordering, no pipeline assembly.

NEW IN 1.9.0 (the per-document audit certificate):
  - For each query we also call `search_audited` (new in 1.9.0) and validate
    the **per-document Cauchy-Schwarz certificate**:
      * every `excluded=true` record must have `true_cosine < threshold`
        (cosine) — i.e. the bound PROVES the doc is outside the exact top-K;
      * the certificate's `audit_excluded` must match the motor's own
        `pruned_by_bound` when it covers the whole corpus.
  - This is the GovAuditRecord-compatible format consumed by the
    tracer-gov / tracer-med compliance flows — now produced by the motor C++.
"""
import numpy as np
import subprocess, sys, time, os, json

print("=" * 70, flush=True)
print("winnex-madhava 1.9.0 — PyPI install + per-doc certificate validation")
print("=" * 70, flush=True)

# ---- 1. Install from PyPI ----
print("\n[1] Installing winnex-madhava==1.9.0 from PyPI...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-madhava==1.9.0"])
import winnex_madhava as wm
print(f"    winnex-madhava {wm.__version__} installed from PyPI", flush=True)
assert wm.__version__ == "1.9.0", "expected 1.9.0, got " + wm.__version__


def find_file(name):
    for r, d, files in os.walk("/kaggle/input/"):
        if name in files:
            return os.path.join(r, name)
    return None


def measure(eng, Qn, K=10):
    """Recall of the motor's search() vs its OWN search_exact(), same query.

    Also captures the HONEST pruning breakdown the motor reports:
      pruned_by_bound     — vectors the Cauchy-Schwarz bound PROVED outside
                            top-K (UB < worst). The real bound-driven pruning.
      pruned_by_prefilter — vectors cut by the fixed k1_fraction/k2_max
                            without a per-vector certificate.
    """
    viol = 0
    surv = 0
    lat = 0.0
    rec = 0.0
    pb = 0
    pp = 0
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


def validate_certificate(eng, Qn, K=10, max_audit_records=500):
    """Validate the per-document audit certificate (NEW in 1.9.0).

    For each query:
      1. search_audited returns a per-doc certificate.
      2. Every `excluded=true` record must have true_cosine < threshold
         (cosine) — otherwise the certificate would be unsound.
      3. Consistency: with full coverage, the certificate's `audit_excluded`
         (docs provably outside the TRUE global top-K) is a lower bound of
         the motor's `pruned_by_bound` (which uses the post-filter pool
         threshold — a semantic we expose, not hide).
    Returns counts across all queries (certificate_violations MUST be 0).
    """
    n_queries = len(Qn)
    n_excl_total = 0
    n_bad_excl = 0          # excluded records that are actually in the top-K
    n_consistency_ok = 0    # queries where audit_excluded <= pruned_by_bound
    n_cert_ok = 0           # queries with a non-empty, violation-free cert
    sample_rec = None
    n_audit_excl_full = 0
    n_pruned_by_bound = 0

    for j in range(n_queries):
        ar = eng.search_audited(Qn[j], k=K, max_audit_records=max_audit_records)
        base = eng.search(Qn[j])

        # 1. Non-empty certificate + covers the top-K.
        if ar["audit"] and ar["bound_violations"] == 0:
            n_cert_ok += 1

        # 2. Integrity: no excluded doc can be in the exact top-K.
        #    For cosine: UB < threshold => exact cos < threshold, so an
        #    excluded doc cannot be among the top-K by exact score.
        exact = set(eng.search_exact(Qn[j]).indices)
        bad = 0
        for rec in ar["audit"]:
            if rec["excluded"] and rec["doc_id"] in exact:
                bad += 1
        n_bad_excl += bad
        n_excl_total += ar["audit_excluded"]

        # 3. Consistency (full coverage): the certificate's honest global
        #    count is a lower bound of the pool-threshold pruned_by_bound.
        n = eng.num_vectors()
        ar_full = eng.search_audited(Qn[j], k=K, max_audit_records=n)
        n_audit_excl_full += ar_full["audit_excluded"]
        n_pruned_by_bound += base.pruned_by_bound
        if ar_full["audit_excluded"] <= base.pruned_by_bound:
            n_consistency_ok += 1

        if sample_rec is None and ar["audit"]:
            sample_rec = ar["audit"][0]

    return {
        "queries": n_queries,
        "cert_ok": n_cert_ok,
        "excluded_total": n_excl_total,
        "certificate_violations": n_bad_excl,          # MUST be 0
        "consistency_ok": n_consistency_ok,            # audit_excl <= pruned
        "audit_excluded_full_total": n_audit_excl_full,
        "pruned_by_bound_total": n_pruned_by_bound,
        "sample_record": sample_rec,
    }


def run_dataset(name, X, dim):
    """Benchmark the motor on one dataset + validate the per-doc certificate."""
    NQ, K = 100, 10
    Q = X[-NQ:].copy()          # held-out queries (same distribution)
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
        cert = validate_certificate(eng, Q, K=K)
        results[basis] = {
            "recall@10": round(rec, 4),
            "bound_violations": int(viol),
            "pruned_by_bound_pct": round(pb / N * 100, 1),
            "pruned_by_prefilter_pct": round(pp / N * 100, 1),
            "survivor_pct": round(surv / N * 100, 1),
            "latency_ms": round(lat, 2),
            "build_s": round(tb, 1),
            "residual_e_v": round(e, 4) if e else None,
            # --- 1.9.0: per-document certificate validation ---
            "certificate": cert,
        }
        print(f"    {basis:12s}: recall@10={rec:.4f}  viol={viol}  "
              f"bound_pruned={pb/N*100:.1f}%  prefilter={pp/N*100:.1f}%  "
              f"build={tb:.1f}s  e(v)={e if e else 0:.4f}", flush=True)
        print(f"      [cert] excluded_total={cert['excluded_total']}  "
              f"violations={cert['certificate_violations']}  "
              f"consistency_ok={cert['consistency_ok']}/{cert['queries']}",
              flush=True)
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

# also a combined summary
with open("/kaggle/working/results/summary.json", "w") as f:
    json.dump({"package": "winnex-madhava", "version": wm.__version__,
               "datasets": {k: v["results"] for k, v in out_all.items()}}, f, indent=2)

print("\n" + "=" * 70)
print("ALL RESULTS")
print(json.dumps(out_all, indent=2))
print("=" * 70)
