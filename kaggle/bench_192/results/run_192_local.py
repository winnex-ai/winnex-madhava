#!/usr/bin/env python3
"""
winnex-madhava 1.9.2 benchmark — LOCAL EXECUTION for auditability.
Reproduces kaggle/bench_192/winnex-madhava-1-9-2-honest.py outside Kaggle,
using the same HONEST PROTOCOL and the same run_dataset() / measure() /
validate_commitment() logic. Inputs are raw public datasets:
  - GloVe  (rtatman/glove)  d=100   from glove.6B.100d.txt
  - BIGANN (shurangwu/bigann-100m) d=128 (uint8) from base.u8bin
  - arXiv  (tomtum/openai-arxiv-embeddings) d=1536 from raw float32 n x 1536
      (extracted from ml-arxiv-embeddings.json — same vectors the Kaggle
       dataset provides as vectors.dat)

Package: winnex-madhava==1.9.2 installed from PyPI (verified version).
"""
import numpy as np
import sys, time, os, json

import winnex_madhava as wm
assert wm.__version__ == "1.9.2", "expected 1.9.2, got " + wm.__version__
assert hasattr(wm.MadhavaL2, "search_with_commitment"), \
    "1.9.2 must expose search_with_commitment"


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
    """Validate the AuditCommitment (NEW in 1.9.2)."""
    n_q = len(Qn)
    n_sample_bounded = 0
    n_exact_match = 0
    n_deterministic = 0
    n_genuine = 0
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


def run_dataset(name, X, dim, out_dir):
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
    out = {
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
    path = os.path.join(out_dir, name.replace("/", "_") + ".json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"    saved {path}", flush=True)
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/results_192"
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 70, flush=True)
    print(f"winnex-madhava 1.9.2 — LOCAL RUN (winnex_madhava {wm.__version__})")
    print("=" * 70, flush=True)

    out_all = {}

    # GloVe d=100
    glove = sys.argv[2] if len(sys.argv) > 2 else "/tmp/glove.6B.100d.txt"
    if os.path.exists(glove):
        rows = []
        with open(glove, "r") as f:
            for i, line in enumerate(f):
                if i >= 20000:
                    break
                parts = line.strip().split()
                if len(parts) >= 101:
                    rows.append([float(x) for x in parts[1:101]])
        X = np.asarray(rows, dtype=np.float32)
        print(f"[2] GloVe loaded: {X.shape}", flush=True)
        out_all["glove_d100"] = run_dataset("rtatman/glove", X, 100, out_dir)
    else:
        print(f"[2] GloVe not found at {glove}", flush=True)

    # BIGANN d=128 uint8
    bigann = sys.argv[3] if len(sys.argv) > 3 else "/home/wnnx_user/zenodo/bigann_data/base.u8bin"
    if os.path.exists(bigann):
        base = np.memmap(bigann, dtype=np.uint8, mode="r", shape=(100_000_000, 128))
        X = np.ascontiguousarray(base[:20000])
        print(f"[2] BIGANN loaded: {X.shape}", flush=True)
        out_all["bigann_d128"] = run_dataset("shurangwu/bigann-100m", X, 128, out_dir)
    else:
        print(f"[2] BIGANN not found at {bigann}", flush=True)

    # arXiv d=1536 (raw float32 n x 1536, extracted to vectors.dat)
    arxiv = sys.argv[4] if len(sys.argv) > 4 else "/tmp/arxiv_vectors_20k.dat"
    if os.path.exists(arxiv):
        n = os.path.getsize(arxiv) // (1536 * 4)
        X = np.ascontiguousarray(np.memmap(arxiv, dtype=np.float32, mode="r",
                                           shape=(n, 1536)))
        print(f"[2] arXiv loaded: {X.shape}", flush=True)
        out_all["arxiv_d1536"] = run_dataset("tomtum/openai-arxiv-embeddings", X, 1536, out_dir)
    else:
        print(f"[2] arXiv not found at {arxiv}", flush=True)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"package": "winnex-madhava", "version": wm.__version__,
                   "datasets": {k: v["results"] for k, v in out_all.items()}}, f, indent=2)
    print("\n" + "=" * 70)
    print("ALL RESULTS")
    print(json.dumps(out_all, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
