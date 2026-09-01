#!/usr/bin/env python3
"""
winnex-madhava 1.9.9 benchmark — five NEW datasets (CSV conversion + L2).
Installs the packages from PyPI and tests the madhava C++ motor on public
Kaggle embedding datasets NOT used in the previous kernels.

Datasets (all raw embeddings, no model):
  1. SBERT Forum Topics    (.csv, 384d) — conversion: CSV VECTOR_* -> float32 -> L2
  2. Twitter Celebrity     (.csv, 384d) — conversion: CSV v* -> float32 -> L2
  3. FastText Crawl 300d   (.vec, 300d) — conversion: text -> float32 -> L2
  4. GloVe Twitter 25d     (.txt, 25d)  — conversion: text -> float32 -> L2
  5. ProtBERT CAFA5        (.npy, 1024d) — conversion: numpy -> float32 -> L2

HONEST PROTOCOL:
  - `pip install winnex-madhava==1.9.9` + `winnex-ai-normalize` from PyPI.
  - CSV datasets are CONVERTED (string -> float32) and then L2-NORMALIZED
    (the cosine contract) — this is the production flow, not a hidden step.
  - Reads public Kaggle embedding datasets RAW (ready vectors, no model):
      1. Google Word2Vec   (sugataghosh/google-word2vec)              d=300
      2. FastText Crawl    (yekenot/fasttext-crawl-300d-2m)           d=300
      3. GloVe Twitter 25d (joshkyh/glove-twitter)                    d=25
      4. ProtBERT CAFA5    (henriupton/protbert-embeddings-for-cafa5) d=1024
      5. SBERT Forum       (bwandowando/...all-minilm-l6-v2)          d=384
  - Passes RAW vectors to the motor (build_engine). The motor normalizes and
    computes e(v) = sqrt(1 - ||Pv||^2) in its own manifold.
  - Measures ONLY what the motor returns: recall@10 vs the motor's OWN
    search_exact (same query, same embedding) — the valid ceiling; pruning
    breakdown (pruned_by_bound vs pruned_by_prefilter); bound violations;
    build time; latency; residual.
  - Validates the AuditCommitment (search_with_commitment): max_sample honored,
    exact count match vs search_audited, determinism, genuine exclusions.
  - NEW in 1.9.9: explicit test of build_engine(float32, pca_corpus) with the
    DEFAULT quant — the path that SEGFAULTED in 1.9.8 (fixed in 1.9.9).
  - No numpy ground-truth, no re-ordering, no pipeline assembly. The engine is
    AGNOSTIC: it consumes float32/uint8 + config.

LIMITATIONS (declared, not hidden):
  - Recall is measured vs the motor's OWN exact scan (valid ceiling), NOT vs
    external semantic labels. Semantic recall vs external GT is a separate
    step (Fase 2) with the evaluator's dataset.
  - Self-executed; reproducibility is the mitigation (any third party can
    re-run this kernel on Kaggle and get the same numbers).
"""
import numpy as np
import subprocess, sys, time, os, json, hashlib

print("=" * 70, flush=True)
print("winnex-madhava 1.9.9 — FIVE NEW DATASETS (PyPI install, honest protocol)", flush=True)
print("=" * 70, flush=True)

# ---- 1. Install from PyPI (motor + normalize — o fluxo de produção) ----
print("\n[1] Installing winnex-madhava==1.9.9 + winnex-ai-normalize from PyPI...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-madhava==1.9.9"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-ai-normalize"])
import winnex_madhava as wm
from winnex_ai_normalize.core.normalize import normalize_l2
from winnex_ai_normalize.core.quality import build_quality_engine
print(f"    winnex-madhava {wm.__version__} + winnex-ai-normalize installed from PyPI", flush=True)
assert wm.__version__ == "1.9.9", "expected 1.9.9, got " + wm.__version__
assert hasattr(wm.MadhavaL2, "search_with_commitment"), "1.9.9 must expose search_with_commitment"

# Wheel hash (integrity)
try:
    dist = os.path.join(os.path.dirname(winnex_madhava.__file__), "..", "winnex_madhava-1.9.9.dist-info")
    if os.path.isdir(dist):
        for f in os.listdir(dist):
            if f.endswith(".whl") or f == "METADATA":
                pass
    # record via pip
    import importlib.metadata
    meta = importlib.metadata.metadata("winnex-madhava")
    wheel_hash = "n/a"
except Exception:
    wheel_hash = "n/a"
print(f"    wheel metadata: {wheel_hash}", flush=True)


def find_file(name):
    for r, d, files in os.walk("/kaggle/input/"):
        if name in files:
            return os.path.join(r, name)
    return None


def load_embedding_csv(path, n_max=20000):
    """CONVERSION: CSV -> float32 embedding matrix.

    Reads a CSV whose numeric columns are the embedding vector (either
    VECTOR_0..VECTOR_N-1 or v1..vN, plus a leading non-numeric column such as
    id/username). Returns (X float32, n_cols_detected).
    """
    import csv as _csv
    rows = []
    dim = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = _csv.reader(f)
        header = next(reader)
        # Detectar quais colunas são numéricas (o embedding)
        num_cols = []
        for i, h in enumerate(header):
            hs = h.strip().lower()
            if hs.startswith("vector_") or (hs.startswith("v") and hs[1:].isdigit()):
                num_cols.append(i)
        for i, row in enumerate(reader):
            if i >= n_max:
                break
            if not num_cols:
                # fallback: tentar converter todas as colunas exceto a primeira não-numérica
                vals = []
                for cell in row[1:]:
                    try:
                        vals.append(float(cell))
                    except (ValueError, IndexError):
                        break
                if len(vals) >= 10:
                    rows.append(vals)
            else:
                try:
                    vals = [float(row[j]) for j in num_cols]
                    rows.append(vals)
                except (ValueError, IndexError):
                    continue
    if not rows:
        return None, None
    X = np.asarray(rows, dtype=np.float32)
    return X, X.shape[1]


def sha256_file(path, max_bytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if max_bytes:
            h.update(f.read(max_bytes))
        else:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def measure(eng, Qn, K=10):
    """Recall of the motor's search() vs its OWN search_exact(), same query."""
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
    """Validate the AuditCommitment (from 1.9.2)."""
    n_q = len(Qn)
    n_sample_bounded = 0; n_exact_match = 0; n_deterministic = 0; n_genuine = 0
    total_excluded = 0; max_payload = 0; sample_rec = None
    for j in range(n_q):
        c = eng.search_with_commitment(Qn[j], k=K, max_sample=max_sample)
        c2 = eng.search_with_commitment(Qn[j], k=K, max_sample=max_sample)
        ar = eng.search_audited(Qn[j], k=K, max_audit_records=max_sample)
        n_s = len(c["sampled_records"])
        total_excluded += c["total_excluded_count"]
        payload = len(json.dumps(c).encode())
        max_payload = max(max_payload, payload)
        if n_s <= max_sample: n_sample_bounded += 1
        if c["total_excluded_count"] == ar["audit_excluded"]: n_exact_match += 1
        if [s["doc_id"] for s in c["sampled_records"]] == \
           [s["doc_id"] for s in c2["sampled_records"]]: n_deterministic += 1
        thr = c["global_threshold"]
        if all(s["upper_bound"] < thr for s in c["sampled_records"]): n_genuine += 1
        if sample_rec is None and c["sampled_records"]: sample_rec = c["sampled_records"][0]
    return {
        "queries": n_q, "sample_bounded": n_sample_bounded,
        "exact_count_match": n_exact_match, "deterministic": n_deterministic,
        "genuine_exclusions": n_genuine, "total_excluded": total_excluded,
        "max_commitment_payload_bytes": max_payload, "sample_record": sample_rec,
    }


def run_dataset(name, X, dim, NQ=100, K=10):
    """Fluxo de PRODUÇÃO: winnex-ai-normalize (L2 + qualidade + roteamento)
    → winnex-madhava (busca com prova)."""
    Q = X[-NQ:].copy()
    Xc = X[:-NQ].copy()
    N = len(Xc)
    print(f"\n--- {name} (d={dim}, N={N}) ---", flush=True)

    # 1. Quality gate: remover linhas degeneradas (norma zero) — o mesmo critério
    #    que o QualityValidator sinalizaria como dataset.degenerate em produção.
    norms_c = np.linalg.norm(Xc.astype(np.float64), axis=1)
    good_c = norms_c > 1e-12
    Xc = Xc[good_c]
    norms_q = np.linalg.norm(Q.astype(np.float64), axis=1)
    good_q = norms_q > 1e-12
    Q = Q[good_q]
    n_removed_c = N - len(Xc)
    n_removed_q = NQ - len(Q)
    print(f"    [quality gate] removidas {n_removed_c} linhas degeneradas do corpus, "
          f"{n_removed_q} das queries", flush=True)
    N = len(Xc)

    # 2. Normalize L2 (o contrato cosine exige vetores unit-norm)
    Xn = normalize_l2(Xc)
    Qn = normalize_l2(Q)

    # 2. Quality gate + roteamento de config (build_quality_engine)
    report_data = {}
    results = {}
    for basis in ["random", "pca_corpus"]:
        try:
            t0 = time.time()
            # Força a base desejada, mantém o roteamento de k1/early_exit do normalize
            eng, report = build_quality_engine(
                Xn, dim=dim, k=K, return_report=True,
                engine_kwargs=dict(basis=basis, metric="cosine",
                                   normalize_input=True))
            tb = time.time() - t0
            rec, viol, surv, lat, e, pb, pp = measure(eng, Qn)
            commit = validate_commitment(eng, Qn, K=K, max_sample=50)
            results[basis] = {
                "recall@10": round(rec, 4), "bound_violations": int(viol),
                "pruned_by_bound_pct": round(pb / N * 100, 1),
                "pruned_by_prefilter_pct": round(pp / N * 100, 1),
                "survivor_pct": round(surv / N * 100, 1),
                "latency_ms": round(lat, 2), "build_s": round(tb, 1),
                "residual_e_v": round(e, 4) if e else None, "commitment": commit,
                "normalize_roteada": {"basis": report.basis,
                                      "k1_fraction": report.k1_fraction,
                                      "early_exit": report.early_exit},
            }
            print(f"    {basis:14s}: recall@10={rec:.4f}  viol={viol}  "
                  f"build={tb:.1f}s  pb={pb/N*100:.1f}%  pp={pp/N*100:.1f}%", flush=True)
            print(f"      [commit] bounded={commit['sample_bounded']}/{commit['queries']}  "
                  f"count={commit['exact_count_match']}/{commit['queries']}  "
                  f"det={commit['deterministic']}/{commit['queries']}  "
                  f"genuine={commit['genuine_exclusions']}/{commit['queries']}", flush=True)
            report_data[basis] = {
                "flags": [{"code": f.code, "severity": f.severity}
                          for f in report.flags][:8],
                "metrics": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in report.metrics.items()},
            }
        except Exception as e:
            results[basis] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"    {basis}: ERROR {str(e)[:80]}", flush=True)

    return {
        "package": "winnex-madhava", "version": wm.__version__,
        "normalize": "winnex-ai-normalize (L2 + quality gate + routing)",
        "installed_from": "PyPI", "dataset": name, "dim": int(dim), "N": int(N),
        "NQ": NQ, "K": K, "metric": "cosine",
        "gt": "the motor's own search_exact on the same query (valid ceiling)",
        "note": "PRODUCTION FLOW: normalize_l2 (contrato cosine) → build_quality_engine "
                "(flags + config roteada) → winnex_madhava busca com prova",
        "results": results, "quality_report": report_data,
    }


# ---- 2. NEW: segfault fix test (float32 + pca_corpus + DEFAULT quant) ----
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

# ---- 3. Load each NEW dataset (raw) and benchmark ----
out_all = {}
datasets_meta = {}

# 3.1 SBERT Forum Topics (.csv, 384d) — CONVERSION + L2
sbert = find_file("ForumTopics_all-MiniLM-L6-v2.csv")
if sbert:
    datasets_meta["sbert_forum"] = {"file": "ForumTopics_all-MiniLM-L6-v2.csv",
                                    "sha256_head": sha256_file(sbert, 1 << 20)}
    X, dim = load_embedding_csv(sbert)
    if X is not None:
        print(f"[3.1] SBERT Forum CSV converted: {X.shape} (dim={dim})", flush=True)
        out_all["sbert_forum_d384"] = run_dataset(
            "bwandowando/kaggle-forum-topics-embeddings-all-minilm-l6-v2", X, dim)
    else:
        print("[3.1] SBERT Forum CSV conversion failed", flush=True)
else:
    print("[3.1] SBERT Forum not found", flush=True)

# 3.2 Twitter Celebrity (.csv, 384d) — CONVERSION + L2
tw = find_file("twitter-celebrity-embed-data.csv")
if tw:
    datasets_meta["twitter_celebrity"] = {"file": "twitter-celebrity-embed-data.csv",
                                          "sha256_head": sha256_file(tw, 1 << 20)}
    X, dim = load_embedding_csv(tw)
    if X is not None:
        print(f"[3.2] Twitter Celebrity CSV converted: {X.shape} (dim={dim})", flush=True)
        out_all["twitter_celebrity"] = run_dataset(
            "ahmedshahriarsakib/top-1000-twitter-celebrity-tweets-embeddings", X, dim)
    else:
        print("[3.2] Twitter Celebrity CSV conversion failed", flush=True)
else:
    print("[3.1] Word2Vec not found", flush=True)

# 3.3 FastText Crawl 300d (.vec text) — language
ft = find_file("crawl-300d-2M.vec")
if ft:
    datasets_meta["fasttext"] = {"file": "crawl-300d-2M.vec",
                                 "sha256_head": sha256_file(ft, 1 << 20)}
    rows = []
    with open(ft, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip().split()
        vocab, dim = int(first[0]), int(first[1])
        for i, line in enumerate(f):
            if i >= 20000:
                break
            parts = line.strip().split()
            if len(parts) >= dim + 1:
                rows.append([float(x) for x in parts[1:dim+1]])
    X = np.asarray(rows, dtype=np.float32)
    print(f"[3.3] FastText loaded: {X.shape}", flush=True)
    out_all["fasttext_d300"] = run_dataset("yekenot/fasttext-crawl-300d-2m", X, dim)
else:
    print("[3.3] FastText not found", flush=True)

# 3.4 GloVe Twitter 25d (.txt) — social
gt = find_file("glove.twitter.27B.25d.txt")
if gt:
    datasets_meta["glove_twitter"] = {"file": "glove.twitter.27B.25d.txt",
                                      "sha256_head": sha256_file(gt, 1 << 20)}
    rows = []
    with open(gt, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            parts = line.strip().split()
            if len(parts) >= 26:
                rows.append([float(x) for x in parts[1:26]])
    X = np.asarray(rows, dtype=np.float32)
    print(f"[3.4] GloVe Twitter loaded: {X.shape}", flush=True)
    out_all["glove_twitter_d25"] = run_dataset("joshkyh/glove-twitter", X, 25)
else:
    print("[3.4] GloVe Twitter not found", flush=True)

# 3.5 ProtBERT CAFA5 (.npy) — proteins
pb = find_file("test_embeddings.npy")
if pb:
    datasets_meta["protbert"] = {"file": "test_embeddings.npy",
                                 "sha256_head": sha256_file(pb, 1 << 20)}
    X = np.load(pb, mmap_mode="r").astype(np.float32)
    # Use a subset
    X = np.ascontiguousarray(X[:20000])
    print(f"[3.5] ProtBERT loaded: {X.shape}", flush=True)
    out_all["protbert_cafa5"] = run_dataset("henriupton/protbert-embeddings-for-cafa5", X, X.shape[1])
else:
    print("[3.5] ProtBERT not found", flush=True)

# ---- 4. Emit one transparent JSON per dataset ----
os.makedirs("/kaggle/working/results", exist_ok=True)
for name, data in out_all.items():
    path = f"/kaggle/working/results/{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {path}", flush=True)

with open("/kaggle/working/results/summary.json", "w") as f:
    json.dump({
        "package": "winnex-madhava", "version": wm.__version__,
        "segfault_fix_float32_pca_default_quant": segfault_fix,
        "datasets": {k: v["results"] for k, v in out_all.items()},
        "datasets_meta": datasets_meta,
        "protocol": "honest — pip install winnex-madhava==1.9.9 from PyPI; "
                    "5 NEW Kaggle embedding datasets; raw vectors; recall vs "
                    "the motor's own exact scan (valid ceiling); AuditCommitment "
                    "validated; no external semantic GT (declared limitation).",
    }, f, indent=2)

print("\n" + "=" * 70)
print("ALL RESULTS")
print(json.dumps({k: {kk: {kkk: vv for kkk, vv in vv.items() if kkk != 'commitment'}
                     for kk, vv in v['results'].items()}
                  for k, v in out_all.items()}, indent=2))
print("=" * 70)
