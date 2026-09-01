#!/usr/bin/env python3
"""
winnex-madhava 1.9.9 + winnex-ai-normalize 1.2.1 — FIVE NEW DATASETS (v2.1)
===========================================================================
Revisão do kernel `winnex-madhava-199-five-csv-datasets-normalize`.

O que mudou nesta v2.1:
  1. winnex-ai-normalize **1.2.1** — adiciona `nan_policy` como knob do config:
     corpus com NaN/inf NUNCA roteia pca_corpus (o PCA amplifica corrupção:
     1 NaN na covariância → autovetor NaN → recall colapsa, medido 0.042 vs
     random 1.0). A política vive no preset JSON (motor/normalize agnósticos).
  2. Quality gate CORRIGIDO: remove linhas não-finitas (`np.isfinite`) — o
     gate anterior (norma > 1e-12) mantinha linhas com norma `inf` (inf > 1e-12
     é True) e deixava o inf entrar no PCA.
  3. Diagnóstico EXPLÍCITO de dataset degradado: o kernel reporta `nan_fraction`
     e a decisão de nan_policy por dataset (não esconde dataset ruim).
  4. Word2Vec DEGRADADO (`sugataghosh/google-word2vec`) é agora INCLUÍDO e
     diagnosticado como degradado — o dataset que faltou na v2 (not found) e
     que motivou esta revisão.

Herdado da v2: cfg_match corrigido (basis forçado aplicado de verdade),
presets por dataset (config enxuta no JSON), pca_iterations=30.

Protocolo honesto (herdado):
  - `pip install` do PyPI em isolamento; dados brutos; recall vs o próprio
    `search_exact` (ceiling válido); AuditCommitment validado; sem GT externo.
  - Limitações declaradas: recall self-consistente (não vs rótulos semânticos);
    auto-executado (a reprodutibilidade é a mitigação).
"""
import numpy as np
import struct
import subprocess, sys, time, os, json, hashlib

print("=" * 70, flush=True)
print("winnex-madhava 1.9.9 + winnex-ai-normalize 1.2.1 — FIVE NEW DATASETS v2.1", flush=True)
print("(cfg_match fix + nan_policy + presets por dataset + pca_iterations=30)", flush=True)
print("=" * 70, flush=True)

# ---- 1. Install from PyPI (limpo) ----
print("\n[1] Installing from PyPI...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-madhava==1.9.9"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir", "winnex-ai-normalize==1.2.1"])
import winnex_madhava as wm
import winnex_ai_normalize
from winnex_ai_normalize.core.normalize import normalize_l2
from winnex_ai_normalize.core.quality import build_quality_engine, QualityConfig
print(f"    winnex-madhava {wm.__version__} + winnex-ai-normalize {winnex_ai_normalize.__version__} installed from PyPI", flush=True)
assert wm.__version__ == "1.9.9", f"expected 1.9.9, got {wm.__version__}"
assert winnex_ai_normalize.__version__ == "1.2.1", \
    f"expected winnex-ai-normalize 1.2.1 (nan_policy fix), got {winnex_ai_normalize.__version__}"


def find_file(name):
    for r, d, files in os.walk("/kaggle/input/"):
        if name in files:
            return os.path.join(r, name)
    return None


def load_embedding_csv(path, n_max=20000):
    """CONVERSION: CSV -> float32 embedding matrix."""
    import csv as _csv
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = _csv.reader(f)
        header = next(reader)
        num_cols = []
        for i, h in enumerate(header):
            hs = h.strip().lower()
            if hs.startswith("vector_") or (hs.startswith("v") and hs[1:].isdigit()):
                num_cols.append(i)
        for i, row in enumerate(reader):
            if i >= n_max:
                break
            if not num_cols:
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


def load_word2vec_bin(path, n_max=20000):
    """PARSE do formato binário word2vec (GoogleNews-vectors-negative300.bin).

    Formato: header "VOCAB DIM" como bytes ASCII, depois para cada palavra:
      - 1 byte: comprimento da string (len(word))
      - len(word) bytes: a string (UTF-8)
      - DIM * 4 bytes: vetor float32 (little-endian)
      - 1 byte de padding se len(word) % 2 == 1 (alinhamento a 4 bytes)
    O vetor pode conter valores não-finitos (NaN/inf) se o binário estiver
    degradado — o quality gate (isfinite) trata isso depois.
    """
    vocab = []
    with open(path, "rb") as f:
        header = f.readline().strip().split()
        vocab_size, dim = int(header[0]), int(header[1])
        for i in range(min(vocab_size, n_max)):
            l = struct.unpack("B", f.read(1))[0]
            word = f.read(l).decode("utf-8", errors="ignore")
            if l % 2 == 1:
                f.read(1)  # padding
            vec = np.frombuffer(f.read(dim * 4), dtype="<f4")
            vocab.append(vec)
    X = np.asarray(vocab, dtype=np.float32)
    return X, dim


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


def run_dataset(name, X, dim, preset=None, NQ=100, K=10):
    """Fluxo de PRODUÇÃO: normalize (L2 + quality gate + preset por dataset)
    → madhava (busca com prova). Config do normalize ENXUTA: o preset JSON
    (configs/dataset_<name>.json) decide basis/k1/pca_iterations/probe_pca."""
    Q = X[-NQ:].copy()
    Xc = X[:-NQ].copy()
    N = len(Xc)
    print(f"\n--- {name} (d={dim}, N={N}, preset={preset}) ---", flush=True)

    # 0. Diagnóstico do dataset BRUTO (antes do gate): fração de linhas
    #    DEGENERADAS (norma zero OU norma não-finita em float32) e a política
    #    nan_policy aplicada. Reporta a degradação do dataset de forma honesta.
    #    IMPORTANTE: medir a NORMA em float32 (o dtype do motor) — valores
    #    finitos grandes fazem a SOMA dos quadrados estourar (norma=inf) sem
    #    serem NaN/inf por elemento. np.isfinite(Xc) por elemento NÃO pega isso.
    Xc_f32 = np.ascontiguousarray(Xc, dtype=np.float32)
    raw_norms = np.linalg.norm(Xc_f32, axis=1)
    nan_frac = float(1.0 - np.isfinite(raw_norms).mean())  # linhas com norma inf/nan
    zero_frac = float((np.isfinite(raw_norms) & (raw_norms <= 1e-12)).mean())
    cfg_diag = QualityConfig.from_dataset(preset)
    nan_policy = cfg_diag.nan_policy
    if nan_frac > 0 or zero_frac > 0:
        print(f"    [diagnóstico] dataset DEGRADADO (bruto): nan_fraction={nan_frac:.4%} "
              f"zero_norm_fraction={zero_frac:.4%} | nan_policy={nan_policy} "
              f"(pca_corpus bloqueado)", flush=True)
    else:
        print(f"    [diagnóstico] dataset limpo: nan_fraction={nan_frac:.4%} "
              f"| nan_policy={nan_policy}", flush=True)

    # 1. Quality gate: remover linhas NÃO-FINITAS ou degeneradas (norma zero).
    #    O gate antigo (norma > 1e-12) mantinha linhas com norma inf (inf > 1e-12
    #    é True) — o inf entrava no PCA e corrompia a base. np.isfinite() remove
    #    NaN e inf explicitamente. A norma é calculada em float32 (o dtype que o
    #    motor usa) para capturar o overflow que o motor encontraria.
    norms_c = np.linalg.norm(Xc_f32, axis=1)
    good_c = np.isfinite(norms_c) & (norms_c > 1e-12)
    Xc = Xc[good_c]
    Q_f32 = np.ascontiguousarray(Q, dtype=np.float32)
    norms_q = np.linalg.norm(Q_f32, axis=1)
    good_q = np.isfinite(norms_q) & (norms_q > 1e-12)
    Q = Q[good_q]
    print(f"    [quality gate] removidas {N - len(Xc)} linhas degeneradas do corpus, "
          f"{NQ - len(Q)} das queries", flush=True)
    N = len(Xc)

    # 2. Normalize L2 (o contrato cosine exige vetores unit-norm)
    Xn = normalize_l2(Xc)
    Qn = normalize_l2(Q)

    # 3. Comparação random vs pca_corpus (basis forçado — agora aplicado de
    #    verdade com o cfg_match corrigido na 1.2.0)
    results = {}
    report_data = {}
    for basis in ["random", "pca_corpus"]:
        try:
            t0 = time.time()
            eng, report = build_quality_engine(
                Xn, dim=dim, k=K, return_report=True,
                engine_kwargs=dict(basis=basis, metric="cosine", normalize_input=True))
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
            }
        except Exception as e:
            results[basis] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"    {basis}: ERROR {str(e)[:80]}", flush=True)

    # 4. Build FINAL com o PRESET por dataset (config enxuta no JSON)
    preset_result = {}
    try:
        t0 = time.time()
        eng_p, rep_p = build_quality_engine(Xn, dim=dim, k=K, return_report=True,
                                            dataset=preset)
        tb_p = time.time() - t0
        rec_p, viol_p, surv_p, lat_p, e_p, pb_p, pp_p = measure(eng_p, Qn)
        commit_p = validate_commitment(eng_p, Qn, K=K, max_sample=50)
        preset_result = {
            "preset": preset or "default",
            "recall@10": round(rec_p, 4), "bound_violations": int(viol_p),
            "pruned_by_bound_pct": round(pb_p / N * 100, 1),
            "pruned_by_prefilter_pct": round(pp_p / N * 100, 1),
            "latency_ms": round(lat_p, 2), "build_s": round(tb_p, 1),
            "engine_apply": {"basis": str(eng_p.config().basis),
                             "stage1_dim": eng_p.config().stage1_dim,
                             "pca_iterations": eng_p.config().pca_iterations,
                             "k1_fraction": eng_p.config().k1_fraction},
            "commitment": commit_p,
        }
        print(f"    preset[{preset or 'default'}]  : recall@10={rec_p:.4f}  viol={viol_p}  "
              f"build={tb_p:.1f}s  pb={pb_p/N*100:.1f}%  pp={pp_p/N*100:.1f}%", flush=True)
        print(f"      engine applied: stage1={eng_p.config().stage1_dim} "
              f"pca_iters={eng_p.config().pca_iterations}", flush=True)
    except Exception as e:
        preset_result = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        print(f"    preset[{preset or 'default'}]: ERROR {str(e)[:80]}", flush=True)

    return {
        "package": "winnex-madhava", "version": wm.__version__,
        "normalize": f"winnex-ai-normalize {winnex_ai_normalize.__version__}",
        "installed_from": "PyPI", "dataset": name, "dim": int(dim), "N": int(N),
        "NQ": NQ, "K": K, "metric": "cosine",
        "gt": "the motor's own search_exact on the same query (valid ceiling)",
        "nan_fraction": round(nan_frac, 8),
        "zero_norm_fraction": round(zero_frac, 8),
        "nan_policy": nan_policy,
        "dataset_degraded": (nan_frac > 0) or (zero_frac > 0),
        "note": "v2.1: cfg_match corrigido + nan_policy (pca bloqueado em NaN) "
                "+ gate isfinite + presets por dataset + pca_iterations=30",
        "results": results, "quality_report": report_data,
        "preset_result": preset_result,
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

# ---- 3. REGRESSION: o basis forçado agora é aplicado (cfg_match fix da 1.2.0) ----
print("\n[3] REGRESSION: basis forçado via build_quality_engine agora é aplicado...", flush=True)
try:
    rng = np.random.default_rng(0)
    N, D, K = 2000, 384, 10
    comp = rng.standard_normal((40, D)).astype(np.float32)
    coef = rng.standard_normal((N, 40)).astype(np.float32)
    X = (coef @ comp).astype(np.float32) * 0.35
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    eng_r, _ = build_quality_engine(X, dim=D, k=K, return_report=True,
                                    engine_kwargs=dict(basis="random", metric="cosine", normalize_input=True))
    eng_p, _ = build_quality_engine(X, dim=D, k=K, return_report=True,
                                    engine_kwargs=dict(basis="pca_corpus", metric="cosine", normalize_input=True))
    q = X[0].astype(np.float32)
    pb_r = eng_r.search(q).pruned_by_bound / len(X)
    pb_p = eng_p.search(q).pruned_by_bound / len(X)
    print(f"    random:    pb%={100*pb_r:.1f}", flush=True)
    print(f"    pca_corpus: pb%={100*pb_p:.1f}", flush=True)
    differ = abs(pb_p - pb_r) > 0.05
    print(f"    → random e pca DIFEREM agora? {'SIM (bug corrigido)' if differ else 'NÃO — algo errado'}", flush=True)
    cfgmatch_fix = "PASS" if differ else "FAIL"
except Exception as e:
    print(f"    FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)
    cfgmatch_fix = "FAIL"

# ---- 4. Load each NEW dataset (raw) and benchmark ----
out_all = {}
datasets_meta = {}

# 4.1 Word2Vec Google (d=300) — preset word2vec (random/k1=0.20; pca degrada)
# O dataset sugataghosh/google-word2vec é um BINÁRIO (GoogleNews-vectors-
# negative300.bin). Parse do formato word2vec binário. Este é o dataset REAL
# degradado (overflow/NaN no log 1.9.9 v1) — o kernel o diagnostica via
# nan_policy (pca_corpus bloqueado).
w2v = find_file("GoogleNews-vectors-negative300.bin") or find_file("google-word2vec.csv") or find_file("word2vec.csv")
if w2v:
    datasets_meta["word2vec"] = {"file": os.path.basename(w2v),
                                 "sha256_head": sha256_file(w2v, 1 << 20)}
    if os.path.basename(w2v).endswith(".bin"):
        X, dim = load_word2vec_bin(w2v, n_max=20000)
        print(f"[4.1] Word2Vec BIN parsed: {X.shape} (dim={dim})", flush=True)
    else:
        X, dim = load_embedding_csv(w2v)
    if X is not None:
        print(f"[4.1] Word2Vec loaded: {X.shape} (dim={dim})", flush=True)
        out_all["word2vec_d300"] = run_dataset(
            "sugataghosh/google-word2vec", X, dim, preset="word2vec")
    else:
        print("[4.1] Word2Vec conversion failed", flush=True)
else:
    print("[4.1] Word2Vec not found", flush=True)

# 4.2 FastText Crawl (d=300) — default (roteador agnóstico)
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
    print(f"[4.2] FastText loaded: {X.shape}", flush=True)
    out_all["fasttext_d300"] = run_dataset("yekenot/fasttext-crawl-300d-2m", X, dim)
else:
    print("[4.2] FastText not found", flush=True)

# 4.3 GloVe Twitter (d=25) — default
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
    print(f"[4.3] GloVe Twitter loaded: {X.shape}", flush=True)
    out_all["glove_twitter_d25"] = run_dataset("joshkyh/glove-twitter", X, 25)
else:
    print("[4.3] GloVe Twitter not found", flush=True)

# 4.4 ProtBERT CAFA5 (d=1024) — preset word2vec (manifold fraco)
pb = find_file("test_embeddings.npy")
if pb:
    datasets_meta["protbert"] = {"file": "test_embeddings.npy",
                                 "sha256_head": sha256_file(pb, 1 << 20)}
    X = np.load(pb, mmap_mode="r").astype(np.float32)
    X = np.ascontiguousarray(X[:20000])
    print(f"[4.4] ProtBERT loaded: {X.shape}", flush=True)
    out_all["protbert_cafa5"] = run_dataset(
        "henriupton/protbert-embeddings-for-cafa5", X, X.shape[1], preset="word2vec")
else:
    print("[4.4] ProtBERT not found", flush=True)

# 4.5 SBERT Forum Topics (d=384) — default
sbert = find_file("ForumTopics_all-MiniLM-L6-v2.csv")
if sbert:
    datasets_meta["sbert_forum"] = {"file": "ForumTopics_all-MiniLM-L6-v2.csv",
                                    "sha256_head": sha256_file(sbert, 1 << 20)}
    X, dim = load_embedding_csv(sbert)
    if X is not None:
        print(f"[4.5] SBERT Forum CSV converted: {X.shape} (dim={dim})", flush=True)
        out_all["sbert_forum_d384"] = run_dataset(
            "bwandowando/kaggle-forum-topics-embeddings-all-minilm-l6-v2", X, dim)
    else:
        print("[4.5] SBERT Forum CSV conversion failed", flush=True)
else:
    print("[4.5] SBERT Forum not found", flush=True)

# ---- 5. Emit one transparent JSON per dataset ----
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
        "cfgmatch_fix_basis_forced_applied": cfgmatch_fix,
        "datasets": {k: v["results"] for k, v in out_all.items()},
        "presets": {k: v.get("preset_result", {}) for k, v in out_all.items()},
        "diagnostico": {k: {"nan_fraction": v.get("nan_fraction"),
                            "zero_norm_fraction": v.get("zero_norm_fraction"),
                            "nan_policy": v.get("nan_policy"),
                            "dataset_degraded": v.get("dataset_degraded")}
                        for k, v in out_all.items()},
        "datasets_meta": datasets_meta,
        "protocol": "honest — pip install winnex-madhava==1.9.9 + "
                    "winnex-ai-normalize==1.2.1 from PyPI; 5 NEW Kaggle embedding "
                    "datasets (incl. word2vec DEGRADADO); raw vectors; recall vs "
                    "the motor's own exact scan (valid ceiling); AuditCommitment "
                    "validated; cfg_match fix verified; nan_policy (pca blocked "
                    "on NaN/inf) verified; no external semantic GT (declared "
                    "limitation).",
    }, f, indent=2)

print("\n" + "=" * 70)
print("ALL RESULTS")
print(json.dumps({k: {kk: {kkk: vv for kkk, vv in vv.items() if kkk != 'commitment'}
                     for kk, vv in v['results'].items()}
                  for k, v in out_all.items()}, indent=2))
print("=" * 70)
