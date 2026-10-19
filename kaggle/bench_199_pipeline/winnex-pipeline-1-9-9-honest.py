#!/usr/bin/env python3
"""
winnex-pipeline-1-9-9-honest.py — Winnex Pipeline Honest Benchmark v199
========================================================================
Fluxo de PRODUÇÃO Normalize → Madhava nas versões novas
(winnex-madhava==1.9.9 + winnex-ai-normalize==1.2.1), com o WITNESS e o
CERTIFICADO de auditoria expostos por dataset.

Base:
  - kleniopadilha/winnex-pipeline-honest-bench (o kernel público anterior):
    instala os pacotes do PyPI, datasets reais, GT exato, sem fallbacks.
  - winnex-madhava-199-five-datasets-normalize-v2: o fluxo de produção
    Normalize (quality gate + presets JSON por dataset) → Madhava.

O QUE MUDOU vs o kernel anterior (e o que este kernel prova):
  1. madhava 1.9.9: fix do SEGFAULT em build_engine(float32, pca_corpus)
     com o quant default (float32 força quant="none").
  2. normalize 1.2.1: quality gate corrigido (np.isfinite — remove NaN/inf),
     `nan_policy` como knob do config (pca_corpus BLOQUEADO em corpus com
     NaN/inf — o PCA amplifica corrupção: 1 NaN → recall colapsa), presets
     JSON por dataset, pca_iterations=30.
  3. O witness e o certificado: o motor CAPTURA na busca o threshold GLOBAL
     (score exato do K-ésimo resultado) e, para CADA doc, o bound
     UB(v,q)=⟨Pv,Pq⟩+e(v)e(q) no momento da decisão (audit_ids/audit_ubs/
     audit_threshold/audit_residuals). `search_audited` EXPANDE isso no
     certificado por documento (GovAuditRecord). Este kernel MOSTRA e
     VERIFICA a integridade do certificado (0 falsos excluídos).

PROTOCOLO HONESTO (herdado):
  - `pip install` do PyPI em isolamento (subprocess com PYTHONPATH limpo);
  - datasets REAIS (Kaggle) — os 3 clássicos (GloVe-100 400k, MNIST-784 60k,
    HN-OpenAI-1536 100k) + 1 dataset DEGRADADO (word2vec simulado com NaN/
    inf/zero — o caso que motivou o nan_policy);
  - ground-truth: top-K EXATO numpy (argpartition, independente do motor) nos
    3 clássicos; recall@10 vs GT externo E vs o search_exact do motor
    (ceiling auto-consistente);
  - verificação de integridade do certificado por query representativa;
  - limitações declaradas (sem GT semântico externo; o custo do GT numpy é o
    custo de auditoria; auto-executado — reprodutibilidade é a mitigação).

Saída: /kaggle/working/results/<dataset>.json + summary.json + hashes wheels.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

VERSION_MADHAVA = "1.9.9"
VERSION_NORMALIZE = "1.2.1"
VERSION_NANO = "0.1.12"
VERSION_XFACTOR = "1.0.2"

K = 10
NQ = 100
PY = sys.executable
IS_KAGGLE = os.path.isdir("/kaggle/input")

print("=" * 72, flush=True)
print("WINNEX PIPELINE — KAGGLE HONEST BENCHMARK v199")
print(f"winnex-madhava=={VERSION_MADHAVA} + winnex-ai-normalize=={VERSION_NORMALIZE}")
print("Fluxo de PRODUÇÃO Normalize → Madhava + witness/certificado")
print("=" * 72, flush=True)


# ---------------------------------------------------------------------------
# 0. Instalação isolada do PyPI (sem vazar o ambiente do notebook)
# ---------------------------------------------------------------------------
def _target_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=f"wxb_{prefix}_")


def _install(target: str, *pkgs: str) -> None:
    # Smoke test LOCAL (WINNEX_BENCH_LOCAL_SMOKE=1): em vez de baixar do PyPI,
    # aponta para os pacotes winnex já instalados no ambiente (symlinks). Só
    # para validar o fluxo de ponta a ponta sem rede; o Kaggle instala do PyPI.
    if os.environ.get("WINNEX_BENCH_LOCAL_SMOKE") == "1":
        import site
        sp = site.getsitepackages()[0]
        pkg_map = {
            "winnex_madhava": "winnex_madhava",
            "winnex_ai_normalize": "winnex_ai_normalize",
            "configs": "configs",
            "winnex_nano": "winnex_nano",
            "winnex_xfactor": "winnex_xfactor",
        }
        for name, src in pkg_map.items():
            s = os.path.join(sp, src)
            d = os.path.join(target, name)
            if os.path.exists(s) and not os.path.exists(d):
                os.symlink(s, d)
        return
    subprocess.check_call(
        [PY, "-m", "pip", "install", "-q", "--no-cache-dir",
         "--disable-pip-version-check", "--target", target, *pkgs],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _run_iso(target: str, code: str, *args: str) -> dict:
    # Smoke test LOCAL: executa o código no MESMO processo (os pacotes já são
    # os locais); o isolamento real (PYTHONPATH limpo) só no Kaggle.
    if os.environ.get("WINNEX_BENCH_LOCAL_SMOKE") == "1":
        import io as _io
        import contextlib
        import winnex_madhava as _wm
        buf = _io.StringIO()
        _old_argv = list(sys.argv)
        sys.argv = [_old_argv[0]] + list(args) if args else _old_argv
        _g = {"__name__": "__engine__", "sys": sys, "np": np, "__file__": "<engine>",
              "winnex_madhava": _wm, "wm": _wm, "K": K}
        try:
            import winnex_ai_normalize as _nrm
            from winnex_ai_normalize.core.normalize import normalize_l2
            from winnex_ai_normalize.core.quality import (
                build_quality_engine, QualityConfig)
            _g.update({"winnex_ai_normalize": _nrm, "normalize_l2": normalize_l2,
                       "build_quality_engine": build_quality_engine,
                       "QualityConfig": QualityConfig})
        except Exception:
            pass  # ENGINE_CODE não usa normalize
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<engine>", "exec"), _g)
        sys.argv = _old_argv
        last = [ln for ln in buf.getvalue().splitlines() if ln.strip()][-1]
        try:
            return json.loads(last)
        except json.JSONDecodeError:
            return {"version": last.strip()}
    env = dict(os.environ)
    env["PYTHONPATH"] = target
    proc = subprocess.run([PY, "-c", code, *args],
                          capture_output=True, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess code failed (rc={proc.returncode}):\n"
            f"--- stdout tail ---\n{proc.stdout[-1500:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}")
    # A saída pode ter linhas de log (flush=True) antes do JSON final; o
    # contrato é: a ÚLTIMA linha não-vazia é o JSON de resultado.
    last = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"version": last.strip()}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wheel_fingerprint(pkg: str) -> dict:
    """SHA256 + nome do wheel que o pip resolve (origem verificável no PyPI)."""
    if os.environ.get("WINNEX_BENCH_LOCAL_SMOKE") == "1":
        import winnex_madhava as _w
        return {"installed": f"local {pkg}", "sha256": None,
                "note": "smoke local (sem download)"}
    import glob
    d = tempfile.mkdtemp(prefix="wxfp_")
    try:
        subprocess.check_call(
            [PY, "-m", "pip", "download", "-q", "--no-deps", "--no-cache-dir",
             "-d", d, pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        files = glob.glob(os.path.join(d, "*"))
        if not files:
            return {"installed": "unknown", "sha256": None}
        f0 = files[0]
        return {"installed": os.path.basename(f0), "sha256": _sha256(f0),
                "size_bytes": os.path.getsize(f0),
                "note": "sdist (no wheel on PyPI for this platform)"}
    wheels = glob.glob(os.path.join(d, "*.whl"))
    f0 = wheels[0] if wheels else glob.glob(os.path.join(d, "*"))[0]
    return {"installed": os.path.basename(f0), "sha256": _sha256(f0),
            "size_bytes": os.path.getsize(f0)}


# ---------------------------------------------------------------------------
# 1. Datasets reais (Kaggle) / locais (smoke test) + word2vec degradado
# ---------------------------------------------------------------------------
def find_file(name: str) -> "str | None":
    if IS_KAGGLE:
        for r, d, files in os.walk("/kaggle/input/"):
            if name in files:
                return os.path.join(r, name)
    return None


def _load_or(loader, **kw):
    """Roda o loader; se o dataset não existir (ex.: smoke test local), retorna None."""
    try:
        return loader(**kw)
    except Exception:
        return None


def load_glove(n: int = 400_000) -> "np.ndarray | None":
    p = find_file("glove.6B.100d.txt")
    if not p:
        return None
    rows = np.empty((n, 100), dtype=np.float32)
    k = 0
    with open(p, "r") as f:
        for line in f:
            if k >= n:
                break
            parts = line.strip().split()
            if len(parts) >= 101:
                rows[k] = [float(x) for x in parts[1:101]]
                k += 1
    return rows[:k] if k else None


def load_mnist(n: int = 60_000) -> "np.ndarray | None":
    p = find_file("train-images.idx3-ubyte")
    if not p:
        p = find_file("train-images-idx3-ubyte")
    if not p:
        return None
    raw = np.fromfile(p, dtype=np.uint8)
    n_total = int.from_bytes(raw[4:8].tobytes(), "big")
    n_use = min(n, n_total)
    X = raw[16:16 + n_use * 784].reshape(n_use, 784).astype(np.float32)
    return np.ascontiguousarray(X)


def load_hn_openai(n: int = 100_000) -> "np.ndarray | None":
    import csv
    import io
    p = find_file("story.csv")
    if not p:
        return None
    out = np.empty((n, 1536), dtype=np.float32)
    k = 0
    if p.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(p) as z, z.open("story.csv") as fh, \
                io.TextIOWrapper(fh, encoding="utf-8") as txt:
            k = _fill_hn(list(csv.reader(txt)), out, n)
    else:
        with open(p, "r", encoding="utf-8") as fh:
            k = _fill_hn(list(csv.reader(fh)), out, n)
    return out[:k] if k else None


def _fill_hn(rows, out, n):
    import json as _json
    header = rows[0]
    emb_col = header.index("embeddings") if "embeddings" in header else 7
    k = 0
    for row in rows[1:]:
        if k >= n:
            break
        try:
            emb = _json.loads(row[emb_col])
            if len(emb) == 1536:
                out[k] = emb
                k += 1
        except (ValueError, IndexError, KeyError):
            continue
    return k


def load_arxiv_local(n: int = 20000) -> "np.ndarray | None":
    """Local smoke test: arXiv embeddings d=1536 (manifold forte)."""
    p = "/home/wnnx_user/zenodo/arxiv_100k.npy"
    if not os.path.exists(p):
        return None
    a = np.load(p, mmap_mode="r")
    return np.ascontiguousarray(a[:n])


def load_sift_local(n: int = 20000) -> "np.ndarray | None":
    """Local smoke test: SIFT-1M d=128 (caminho float32, presets sift)."""
    p = "/home/wnnx_user/zenodo/sift_data/train.npy"
    if not os.path.exists(p):
        return None
    a = np.load(p, mmap_mode="r")
    return np.ascontiguousarray(a[:n])


def build_word2vec_degraded(n: int = 20000, d: int = 300, seed: int = 0):
    """WORD2VEC DEGRADADO (simulado, determinístico) — o caso do nan_policy.

    Reproduz o que o log 1.9.9 v1 encontrou no dataset Google Word2Vec real
    (linhas de norma zero + valores estourados/NaN). O Normalize deve
    DIAGNOSTICAR (flag dataset.nan) e o nan_policy='block_pca' deve rotear
    para random/k1=0.20 (nunca pca_corpus sobre NaN — o PCA amplifica a
    corrupção: 1 NaN na covariância → autovetor NaN → recall colapsa).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 40)) @ rng.standard_normal((40, d))
    X = X.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n_zero = int(n * 0.01)
    n_nan = int(n * 0.005)
    n_inf = int(n * 0.005)
    idx = rng.choice(n, n_zero + n_nan + n_inf, replace=False)
    X[idx[:n_zero]] = 0.0                       # norma zero
    X[idx[n_zero:n_zero + n_nan]] = np.nan      # NaN (estoura float32 soma)
    X[idx[n_zero + n_nan:]] = 1e30              # inf na norma float32
    return np.ascontiguousarray(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. Motor (código executado NO ambiente isolado — nada do notebook vaza)
# ---------------------------------------------------------------------------
ENGINE_CODE = r"""
import json, os, sys, time
import numpy as np
import winnex_madhava as wm

K = 10  # top-K (self-contained: este código roda como subprocess, sem globals do kernel)

# argv: corpus.npy, queries.npy (opcional), gt.npy (opcional), nq
X = np.load(sys.argv[1])
Qpath = sys.argv[2] if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else None
gtp = sys.argv[3] if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None
NQ = int(sys.argv[4]) if len(sys.argv) > 4 else 100
Q = np.load(Qpath) if Qpath else X[-NQ:].copy()
Xc = X if Qpath else X[:-NQ].copy()
gt = np.load(gtp) if gtp else None

def exact_gt(Xc, Q, k=10):
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Qn.T
    if len(Xc) > 200_000:
        idx = np.argpartition(-S, kth=k - 1, axis=0)[:k]
        vals = np.take_along_axis(S, idx, axis=0)
        order = np.argsort(-vals, axis=0)
        return np.take_along_axis(idx, order, axis=0).T
    return np.argsort(-S, axis=0)[:k].T

def recall(pred, gt):
    return sum(1 for i in pred if i in gt) / K

gt = exact_gt(Xc, Q) if gt is None else gt
D = int(X.shape[1])
res = {}
for basis in ["random", "pca_corpus"]:
    for k1 in [0.05, 0.2]:
        t0 = time.time()
        eng = wm.build_engine(Xc, dim=D, metric="cosine", quant="none",
                              basis=basis, stage1_dim=min(192, D), stage2_dim=0,
                              k=K, k1_fraction=k1, k2_fraction=0.05, k2_max=2000,
                              normalize_input=True, early_exit=False,
                              pca_iterations=30)
        build_s = time.time() - t0
        rec = viol = pb = pp = lat = 0.0
        for j, q in enumerate(Q):
            t0 = time.time(); r = eng.search(q); lat += (time.time() - t0) * 1000
            viol += r.bound_violations
            pb += getattr(r, "pruned_by_bound", 0) or 0
            pp += getattr(r, "pruned_by_prefilter", 0) or 0
            rec += recall(r.indices, gt[j])
        N = len(Xc)
        res[f"{basis}_k1={k1}"] = {
            "recall@10_ext_gt": round(rec / NQ, 4), "latency_ms": round(lat / NQ, 3),
            "qps": round(NQ / (lat / 1000), 1), "bound_violations": int(viol),
            "pruned_by_bound_pct": round(pb / NQ / N * 100, 1),
            "pruned_by_prefilter_pct": round(pp / NQ / N * 100, 1),
            "build_s": round(build_s, 2)}
print(json.dumps({"version": wm.__version__, "results": res}))
"""


def scenario_motor(target: str, X: np.ndarray, name: str, dataset_id: str,
                   nq: int = NQ, local_gt: bool = False) -> dict:
    """Mede o motor isolado num venv limpo: recall@10 vs GT externo exato."""
    print(f"\n=== [MOTOR] {name} d={X.shape[1]} N={len(X) - nq} nq={nq} ===", flush=True)
    tmp = tempfile.mktemp(suffix=".npy")
    tq = ty = ""
    Q = X[-nq:].copy()
    Xc = X[:-nq].copy()
    Y = None
    if local_gt:  # GT numpy exato EXTERNO ao motor (VIBE-style recall@K)
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Qn.T
        Y = np.argsort(-S, axis=0)[:K].T
    # IMPORTANTE: salvar SÓ o corpus (Xc, sem as queries) — o motor mede o
    # corpus; as queries vão em arquivo separado. Se o dataset inteiro (com as
    # queries dentro) fosse o corpus, a query vazaria como documento do top-K.
    np.save(tmp, Xc)
    if len(Q):
        tq = tempfile.mktemp(suffix=".npy"); np.save(tq, Q)
    if Y is not None:
        ty = tempfile.mktemp(suffix=".npy"); np.save(ty, Y)
    try:
        data = _run_iso(target, ENGINE_CODE, tmp, tq, ty, str(nq))
    finally:
        for p in (tmp, tq, ty):
            if p:
                os.unlink(p)
    print(f"  {name}: recall@10 max=", max(v["recall@10_ext_gt"] for v in data["results"].values()), flush=True)
    for k, v in data["results"].items():
        print(f"    {k:20s} rec={v['recall@10_ext_gt']} lat={v.get('latency_ms')}ms "
              f"qps={v.get('qps')} viol={v['bound_violations']} "
              f"pb={v.get('pruned_by_bound_pct','-')}% build={v.get('build_s','-')}s", flush=True)
    return {"package": "winnex-madhava", "version": data["version"],
            "installed_from": "PyPI (isolated venv)", "dataset": dataset_id,
            "dim": int(X.shape[1]), "N": int(len(Xc)), "NQ": nq, "K": K,
            "metric": "cosine",
            "gt": "exact top-K brute-force numpy (VIBE-style recall@K) vs the motor's own search_exact",
            "results": data["results"]}


# ---------------------------------------------------------------------------
# 3. Fluxo de PRODUÇÃO Normalize → Madhava (headline) + witness/certificado
# ---------------------------------------------------------------------------
PROD_CODE = r"""
import json, os, sys, time
import numpy as np
import winnex_madhava as wm
import winnex_ai_normalize
from winnex_ai_normalize.core.normalize import normalize_l2
from winnex_ai_normalize.core.quality import build_quality_engine, QualityConfig

name = sys.argv[1]; preset = sys.argv[2]; NQ = int(sys.argv[3])
X = np.load(sys.argv[4])                     # corpus BRUTO (pré-normalize)
Q = X[-NQ:].copy(); Xc = X[:-NQ].copy()
D = int(X.shape[1]); K = 10

# --- 0. Diagnóstico do dataset BRUTO (antes do gate), em float32 ---
Xf = np.ascontiguousarray(Xc, dtype=np.float32)
raw_norms = np.linalg.norm(Xf, axis=1)
nan_frac = float(1.0 - np.isfinite(raw_norms).mean())          # linha c/ norma nan/inf
zero_frac = float((np.isfinite(raw_norms) & (raw_norms <= 1e-12)).mean())
cfg_diag = QualityConfig.from_dataset(preset)
nan_policy = cfg_diag.nan_policy
degraded = (nan_frac > 0) or (zero_frac > 0)
print(f"    [diag] {name}: nan={nan_frac:.4%} zero={zero_frac:.4%} "
      f"policy={nan_policy} degraded={degraded}", flush=True)

# --- 1. Quality gate: remove linhas não-finitas/degeneradas (norma em float32) ---
g = np.isfinite(raw_norms) & (raw_norms > 1e-12)
Xc = Xc[g]
qraw = np.linalg.norm(np.ascontiguousarray(Q, dtype=np.float32), axis=1)
qg = np.isfinite(qraw) & (qraw > 1e-12)
Q = Q[qg]
print(f"    [gate] removed {int((~g).sum())} corpus + {int((~qg).sum())} query "
      f"rows (nan/inf/zero)", flush=True)

# --- 2. Normalize L2 (contrato cosine) ---
Xn = normalize_l2(Xc); Qn = normalize_l2(Q)
N = len(Xc)

# --- 3. Build com o PRESET por dataset (a config do normalize) ---
t0 = time.time()
eng, rep = build_quality_engine(Xn, dim=D, k=K, return_report=True, dataset=preset)
build_s = time.time() - t0
cfg = eng.config()

# --- 4. Medição: recall vs search_exact do motor (ceiling) ---
rec = viol = lat = 0.0
surv = pb = pp = 0
for j in range(len(Qn)):
    t0 = time.time(); r = eng.search(Qn[j]); lat += (time.time() - t0) * 1000
    viol += r.bound_violations; surv += r.k3
    pb += getattr(r, "pruned_by_bound", 0) or 0
    pp += getattr(r, "pruned_by_prefilter", 0) or 0
    r_ex = eng.search_exact(Qn[j])
    rec += sum(1 for i in r.indices if i in r_ex.indices) / K
rec /= max(len(Qn), 1); lat /= max(len(Qn), 1); surv /= max(len(Qn), 1)
pb /= max(len(Qn), 1); pp /= max(len(Qn), 1)
resid = eng.residuals1()
e_v = float(np.mean(resid)) if len(resid) else None

# --- 5. WITNESS (o motor captura o threshold global + bounds no momento) ---
q0 = np.ascontiguousarray(Qn[0], dtype=np.float32)
r0 = eng.search(q0)
witness = {
    "top10": [int(i) for i in r0.indices],
    "global_kth_threshold": float(r0.audit_threshold),
    "pruned_by_bound": int(r0.pruned_by_bound),
    "pruned_by_prefilter": int(r0.pruned_by_prefilter),
    "audit_ids_count": int(len(r0.audit_ids)),
    "bound_proves_outside_topk_pct": round(100.0 * r0.pruned_by_bound / N, 1),
    # os docs que o bound PROVOU fora do top-K (UB < threshold):
    "excluded_sample": [
        {"doc_id": int(d), "upper_bound": round(float(u), 6),
         "residual_e": round(float(x), 6), "threshold": round(float(r0.audit_threshold), 6),
         "proved_out": bool(u < r0.audit_threshold)}
        for d, u, x in zip(r0.audit_ids[:8], r0.audit_ubs[:8], r0.audit_residuals[:8])],
}
# --- 6. CERTIFICADO (search_audited expande o witness por documento) ---
ar = eng.search_audited(q0, k=K, max_audit_records=120)
excluded_recs = [x for x in ar["audit"] if x["excluded"]]
kept_recs = [x for x in ar["audit"] if not x["excluded"]]
# integridade do certificado:
bad_ub = [x for x in excluded_recs if not (x["upper_bound"] < x["threshold"])]
bad_true = [x for x in excluded_recs if not (x["true_cosine"] < x["threshold"] - 1e-9)]
in_topk_ok = sorted(x["doc_id"] for x in kept_recs) == sorted(int(i) for i in r0.indices)
cert = {
    "query": "first normalized query (representative)",
    "audit_candidates": int(ar["audit_candidates"]),
    "audit_excluded": int(ar["audit_excluded"]),
    "cert_integrity": {
        "all_excluded_ub_lt_threshold": len(bad_ub) == 0,
        "all_excluded_true_cosine_lt_threshold": len(bad_true) == 0,
        "kept_in_topk_equal_returned": bool(in_topk_ok),
        "audit_excluded_equals_pruned_by_bound": int(ar["audit_excluded"]) == int(r0.pruned_by_bound),
    },
    "record_excluded_sample": [
        {"doc_id": int(x["doc_id"]), "true_cosine": round(float(x["true_cosine"]), 6),
         "projected_cosine": round(float(x["projected_cosine"]), 6),
         "residual_norm": round(float(x["residual_norm"]), 6),
         "upper_bound": round(float(x["upper_bound"]), 6),
         "threshold": round(float(x["threshold"]), 6),
         "excluded": bool(x["excluded"]), "stage": x["stage"]}
        for x in excluded_recs[:5]],
    "record_kept_sample": [
        {"doc_id": int(x["doc_id"]), "true_cosine": round(float(x["true_cosine"]), 6),
         "threshold": round(float(x["threshold"]), 6),
         "excluded": bool(x["excluded"]), "stage": x["stage"]}
        for x in kept_recs[:3]],
    "n_excluded_records": len(excluded_recs),
    "n_kept_records": len(kept_recs),
}
# --- 7. AuditCommitment (agregado sobre NQ queries — o registro de produção) ---
commit = {"queries": int(len(Qn)), "sample_bounded": 0, "exact_count_match": 0,
          "deterministic": 0, "genuine_exclusions": 0, "total_excluded": 0,
          "max_payload_bytes": 0, "sample_record": None}
for j in range(len(Qn)):
    c = eng.search_with_commitment(Qn[j], k=K, max_sample=50)
    c2 = eng.search_with_commitment(Qn[j], k=K, max_sample=50)
    ar2 = eng.search_audited(Qn[j], k=K, max_audit_records=50)
    n_s = len(c["sampled_records"])
    commit["sample_bounded"] += int(n_s <= 50)
    commit["exact_count_match"] += int(c["total_excluded_count"] == ar2["audit_excluded"])
    commit["deterministic"] += int([s["doc_id"] for s in c["sampled_records"]] ==
                                   [s["doc_id"] for s in c2["sampled_records"]])
    commit["genuine_exclusions"] += int(all(s["upper_bound"] < c["global_threshold"]
                                            for s in c["sampled_records"]))
    commit["total_excluded"] += int(c["total_excluded_count"])
    commit["max_payload_bytes"] = max(commit["max_payload_bytes"], len(json.dumps(c).encode()))
    if commit["sample_record"] is None and c["sampled_records"]:
        commit["sample_record"] = {"doc_id": int(c["sampled_records"][0]["doc_id"]),
                                   "upper_bound": float(c["sampled_records"][0]["upper_bound"]),
                                   "global_threshold": float(c["global_threshold"]),
                                   "proved_out": bool(c["sampled_records"][0]["upper_bound"] < c["global_threshold"])}

print(f"    [prod] {name}: recall@10={rec:.4f} viol={int(viol)} build={build_s:.1f}s "
      f"pb%={pb/N*100:.1f} pp%={pp/N*100:.1f} lat={lat:.2f}ms e(v)={e_v:.4f}", flush=True)
print(f"    [witness] top10={witness['top10']} threshold={witness['global_kth_threshold']:.4f} "
      f"proved_out={witness['pruned_by_bound']} ({witness['bound_proves_outside_topk_pct']:.1f}%)", flush=True)
print(f"    [cert] candidates={cert['audit_candidates']} excluded={cert['audit_excluded']} "
      f"integrity={cert['cert_integrity']}", flush=True)
print(f"    [commit] bounded={commit['sample_bounded']}/{commit['queries']} "
      f"count={commit['exact_count_match']}/{commit['queries']} "
      f"det={commit['deterministic']}/{commit['queries']} "
      f"genuine={commit['genuine_exclusions']}/{commit['queries']} "
      f"payload={commit['max_payload_bytes']}B", flush=True)

print(json.dumps({
    "dataset": name, "dim": D, "N": int(N), "preset": preset or "default",
    "nan_fraction": round(nan_frac, 8), "zero_norm_fraction": round(zero_frac, 8),
    "nan_policy": nan_policy, "dataset_degraded": degraded,
    "recall@10": round(rec, 4), "bound_violations": int(viol),
    "pruned_by_bound_pct": round(pb / N * 100, 1),
    "pruned_by_prefilter_pct": round(pp / N * 100, 1),
    "survivor_pct": round(surv / N * 100, 1),
    "latency_ms": round(lat, 3), "build_s": round(build_s, 1),
    "residual_e_v": round(e_v, 4) if e_v else None,
    "engine_applied": {"basis": str(cfg.basis), "stage1_dim": int(cfg.stage1_dim),
                       "pca_iterations": int(cfg.pca_iterations),
                       "k1_fraction": float(cfg.k1_fraction)},
    "quality_report": {
        "flags": [{"code": f.code, "severity": f.severity, "metric": f.metric} for f in rep.flags][:8],
        "verdict": ("fail" if rep.has_fail else "warn" if rep.warn_flags else "pass")},
    "witness": witness, "certificate": cert, "commitment": commit,
}))
"""


def scenario_prod(target: str, X: np.ndarray, name: str, preset: "str | None",
                  nq: int = NQ) -> dict:
    print(f"\n=== [PRODUÇÃO] {name} (preset={preset or 'default'}) ===", flush=True)
    tmp = tempfile.mktemp(suffix=".npy")
    np.save(tmp, X)
    try:
        return _run_iso(target, PROD_CODE, name, preset or "default", str(nq), tmp)
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# 4. Cenário pipeline (pesquisa/integração) — declarado, sem claim de produção
# ---------------------------------------------------------------------------
PIPE_CODE = r"""
import json
import numpy as np
import winnex_madhava as wm
import winnex_nano as wn
import winnex_xfactor as wxf
import winnex_ai_normalize as wnrm

# INTEGRAÇÃO DE PESQUISA (declarada): nano/xfactor são marcados como
# conceitual/refutado nos docs do workspace. Este cenário mede o ENCAMINHAMENTO
# texto->tokens->projeção->normalize->madhava como PROVA DE CONCEITO de
# integração, NÃO como claim de produção.
tok = wn.SpectralTokenizer(embed_dim=64)
texts = [
    "paciente com hipertensao arterial severa", "diabetes mellitus tipo 2",
    "suspeita de cancer de mama", "transtorno de ansiedade generalizada",
    "fraturas e ortopedia", "cardiologia pediatrica", "infeccao urinaria",
    "saude mental do idoso", "ginecologia obstetricia", "dor lombar cronica",
]
D = 384
E = np.random.default_rng(7).standard_normal((5000, D)).astype(np.float32)
P, rank, var = wxf.build_xfactor(E, dim=D, variance_tau=0.95)

def embed(t):
    ps = np.array(tok.encode_histogram(t, 64), dtype=np.float32)
    pp = np.ascontiguousarray(wxf.embed_spectral(ps, P), dtype=np.float32)
    return np.ascontiguousarray(wnrm.normalize_l2(pp[None, :])[0], dtype=np.float32)

corpus = np.stack([embed(t) for t in texts])
eng = wm.build_engine(corpus, dim=D, metric="cosine", quant="none",
                      stage1_dim=64, stage2_dim=128, k=3,
                      k1_fraction=0.5, k2_fraction=0.5, normalize_input=True,
                      early_exit=False)
qtext = "paciente hipertenso com pressao alta"
q = embed(qtext)
t0 = __import__("time").time(); r = eng.search(q); search_ms = (__import__("time").time() - t0) * 1000
print(json.dumps({
    "status": "research_integration",
    "note": "nano/xfactor are marked conceitual/refutado in the workspace docs; "
            "this is a PoC of text->tokens->projection->normalize->madhava wiring, "
            "NOT a production claim",
    "versions": {"madhava": wm.__version__, "nano": wn.__version__,
                 "xfactor": wxf.__version__,
                 "normalize": getattr(wnrm, "__version__", "n/a")},
    "top": [int(i) for i in r.indices],
    "bound_violations": int(r.bound_violations),
    "search_ms": round(search_ms, 3),
    "manifold": {"rank": int(rank), "variance_explained": float(var)},
}))
"""


def scenario_pipeline(target: str) -> dict:
    print("\n=== [PIPELINE/POC] texto → nano → xfactor → normalize → madhava ===", flush=True)
    data = _run_iso(target, PIPE_CODE)
    v = data["versions"]
    print(f"  madhava {v['madhava']} · nano {v['nano']} · xfactor {v['xfactor']} "
          f"· normalize {v['normalize']} | top={data['top']} viol={data['bound_violations']}", flush=True)
    data["installed_from"] = "PyPI (isolated venv)"
    return data


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    all_out = {"meta": {}, "scenarios": []}

    # --- 1. Motor isolado (GT externo exato) + fingerprint ---
    engine_target = _target_dir("madhava")
    _install(engine_target, f"winnex-madhava=={VERSION_MADHAVA}")
    wm_ver = _run_iso(engine_target, "import winnex_madhava as w; print(w.__version__)")
    all_out["meta"]["wheel_fingerprints"] = {}
    all_out["meta"]["wheel_fingerprints"]["winnex-madhava"] = wheel_fingerprint(
        f"winnex-madhava=={VERSION_MADHAVA}")
    print(f"\nMotor: winnex-madhava {wm_ver} "
          f"({all_out['meta']['wheel_fingerprints']['winnex-madhava']['installed']})", flush=True)

    # --- 2. Ambiente de produção (normalize+madhava) ---
    prod_target = _target_dir("prod")
    _install(prod_target, f"winnex-madhava=={VERSION_MADHAVA}",
             f"winnex-ai-normalize=={VERSION_NORMALIZE}")
    prod_vers = _run_iso(prod_target,
        "import winnex_madhava as w, winnex_ai_normalize as n, json; "
        "print(json.dumps({'madhava': w.__version__, 'normalize': n.__version__}))")
    for pkg in [f"winnex-madhava=={VERSION_MADHAVA}", f"winnex-ai-normalize=={VERSION_NORMALIZE}"]:
        all_out["meta"]["wheel_fingerprints"][pkg.split("==")[0]] = wheel_fingerprint(pkg)
    print(f"Produção: madhava {prod_vers['madhava']} · normalize {prod_vers['normalize']}", flush=True)

    # --- Datasets (Kaggle reais; smoke local usa arxiv/sift locais) ---
    # Cada dataset tem: loader (None = sintético), name, dataset_id, preset,
    # use_ext_gt (True = GT numpy externo viável em N razoável).
    datasets = []
    if IS_KAGGLE:
        datasets = [
            (load_glove, "glove", "rtatman/glove-global-vectors-for-word-representation", "default", True),
            (load_mnist, "mnist", "hojjatk/mnist-dataset", "default", True),
            (load_hn_openai, "hn_openai", "julien040/hacker-news-openai-embeddings", "default", True),
            (None, "word2vec_degraded", "simulated (Google word2vec case)", "word2vec", False),
        ]
    else:
        datasets = [
            (load_arxiv_local, "arxiv_local", "arxiv_100k.npy local (smoke)", "arxiv", True),
            (load_sift_local, "sift_local", "sift train.npy local (smoke)", "sift", True),
            (None, "word2vec_degraded", "simulated (Google word2vec case)", "word2vec", False),
        ]

    for loader, name, did, preset, use_ext_gt in datasets:
        if loader is not None:
            X = loader()
            if X is None:
                print(f"[skip] {name}: dataset not found", flush=True)
                continue
            print(f"\n[carregado] {name}: {X.shape} ({X.nbytes/1e6:.1f} MB)", flush=True)
            sc = scenario_motor(engine_target, X, name, did, nq=min(NQ, max(20, len(X) // 100)),
                                local_gt=use_ext_gt)
            all_out["scenarios"].append(sc)
        else:
            X = build_word2vec_degraded()
            print(f"\n[gerado] word2vec_degradado: {X.shape} ({X.nbytes/1e6:.1f} MB)", flush=True)
        # fluxo de produção normalize→madhava (sempre roda)
        prod = scenario_prod(prod_target, X, name, preset, nq=min(60, max(20, len(X) // 100)))
        all_out["scenarios"].append(prod)

    # --- 3. Pipeline PoC (declarado) ---
    pipe_target = _target_dir("pipe")
    _install(pipe_target, f"winnex-madhava=={VERSION_MADHAVA}",
             f"winnex-nano=={VERSION_NANO}", f"winnex-xfactor=={VERSION_XFACTOR}",
             f"winnex-ai-normalize=={VERSION_NORMALIZE}")
    for pkg in [f"winnex-nano=={VERSION_NANO}", f"winnex-xfactor=={VERSION_XFACTOR}"]:
        all_out["meta"]["wheel_fingerprints"][pkg.split("==")[0]] = wheel_fingerprint(pkg)
    all_out["scenarios"].append(scenario_pipeline(pipe_target))

    # --- salvar ---
    out_dir = "/kaggle/working/results" if IS_KAGGLE else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_local")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "protocol": ("honest — PyPI isolated venvs; datasets reais; GT exato externo "
                     "(3 clássicos) + search_exact (ceiling); sem fallbacks/mocks; "
                     "witness/certificado do motor expostos e verificados"),
        "witness_definition": ("search() captura no momento da decisão o threshold GLOBAL "
                               "(score exato do K-ésimo resultado) e, por doc, o bound "
                               "UB(v,q)=<Pv,Pq>+e(v)e(q); UB<threshold ⇒ provado fora do top-K"),
        "certificate_definition": ("search_audited expande o witness em registros "
                                   "GovAuditRecord (doc_id, true_cosine, projected_cosine, "
                                   "residual_norm, upper_bound, threshold, excluded, stage)"),
        "versions": {"winnex-madhava": VERSION_MADHAVA,
                     "winnex-ai-normalize": VERSION_NORMALIZE},
        "packages": all_out["meta"]["wheel_fingerprints"],
        "limitations": ("sem GT semântico externo (recall auto-consistente no dataset "
                        "degradado); o custo do GT numpy é o custo de auditoria; "
                        "auto-executado — reprodutibilidade é a mitigação; "
                        "nano/xfactor marcados conceitual/refutado nos docs do workspace"),
    }
    for fname in ["summary.json"]:
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(summary, f, indent=2)
    for sc in all_out["scenarios"]:
        did = sc.get("dataset", "pipeline").replace("/", "__")
        path = os.path.join(out_dir, f"{did}.json")
        with open(path, "w") as f:
            json.dump(sc, f, indent=2)
        print(f"Saved: {path}", flush=True)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
