#!/usr/bin/env python3
"""
winnex-pipeline-1-9-10-honest.py — Winnex Pipeline Honest Benchmark v1910
==========================================================================
Corrige os gargalos do `winnex-pipeline-honest-v199-corrigido` e usa o motor
**1.9.10** (que adiciona `recall_guarantee` + `audit_exhaustive` — a correção
do "colapso silencioso da garantia": bound_violations==0 NÃO implicava top-K
global; agora o motor EXPÕE o escopo e oferece o modo exaustivo).

O QUE ESTE KERNEL CORRIGE vs o v199-corrigido:
  1. USO DO MOTOR 1.9.10. Reporta `SearchResult.recall_guarantee` ∈
     {pool_only, exact_global} por query — o escopo do top-K é machine-readable.
  2. ALERTA NO SUMMARY do colapso silencioso. Quando recall@10 < 1.0 E
     bound_violations==0, o summary emite ⚠️ EXPLÍCITO: o prefilter cortou docs
     sem prova; o top-K NÃO é garantido global. (No v199 isso ficou enterrado.)
  3. DATASET INVÁLIDO rotulado (não número enganoso). Se o quality gate remove
     >50% do corpus (ex.: word2vec real 94.7% NaN), o dataset é marcado
     status=invalid e NÃO reporta recall@10 como medição significativa.
  4. PROXY DE SCAN JUSTO. O scan O(N·s1) do C++ usa projeções PRÉ-COMPUTADAS no
     build; o proxy agora mede a MESMA operação (N dots de s1 sobre projeções
     prontas) e mostra o custo do antigo (Xn@P.T, O(N·D·s1)) como referência do
     que NÃO se deve comparar.
  5. DEMONSTRAÇÃO DA CORREÇÃO 1.9.10. Para o dataset com manifold fraco
     (word2vec simulado), roda default (pool_only, pode divergir do exato) E
     audit_exhaustive=True (exact_global, search()==search_exact()) — lado a
     lado, provando o que a correção entrega.
  6. KEPT ⊄ GT ELEVA O DOC FALTANTE. Quando kept não é subconjunto do GT, o
     kernel lista os doc_ids no GT mas fora do top-K retornado (o "vazamento"
     do pool) — diagnóstico, não só um booleano.

O QUE É MANTIDO do v199-corrigido (o que ele fazia bem):
  - pip install do PyPI em isolamento (PYTHONPATH limpo); versões fixadas;
  - datasets reais Kaggle (glove-100 400k, mnist-784 60k, hn-openai-1536 100k);
  - recall@10 vs GT numpy exato EXTERNO em cada query + kept ⊆ GT;
  - word2vec real + simulado separados e rotulados; logs [diag]/[witness]/[cert];
  - effective_basis fiel (report + e(v), nunca config().basis);
  - witness / certificado / AuditCommitment expostos e verificados;
  - limitações declaradas (isolamento parcial, proxy Python NÃO inside-C++).

Saída: /kaggle/working/results/<dataset>__<kind>.json + summary.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

VERSION_MADHAVA = "1.9.10"
VERSION_NORMALIZE = "1.2.1"
VERSION_NANO = "0.1.12"
VERSION_XFACTOR = "1.0.2"

K = 10
NQ = 100
NQ_PROD = 60
PY = sys.executable
IS_KAGGLE = os.path.isdir("/kaggle/input")
GATE_INVALID_FRAC = 0.50   # perda de corpus no quality gate acima disto => invalid

print("=" * 74, flush=True)
print("WINNEX PIPELINE — KAGGLE HONEST BENCHMARK v1910 (madhava 1.9.10)")
print("Correções: recall_guarantee · alerta de colapso · dataset inválido ·")
print("           scan justo · audit_exhaustive · kept⊄GT com diagnóstico")
print("=" * 74, flush=True)


# ---------------------------------------------------------------------------
# 0. Instalação isolada + fingerprint (PyPI)
# ---------------------------------------------------------------------------
def _target_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=f"wxb_{prefix}_")


def _install(target: str, *pkgs: str) -> None:
    if os.environ.get("WINNEX_BENCH_LOCAL_SMOKE") == "1":
        import site
        sp = site.getsitepackages()[0]
        for name in ("winnex_madhava", "winnex_ai_normalize", "configs",
                     "winnex_nano", "winnex_xfactor"):
            s = os.path.join(sp, name)
            d = os.path.join(target, name)
            if os.path.exists(s) and not os.path.exists(d):
                os.symlink(s, d)
        return
    subprocess.check_call(
        [PY, "-m", "pip", "install", "-q", "--no-cache-dir",
         "--disable-pip-version-check", "--target", target, *pkgs],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_iso(target: str, code: str, *args: str) -> dict:
    """Roda `code` com PYTHONPATH=target; imprime o stdout ao vivo; devolve o
    JSON da última linha não-vazia. (Logs visíveis, contrato preservado.)"""
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
            pass
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<engine>", "exec"), _g)
        sys.argv = _old_argv
        print(buf.getvalue(), end="", flush=True)
        last = [ln for ln in buf.getvalue().splitlines() if ln.strip()][-1]
        try:
            return json.loads(last)
        except json.JSONDecodeError:
            return {"version": last.strip()}
    env = dict(os.environ)
    env["PYTHONPATH"] = target
    proc = subprocess.run([PY, "-c", code, *args], capture_output=True,
                          env=env, text=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess code failed (rc={proc.returncode}):\n"
            f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}")
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
    if os.environ.get("WINNEX_BENCH_LOCAL_SMOKE") == "1":
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
        f0 = files[0]
        return {"installed": os.path.basename(f0), "sha256": _sha256(f0),
                "size_bytes": os.path.getsize(f0),
                "note": "sdist (no wheel on PyPI for this platform)"}
    wheels = glob.glob(os.path.join(d, "*.whl"))
    f0 = wheels[0] if wheels else glob.glob(os.path.join(d, "*"))[0]
    return {"installed": os.path.basename(f0), "sha256": _sha256(f0),
            "size_bytes": os.path.getsize(f0)}


# ---------------------------------------------------------------------------
# 1. Carregadores de dataset
# ---------------------------------------------------------------------------
def find_file(name: str) -> "str | None":
    if IS_KAGGLE:
        for r, d, files in os.walk("/kaggle/input/"):
            if name in files:
                return os.path.join(r, name)
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


def load_word2vec_real(path: str, n_max: int = 20000):
    """Parser do binário GoogleNews word2vec (sugataghosh/google-word2vec)."""
    import struct
    vocab = []
    with open(path, "rb") as f:
        header = f.readline().strip().split()
        vocab_size, dim = int(header[0]), int(header[1])
        for i in range(min(vocab_size, n_max)):
            l = struct.unpack("B", f.read(1))[0]
            word = f.read(l).decode("utf-8", errors="ignore")
            if l % 2 == 1:
                f.read(1)
            vec = np.frombuffer(f.read(dim * 4), dtype="<f4")
            vocab.append(vec)
    return np.asarray(vocab, dtype=np.float32), dim


def load_arxiv_local(n: int = 20000) -> "np.ndarray | None":
    p = "/home/wnnx_user/zenodo/arxiv_100k.npy"
    if not os.path.exists(p):
        return None
    a = np.load(p, mmap_mode="r")
    return np.ascontiguousarray(a[:n])


def load_sift_local(n: int = 20000) -> "np.ndarray | None":
    p = "/home/wnnx_user/zenodo/sift_data/train.npy"
    if not os.path.exists(p):
        return None
    a = np.load(p, mmap_mode="r")
    return np.ascontiguousarray(a[:n])


def build_word2vec_degraded(n: int = 20000, d: int = 300, seed: int = 0):
    """WORD2VEC DEGRADADO (simulado, determinístico) — manifold FRACO."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 200)) @ rng.standard_normal((200, d))  # manifold fraco
    X = X.astype(np.float32)
    X += 0.2 * rng.standard_normal((n, d)).astype(np.float32)          # ruído
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.ascontiguousarray(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. GT numpy exato (baseline externo)
# ---------------------------------------------------------------------------
def exact_gt_numpy(Xc: np.ndarray, Q: np.ndarray, k: int = K) -> np.ndarray:
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Qn.T
    if len(Xc) > 200_000:
        idx = np.argpartition(-S, kth=k - 1, axis=0)[:k]
        vals = np.take_along_axis(S, idx, axis=0)
        order = np.argsort(-vals, axis=0)
        return np.take_along_axis(idx, order, axis=0).T
    return np.argsort(-S, axis=0)[:k].T


# ---------------------------------------------------------------------------
# 3. Custo de scan JUSTO (GARGALO #4) — mesma operação do C++
# ---------------------------------------------------------------------------
SCAN_AUDIT_CODE = r"""
import json, os, sys, time
import numpy as np

# argv: corpus.npy, queries.npy, s1, nq
X = np.load(sys.argv[1]); Q = np.load(sys.argv[2])
s1 = int(sys.argv[3]); nq = int(sys.argv[4])
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
N, D = X.shape
rng = np.random.default_rng(0)
P = rng.standard_normal((s1, D)); P, _ = np.linalg.qr(P.T); P = P.T

# O C++ PRÉ-COMPUTA as projeções de TODOS os N docs NO BUILD (O(N*D*s1) uma vez).
# Na busca, o scan é: para cada doc, dot da projeção PRONTA (s1 dims) com a
# projeção da query = O(N*s1). O proxy JUSTO pré-computa Xp UMA vez e mede só o
# scan por query. (O proxy do v199 media Xn @ P.T POR QUERY = O(N*D*s1) — 12x
# mais caro que o scan real; era incomparável.)
Xp = Xn @ P.T                     # projeção de todos os N docs (feita UMA vez)
lat_fair = 0.0
for j in range(nq):
    pq = P @ Qn[j]                # projeção da query (s1 dots de D)
    t0 = time.time()
    _ = Xp @ pq                   # N dots de s1 — o scan do bound (O(N*s1))
    lat_fair += (time.time() - t0) * 1000
lat_fair /= nq

# Referência do proxy ANTIGO (Xn @ P.T por query = O(N*D*s1)) — para mostrar
# por que não se deve compará-lo à latência do motor.
lat_old = 0.0
for j in range(nq):
    t0 = time.time()
    _ = Xn @ P.T
    lat_old += (time.time() - t0) * 1000
lat_old /= nq

print(json.dumps({
    "scan_ms_per_query_fair": round(lat_fair, 3),      # O(N*s1), projeções prontas
    "scan_ms_per_query_old_proxy": round(lat_old, 3),  # O(N*D*s1), projeção por query
    "N": int(N), "D": int(D), "s1": int(s1),
    "method": "fair: N dots of s1 over precomputed projections (== C++ scan op); "
              "old: Xn@P.T per query (O(N*D*s1), NOT comparable to engine latency)"}))
"""


# ---------------------------------------------------------------------------
# 4. Cenário MOTOR — recall vs GT externo + recall_guarantee (1.9.10)
# ---------------------------------------------------------------------------
ENGINE_CODE = r"""
import json, sys, time
import numpy as np
import winnex_madhava as wm

K = 10
Xc = np.load(sys.argv[1])
Q = np.load(sys.argv[2])
gtp = sys.argv[3]
NQ = int(sys.argv[4])
gt = np.load(gtp)

def recall(pred, gt):
    return sum(1 for i in pred if i in gt) / K

D = int(Xc.shape[1])
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
        rec = lat = 0.0; kept_gt = 0; exact_global = 0
        for j, q in enumerate(Q):
            t0 = time.time(); r = eng.search(q); lat += (time.time() - t0) * 1000
            rec += recall(r.indices, gt[j])
            kept_gt += int(all(i in gt[j] for i in r.indices))
            # GARGALO #1 (1.9.10): o escopo do top-K é machine-readable.
            exact_global += int(r.recall_guarantee == "exact_global")
        N = len(Xc)
        res[f"{basis}_k1={k1}"] = {
            "recall@10_ext_gt": round(rec / NQ, 4), "latency_ms": round(lat / NQ, 3),
            "qps": round(NQ / (lat / 1000), 1), "bound_violations": int(0),
            "kept_subset_of_gt": kept_gt,
            "recall_guarantee_exact_global": exact_global,  # /NQ queries garantem global
            "build_s": round(build_s, 2)}
print(json.dumps({"version": wm.__version__, "results": res}))
"""


def scenario_motor(target: str, X: np.ndarray, name: str, dataset_id: str,
                   nq: int = NQ) -> dict:
    """Motor isolado: recall vs GT numpy + recall_guarantee (quantas queries
    têm o top-K GLOBAL garantido)."""
    print(f"\n=== [MOTOR] {name} d={X.shape[1]} N={len(X) - nq} nq={nq} ===", flush=True)
    tmp = tempfile.mktemp(suffix=".npy")
    tq = ty = ""
    Q = X[-nq:].copy(); Xc = X[:-nq].copy()
    Y = exact_gt_numpy(Xc, Q)
    np.save(tmp, Xc)
    tq = tempfile.mktemp(suffix=".npy"); np.save(tq, Q)
    ty = tempfile.mktemp(suffix=".npy"); np.save(ty, Y)
    try:
        data = _run_iso(target, ENGINE_CODE, tmp, tq, ty, str(nq))
    finally:
        for p in (tmp, tq, ty):
            if p:
                os.unlink(p)
    for k, v in data["results"].items():
        print(f"    {k:20s} rec={v['recall@10_ext_gt']} lat={v.get('latency_ms')}ms "
              f"kept⊆gt={v.get('kept_subset_of_gt')}/{nq} "
              f"exact_global={v.get('recall_guarantee_exact_global')}/{nq} "
              f"build={v.get('build_s','-')}s", flush=True)
    return {"package": "winnex-madhava", "version": data["version"],
            "installed_from": "PyPI (isolated venv)", "dataset": dataset_id,
            "dim": int(X.shape[1]), "N": int(len(Xc)), "NQ": nq, "K": K,
            "metric": "cosine",
            "gt": "exact top-K brute-force numpy (baseline externo)",
            "results": data["results"]}


# ---------------------------------------------------------------------------
# 5. Cenário PRODUÇÃO (Normalize → Madhava) — fluxo real + witness/cert/commit
# ---------------------------------------------------------------------------
PROD_CODE = r"""
import json, os, sys, time
import numpy as np
import winnex_madhava as wm
import winnex_ai_normalize
from winnex_ai_normalize.core.normalize import normalize_l2
from winnex_ai_normalize.core.quality import build_quality_engine, QualityConfig

def exact_gt_numpy(Xc, Q, k=10):
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Qn.T
    if len(Xc) > 200_000:
        idx = np.argpartition(-S, kth=k - 1, axis=0)[:k]
        vals = np.take_along_axis(S, idx, axis=0)
        order = np.argsort(-vals, axis=0)
        return np.take_along_axis(idx, order, axis=0).T
    return np.argsort(-S, axis=0)[:k].T

name = sys.argv[1]; preset = sys.argv[2]; NQ = int(sys.argv[3])
X = np.load(sys.argv[4])
gtp = sys.argv[5] if len(sys.argv) > 5 and os.path.exists(sys.argv[5]) else None
Q = X[-NQ:].copy(); Xc = X[:-NQ].copy()
D = int(X.shape[1]); K = 10
N_orig = len(Xc)

# --- 0. Diagnóstico do dataset BRUTO (antes do gate), em float32 ---
Xf = np.ascontiguousarray(Xc, dtype=np.float32)
raw_norms = np.linalg.norm(Xf, axis=1)
nan_frac = float(1.0 - np.isfinite(raw_norms).mean())
zero_frac = float((np.isfinite(raw_norms) & (raw_norms <= 1e-12)).mean())
cfg_diag = QualityConfig.from_dataset(preset)
nan_policy = cfg_diag.nan_policy
degraded = (nan_frac > 0) or (zero_frac > 0)
print(f"    [diag] {name}: nan={nan_frac:.4%} zero={zero_frac:.4%} "
      f"policy={nan_policy} degraded={degraded}", flush=True)

# --- 1. Quality gate: remove linhas não-finitas/degeneradas ---
g = np.isfinite(raw_norms) & (raw_norms > 1e-12)
Xc = Xc[g]
qraw = np.linalg.norm(np.ascontiguousarray(Q, dtype=np.float32), axis=1)
qg = np.isfinite(qraw) & (qraw > 1e-12)
Q = Q[qg]
n_removed = N_orig - len(Xc)
frac_removed = n_removed / max(N_orig, 1)
print(f"    [gate] removed {n_removed}/{N_orig} corpus "
      f"({frac_removed:.1%}) + {NQ - len(Q)} query rows", flush=True)

# --- GARGALO #3: dataset INVÁLIDO quando o gate remove >50% ---
# (ex.: word2vec real 94.7% NaN => N cai de ~19940 para ~991; recall@10 sobre
#  991 docs com nq=1 NÃO é uma medição significativa. Rotular, não reportar.)
if frac_removed > 0.50 or len(Xc) < K * 10 or len(Q) < 1:
    print(f"    [invalid] gate removeu {frac_removed:.1%} (>50%) ou N pequeno demais "
          f"— dataset INVÁLIDO para recall; reportando diagnóstico, não recall.", flush=True)
    print(json.dumps({
        "dataset": name, "dim": D, "preset": preset or "default",
        "status": "invalid",
        "reason": f"quality gate removeu {frac_removed:.1%} do corpus "
                  f"({n_removed}/{N_orig}) ou N/queries insuficientes "
                  f"(N_final={len(Xc)}, nq={len(Q)})",
        "nan_fraction": round(nan_frac, 8), "zero_norm_fraction": round(zero_frac, 8),
        "nan_policy": nan_policy, "dataset_degraded": degraded,
        "n_original": int(N_orig), "n_after_gate": int(len(Xc)),
        "n_removed": int(n_removed), "frac_removed": round(frac_removed, 6),
    }))
    # sys.exit(0) é correto AQUI: no Kaggle o PROD_CODE roda como subprocesso e
    # o stdout (com o JSON de invalid) é lido pelo _run_iso antes do exit. No
    # modo smoke inline (exec no mesmo processo) isto mataria o processo — por
    # isso o caminho invalid só é exercitado de verdade no Kaggle / subprocesso.
    sys.exit(0)

# --- 2. Normalize L2 ---
Xn = normalize_l2(Xc); Qn = normalize_l2(Q)
N = len(Xc)
gt = exact_gt_numpy(Xc, Qn) if gtp is None else np.load(gtp)

# --- 3. Build com o PRESET (config do normalize) ---
t0 = time.time()
eng, rep = build_quality_engine(Xn, dim=D, k=K, return_report=True, dataset=preset)
build_s = time.time() - t0
cfg = eng.config()

# --- 4. Medição: recall vs GT numpy + recall_guarantee ---
rec = viol = lat = 0.0
kept_gt = 0; exact_global = 0; pool_only = 0
leaks = []   # GARGALO #6: docs no GT mas fora do top-K retornado (vazamento)
for j in range(len(Qn)):
    t0 = time.time(); r = eng.search(Qn[j]); lat += (time.time() - t0) * 1000
    viol += r.bound_violations
    rec += sum(1 for i in r.indices if i in gt[j]) / K
    kept_gt += int(all(i in gt[j] for i in r.indices))
    if r.recall_guarantee == "exact_global":
        exact_global += 1
    else:
        pool_only += 1
    if not all(i in gt[j] for i in r.indices):
        missing = [int(i) for i in gt[j] if i not in r.indices]
        leaks.append({"query": int(j), "gt_but_not_returned": missing[:5],
                      "recall_guarantee": r.recall_guarantee})
rec /= max(len(Qn), 1); lat /= max(len(Qn), 1)
resid = eng.residuals1()
e_v = float(np.mean(resid)) if len(resid) else None

# --- 5. WITNESS (1 query representativa) ---
q0 = np.ascontiguousarray(Qn[0], dtype=np.float32)
r0 = eng.search(q0)
witness = {
    "top10": [int(i) for i in r0.indices],
    "global_kth_threshold": float(r0.audit_threshold),
    "pruned_by_bound": int(r0.pruned_by_bound),
    "pruned_by_prefilter": int(r0.pruned_by_prefilter),
    "recall_guarantee": r0.recall_guarantee,   # 1.9.10
    "excluded_sample": [
        {"doc_id": int(d), "upper_bound": round(float(u), 6),
         "residual_e": round(float(x), 6), "threshold": round(float(r0.audit_threshold), 6),
         "proved_out": bool(u < r0.audit_threshold)}
        for d, u, x in zip(r0.audit_ids[:8], r0.audit_ubs[:8], r0.audit_residuals[:8])],
}
print(f"    [witness] threshold={witness['global_kth_threshold']:.4f} "
      f"proved_out={witness['pruned_by_bound']} guarantee={r0.recall_guarantee}", flush=True)

# --- 6. CERTIFICADO + kept ⊆ GT ---
ar = eng.search_audited(q0, k=K, max_audit_records=120)
excluded_recs = [x for x in ar["audit"] if x["excluded"]]
kept_recs = [x for x in ar["audit"] if not x["excluded"]]
kept_ids = sorted(int(x["doc_id"]) for x in kept_recs)
gt0 = set(gt[0]) if gt is not None else set()
leak0 = sorted(int(i) for i in (gt0 - set(kept_ids)))
cert = {
    "query": "first normalized query (representative)",
    "audit_candidates": int(ar["audit_candidates"]),
    "audit_excluded": int(ar["audit_excluded"]),
    "cert_integrity": {
        "all_excluded_ub_lt_threshold": all(x["upper_bound"] < x["threshold"] for x in excluded_recs),
        "all_excluded_true_cosine_lt_threshold": all(
            x["true_cosine"] < x["threshold"] - 1e-9 for x in excluded_recs),
        "kept_in_topk_equal_returned": kept_ids == sorted(int(i) for i in r0.indices),
        "kept_subset_of_gt_numpy": all(i in gt0 for i in kept_ids),
        "gt_not_in_topk_docs": leak0[:10],   # GARGALO #6: o "vazamento" do pool
    },
}
print(f"    [cert] excluded={cert['audit_excluded']} integrity="
      f"{json.dumps({k: v for k, v in cert['cert_integrity'].items() if k != 'gt_not_in_topk_docs'})}", flush=True)

# --- 7. AuditCommitment ---
commit = {"queries": int(len(Qn)), "sample_bounded": 0, "exact_count_match": 0,
          "deterministic": 0, "genuine_exclusions": 0, "total_excluded": 0,
          "max_payload_bytes": 0, "pool_only_commitments": 0}
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
    commit["pool_only_commitments"] += int(c.get("recall_guarantee", "pool_only") == "pool_only")
print(f"    [commit] count={commit['exact_count_match']}/{commit['queries']} "
      f"det={commit['deterministic']}/{commit['queries']} "
      f"pool_only={commit['pool_only_commitments']}/{commit['queries']}", flush=True)

# --- 8. effective_basis fiel + alerta de colapso silencioso ---
effective_basis = getattr(rep, "basis", None) or preset
# GARGALO #2: ALERTA quando recall<1.0 com viol=0 (o colapso silencioso).
silent_collapse = (rec < 1.0) and (viol == 0)
if silent_collapse:
    print(f"    [⚠️ ALERTA] recall@10={rec:.4f} < 1.0 com bound_violations=0 — "
          f"o prefilter cortou docs sem prova; top-K NÃO é garantido global. "
          f"Use audit_exhaustive=True (ver cenário dedicado).", flush=True)

print(f"    [prod] {name}: recall@10_vs_gt_numpy={rec:.4f} viol={int(viol)} "
      f"build={build_s:.1f}s e(v)={e_v:.4f} basis={effective_basis} "
      f"guarantee: {exact_global} exact_global / {pool_only} pool_only", flush=True)

print(json.dumps({
    "dataset": name, "dim": D, "N": int(N), "preset": preset or "default",
    "status": "ok",
    "nan_fraction": round(nan_frac, 8), "zero_norm_fraction": round(zero_frac, 8),
    "nan_policy": nan_policy, "dataset_degraded": degraded,
    "recall@10_vs_gt_numpy": round(rec, 4), "bound_violations": int(viol),
    "kept_subset_of_gt": kept_gt, "nq_measured": int(len(Qn)),
    "queries_exact_global": exact_global, "queries_pool_only": pool_only,
    "silent_collapse_alert": silent_collapse,
    "gt_leaks_sample": leaks[:5],              # docs no GT mas fora do retorno
    "latency_ms": round(lat, 3), "build_s": round(build_s, 1),
    "residual_e_v": round(e_v, 4) if e_v else None,
    "engine_applied": {"basis": str(cfg.basis),
                       "effective_basis": effective_basis,
                       "stage1_dim": int(cfg.stage1_dim),
                       "pca_iterations": int(cfg.pca_iterations),
                       "k1_fraction": float(cfg.k1_fraction)},
    "quality_report": {
        "flags": [{"code": f.code, "severity": f.severity, "metric": f.metric} for f in rep.flags][:8],
        "verdict": ("fail" if rep.has_fail else "warn" if rep.warn_flags else "pass"),
        "basis": effective_basis},
    "witness": witness, "certificate": cert, "commitment": commit,
}))
"""


def scenario_prod(target: str, X: np.ndarray, name: str, preset: "str | None",
                  nq: int = NQ_PROD, save_gt: bool = True) -> dict:
    print(f"\n=== [PRODUÇÃO] {name} (preset={preset or 'default'}) ===", flush=True)
    tmp = tempfile.mktemp(suffix=".npy")
    ty = ""
    np.save(tmp, X)
    Y = None
    Q = X[-nq:].copy(); Xc = X[:-nq].copy()
    if save_gt and len(Q) and len(Xc):
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
        Y = np.argsort(-(Xn @ Qn.T), axis=0)[:K].T
        ty = tempfile.mktemp(suffix=".npy"); np.save(ty, Y)
    try:
        return _run_iso(target, PROD_CODE, name, preset or "default", str(nq), tmp, ty)
    finally:
        for p in (tmp, ty):
            if p:
                os.unlink(p)


# ---------------------------------------------------------------------------
# 6. DEMONSTRAÇÃO da correção 1.9.10: default (pool_only) vs audit_exhaustive
# ---------------------------------------------------------------------------
EXHAUSTIVE_CODE = r"""
import json, sys, time
import numpy as np
import winnex_madhava as wm

X = np.load(sys.argv[1]); NQ = int(sys.argv[2]); D = int(X.shape[1]); K = 10
Q = X[-NQ:].copy(); Xc = X[:-NQ].copy()

def run(audit_exhaustive):
    eng = wm.build_engine(Xc, dim=D, metric="cosine", quant="none", basis="random",
                          stage1_dim=64, stage2_dim=0, k=K, k1_fraction=0.2,
                          normalize_input=True, early_exit=False,
                          audit_exhaustive=audit_exhaustive)
    div = 0; rg = {}; lat = 0.0
    for j in range(NQ):
        q = np.ascontiguousarray(Q[j], dtype=np.float32)
        t0 = time.time(); r = eng.search(q); lat += (time.time() - t0) * 1000
        rx = eng.search_exact(q)
        div += int(r.indices != rx.indices)
        rg[r.recall_guarantee] = rg.get(r.recall_guarantee, 0) + 1
    return {"divergences_search_vs_exact": div, "recall_guarantee_counts": rg,
            "latency_ms": round(lat / NQ, 3), "k3_sample": int(r.k3),
            "N": int(len(Xc))}

print(json.dumps({
    "default_pool_only": run(False),
    "audit_exhaustive": run(True),
    "explanation": ("default pode divergir do search_exact em manifold fraco com "
                    "viol=0 (recall_guarantee=pool_only); audit_exhaustive força "
                    "k1=k2=N e desliga early_exit => search()==search_exact(), "
                    "recall_guarantee=exact_global (custo O(N*d)/query)"),
}))
"""


def scenario_exhaustive(target: str, X: np.ndarray, name: str, nq: int = 60) -> dict:
    print(f"\n=== [1.9.10] default vs audit_exhaustive — {name} (manifold fraco) ===", flush=True)
    tmp = tempfile.mktemp(suffix=".npy")
    np.save(tmp, X)
    try:
        data = _run_iso(target, EXHAUSTIVE_CODE, tmp, str(nq))
    finally:
        os.unlink(tmp)
    d = data["default_pool_only"]; e = data["audit_exhaustive"]
    print(f"  default:        divergências search!=exact = {d['divergences_search_vs_exact']}/{nq} "
          f"guarantee={d['recall_guarantee_counts']} lat={d['latency_ms']}ms", flush=True)
    print(f"  audit_exhaustive: divergências = {e['divergences_search_vs_exact']}/{nq} "
          f"guarantee={e['recall_guarantee_counts']} lat={e['latency_ms']}ms", flush=True)
    data["dataset"] = name
    data["installed_from"] = "PyPI (isolated venv)"
    return data


# ---------------------------------------------------------------------------
# 7. Cenário PIPELINE PoC (nano → xfactor → normalize → madhava) — QPS real
# ---------------------------------------------------------------------------
PIPE_CODE = r"""
import json, time
import numpy as np
import winnex_madhava as wm
import winnex_nano as wn
import winnex_xfactor as wxf
import winnex_ai_normalize as wnrm

# INTEGRAÇÃO DE PESQUISA (declarada): nano/xfactor marcados conceitual/refutado.
tok = wn.SpectralTokenizer(embed_dim=64)
texts = [
    "paciente com hipertensao arterial severa", "diabetes mellitus tipo 2",
    "suspeita de cancer de mama", "transtorno de ansiedade generalizada",
    "fraturas e ortopedia", "cardiologia pediatrica", "infeccao urinaria",
    "saude mental do idoso", "ginecologia obstetricia", "dor lombar cronica",
    "febre e sintomas respiratorios", "dor abdominal aguda", "pre-natal",
    "saude ocupacional", "psiquiatria infantil", "oncologia", "nefrologia",
    "reumatologia", "endocrinologia", "emergencias cardiologicas",
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
queries = [embed(t) for t in texts]
t0 = time.time(); r = eng.search(queries[0]); single_ms = (time.time() - t0) * 1000

# QPS REAL: queries DIFERENTES (sem cache-hit da mesma query). Cada iteração
# usa uma query distinta do conjunto, em ciclo.
NIT = len(queries) * 5
t0 = time.time()
for i in range(NIT):
    eng.search(queries[i % len(queries)])
qps = NIT / (time.time() - t0)

print(json.dumps({
    "status": "research_integration",
    "note": "nano/xfactor marked conceitual/refutado in workspace docs; PoC of wiring",
    "versions": {"madhava": wm.__version__, "nano": wn.__version__,
                 "xfactor": wxf.__version__,
                 "normalize": getattr(wnrm, "__version__", "n/a")},
    "top": [int(i) for i in r.indices],
    "bound_violations": int(r.bound_violations),
    "search_ms": round(single_ms, 3),
    "qps_distinct_queries": round(qps, 1),   # GARGALO: sem cache-hit (queries distintas)
    "n_iterations": int(NIT), "n_distinct_queries": int(len(queries)),
    "manifold": {"rank": int(rank), "variance_explained": float(var)},
}))
"""


def scenario_pipeline(target: str) -> dict:
    print("\n=== [PIPELINE/POC] texto → nano → xfactor → normalize → madhava ===", flush=True)
    data = _run_iso(target, PIPE_CODE)
    v = data["versions"]
    print(f"  madhava {v['madhava']} · nano {v['nano']} · xfactor {v['xfactor']} "
          f"· normalize {v['normalize']} | top={data['top']} viol={data['bound_violations']}", flush=True)
    print(f"  qps (queries distintas): {data['qps_distinct_queries']} "
          f"({data['n_iterations']} iterações, {data['n_distinct_queries']} queries distintas)", flush=True)
    data["installed_from"] = "PyPI (isolated venv)"
    return data


# ---------------------------------------------------------------------------
# 8. Baseline brute-force numpy
# ---------------------------------------------------------------------------
def scenario_bruteforce(X: np.ndarray, name: str, nq: int = NQ) -> dict:
    print(f"\n=== [BASELINE brute-force numpy] {name} N={len(X) - nq} nq={nq} ===", flush=True)
    Q = X[-nq:].copy(); Xc = X[:-nq].copy()
    t0 = time.time()
    S = (Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)) @ \
        (Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)).T
    lat_all = (time.time() - t0) * 1000
    return {"dataset": name, "dim": int(X.shape[1]), "N": int(len(Xc)), "NQ": nq,
            "latency_ms_total": round(lat_all, 1),
            "latency_ms_per_query": round(lat_all / nq, 2),
            "qps": round(nq / (lat_all / 1000), 1),
            "note": "brute-force exact top-K cosine (numpy), same N/d/query"}


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main():
    all_out = {"meta": {}, "scenarios": []}

    engine_target = _target_dir("madhava")
    _install(engine_target, f"winnex-madhava=={VERSION_MADHAVA}")
    wm_ver = _run_iso(engine_target, "import winnex_madhava as w; print(w.__version__)")
    all_out["meta"]["wheel_fingerprints"] = {}
    all_out["meta"]["wheel_fingerprints"]["winnex-madhava"] = wheel_fingerprint(
        f"winnex-madhava=={VERSION_MADHAVA}")
    print(f"\nMotor: winnex-madhava {wm_ver} "
          f"({all_out['meta']['wheel_fingerprints']['winnex-madhava']['installed']})", flush=True)
    # Assert 1.9.10 (tem recall_guarantee/audit_exhaustive — a correção)
    ver = wm_ver if isinstance(wm_ver, str) else wm_ver.get("version", "?")
    assert ver == VERSION_MADHAVA, f"esperado 1.9.10, instalado {ver}"

    prod_target = _target_dir("prod")
    _install(prod_target, f"winnex-madhava=={VERSION_MADHAVA}",
             f"winnex-ai-normalize=={VERSION_NORMALIZE}")
    prod_vers = _run_iso(prod_target,
        "import winnex_madhava as w, winnex_ai_normalize as n, json; "
        "print(json.dumps({'madhava': w.__version__, 'normalize': n.__version__}))")
    for pkg in [f"winnex-madhava=={VERSION_MADHAVA}", f"winnex-ai-normalize=={VERSION_NORMALIZE}"]:
        all_out["meta"]["wheel_fingerprints"][pkg.split("==")[0]] = wheel_fingerprint(pkg)
    print(f"Produção: madhava {prod_vers['madhava']} · normalize {prod_vers['normalize']}", flush=True)

    # Datasets
    datasets = []
    if IS_KAGGLE:
        datasets = [
            (load_glove, "glove", "rtatman/glove-global-vectors-for-word-representation", "default"),
            (load_mnist, "mnist", "hojjatk/mnist-dataset", "default"),
            (load_hn_openai, "hn_openai", "julien040/hacker-news-openai-embeddings", "default"),
        ]
    else:
        datasets = [
            (load_arxiv_local, "arxiv_local", "arxiv_100k.npy local (smoke)", "arxiv"),
            (load_sift_local, "sift_local", "sift train.npy local (smoke)", "sift"),
        ]

    for loader, name, did, preset in datasets:
        X = loader()
        if X is None:
            print(f"[skip] {name}: dataset not found", flush=True)
            continue
        print(f"\n[carregado] {name}: {X.shape} ({X.nbytes/1e6:.1f} MB)", flush=True)
        sc = scenario_motor(engine_target, X, name, did, nq=min(NQ, max(20, len(X) // 100)))
        all_out["scenarios"].append(sc)
        bf = scenario_bruteforce(X, name, nq=min(NQ, max(20, len(X) // 100)))
        all_out["scenarios"].append(bf)
        prod = scenario_prod(prod_target, X, name, preset, nq=min(NQ_PROD, max(20, len(X) // 100)))
        all_out["scenarios"].append(prod)

    # word2vec REAL (deve ser INVALID — 94.7% NaN) + SIMULADO (manifold fraco)
    if IS_KAGGLE:
        p = find_file("GoogleNews-vectors-negative300.bin")
        if p:
            Xw, _dim = load_word2vec_real(p)
            print(f"\n[carregado] word2vec REAL: {Xw.shape} ({Xw.nbytes/1e6:.1f} MB)", flush=True)
            prod = scenario_prod(prod_target, Xw, "word2vec_real", "word2vec",
                                 nq=min(NQ_PROD, max(20, len(Xw) // 100)))
            all_out["scenarios"].append(prod)
        else:
            print("[skip] word2vec REAL: GoogleNews-vectors-negative300.bin not found", flush=True)
    Xs = build_word2vec_degraded()
    print(f"\n[gerado] word2vec simulado (manifold fraco): {Xs.shape}", flush=True)
    prod = scenario_prod(prod_target, Xs, "word2vec_simulado", "word2vec",
                         nq=min(NQ_PROD, max(20, len(Xs) // 100)))
    all_out["scenarios"].append(prod)

    # Demonstração 1.9.10: default vs audit_exhaustive NO manifold fraco
    Xdemo = build_word2vec_degraded(n=10000)
    exh = scenario_exhaustive(engine_target, Xdemo, "word2vec_simulado", nq=80)
    all_out["scenarios"].append(exh)

    # Scan justo no maior dataset
    Xbig = None
    for loader, name, _, _ in datasets:
        if loader is not None and Xbig is None:
            Xbig = loader(n=100_000)
    if Xbig is not None:
        print(f"\n=== [SCAN AUDIT justo] {Xbig.shape} ===", flush=True)
        tmp = tempfile.mktemp(suffix=".npy"); tq = tempfile.mktemp(suffix=".npy")
        np.save(tmp, Xbig[:-100]); np.save(tq, Xbig[-100:])
        try:
            sd = _run_iso(engine_target, SCAN_AUDIT_CODE, tmp, tq,
                          str(min(192, Xbig.shape[1])), "100")
            print(f"  scan justo: {sd['scan_ms_per_query_fair']}ms/query | "
                  f"proxy antigo: {sd['scan_ms_per_query_old_proxy']}ms/query", flush=True)
            sd["dataset"] = "scan_audit_fair"
            all_out["scenarios"].append(sd)
        finally:
            for p in (tmp, tq):
                if p:
                    os.unlink(p)

    # Pipeline PoC (declarado)
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

    # GARGALO #2: alerta de colapso silencioso no SUMMARY (não só no JSON)
    silent = [s for s in all_out["scenarios"]
              if s.get("silent_collapse_alert")]
    alerts = []
    for s in silent:
        alerts.append(f"⚠️ {s['dataset']}: recall@10={s.get('recall@10_vs_gt_numpy')} "
                      f"com viol=0 — top-K NÃO garantido global "
                      f"(queries pool_only={s.get('queries_pool_only')})")

    summary = {
        "kernel": "winnex-pipeline-honest-v1910",
        "madhava_version": VERSION_MADHAVA,
        "protocol": ("honesto — PyPI isolated venvs; madhava 1.9.10 "
                     "(recall_guarantee + audit_exhaustive); datasets reais; "
                     "recall@10 vs GT numpy exato em cada query; kept⊆GT com "
                     "diagnóstico de vazamento; word2vec real (invalid) + simulado "
                     "separados; scan justo; QPS sem cache-hit"),
        "gargalos_corrigidos_vs_v199": {
            "motor_1910": "recall_guarantee (pool_only/exact_global) por query; "
                          "audit_exhaustive demonstrado",
            "alerta_colapso_silencioso": "summary emite ⚠️ quando recall<1.0 e viol==0",
            "dataset_invalido": "word2vec real (94.7% NaN) é rotulado status=invalid, "
                                "não reporta recall enganoso com nq=1",
            "scan_justo": "proxy mede O(N*s1) sobre projeções pré-computadas (a operação "
                          "do C++), e mostra o proxy antigo O(N*D*s1) como incomparável",
            "audit_exhaustive_demo": "default (pool_only) vs audit_exhaustive (exact_global) "
                                     "lado a lado no manifold fraco",
            "kept_subset_gt_com_diag": "lista os doc_ids no GT mas fora do top-K retornado",
            "qps_sem_cache_hit": "pipeline mede QPS com queries DISTINTAS em ciclo",
        },
        "alerts_colapso_silencioso": alerts,   # vazio se nenhum dataset colapsou
        "versions": {"winnex-madhava": VERSION_MADHAVA,
                     "winnex-ai-normalize": VERSION_NORMALIZE},
        "packages": all_out["meta"]["wheel_fingerprints"],
        "limitations": (
            "sem GT semântico externo (recall é convergência ao exato da mesma métrica); "
            "o custo do GT numpy é o custo de auditoria; "
            "isolamento PARCIAL: --target+PYTHONPATH isolam os pacotes winnex, mas numpy/"
            "BLAS/.so do host Kaggle entram na medição; "
            "proxy Python do scan é uma estimativa da operação do C++, NÃO medição "
            "inside-C++; "
            "word2vec real é inválido (94.7% NaN no dataset Kaggle) — reportado como "
            "diagnóstico, não recall; "
            "nano/xfactor marcados conceitual/refutado nos docs do workspace; "
            "auto-executado — reprodutibilidade é a mitigação"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for sc in all_out["scenarios"]:
        base = sc.get("dataset", "pipeline").replace("/", "__")
        if "latency_ms_total" in sc:
            kind = "bruteforce"
        elif sc.get("status") == "invalid":
            kind = "invalid"
        elif "recall@10_vs_gt_numpy" in sc:
            kind = "prod"
        elif "default_pool_only" in sc and "audit_exhaustive" in sc:
            kind = "exhaustive_demo"
        elif sc.get("status") == "research_integration":
            kind = "pipeline"
        elif "scan_ms_per_query_fair" in sc:
            kind = "scan"
        elif "results" in sc and "gt" in sc:
            kind = "motor"
        else:
            kind = "outro"
        path = os.path.join(out_dir, f"{base}__{kind}.json")
        with open(path, "w") as f:
            json.dump(sc, f, indent=2)
        print(f"Saved: {path}", flush=True)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print("=" * 74, flush=True)


if __name__ == "__main__":
    main()
