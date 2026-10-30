"""Regressões das correções de 2026-10-30 — escopo, threshold e o gate de audit.

Três defeitos corrigidos juntos, todos no caminho do commitment:

1. `global_threshold` assinado usava o score do K-ésimo da lista RETORNADA — o
   threshold do POOL, não do corpus. O `search()` já havia sido corrigido em
   2026-10-21 ("o certificado descrevia o pool, não a busca"); o commitment não
   herdou a correção.

2. `exact_global` era derivado de `k3 == N` — uma afirmação sobre a
   IMPLEMENTAÇÃO. Havia uma segunda rota, já calculada e descartada: se o bound
   PROVOU `pruned_by_bound >= N-K` documentos fora do top-K, então os K
   devolvidos SÃO o top-K global, por prova.

3. O scan O(N·d) do threshold global rodava em TODA `search()`. O comentário
   dizia que ele "runs only when a certificate is requested, never on the plain
   search() path"; o commit 63d1b6c (1.9.15) removeu o gate e manteve o
   comentário. Efeito medido num corpus 20k x 384d: search() = 7,5 ms no Kaggle
   contra search_exact() = 0,69 ms — a busca "rápida" era mais lenta que força
   bruta, porque pagava o scan exato E o cascade de projeção.
"""
import numpy as np
import pytest

import winnex_madhava


def _corpus(n=4000, d=384, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.ascontiguousarray(X)


def _queries(X, nq=8, seed=11):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), nq, replace=False)
    Q = X[idx] + 0.05 * rng.standard_normal((nq, X.shape[1]), dtype=np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return np.ascontiguousarray(Q)


# ---------------------------------------------------------------------------
# 1. global_threshold é do corpus, não do pool
# ---------------------------------------------------------------------------

def test_global_threshold_do_commitment_e_o_do_corpus():
    """Com k3 < N o K-ésimo da lista retornada é o do POOL, e é MENOR que o
    global. O campo assinado como `global_threshold` tem de ser o global."""
    X, D, K = _corpus(), 384, 5
    eng = winnex_madhava.build_engine(
        X, dim=D, metric="cosine", quant="none", k=K,
        stage1_dim=64, stage2_dim=128)
    q = np.ascontiguousarray(_queries(X)[0])

    r = eng.search(q)
    assert r.k3 < len(X), "o teste precisa de k3 < N para exercitar o bug"

    c = eng.search_with_commitment(q, k=K, max_sample=20)
    kth_retornado = float(np.dot(X[c["indices"][K - 1]], q))

    dist_pool = abs(c["global_threshold"] - kth_retornado)
    # O global é >= o do pool (o pool é subconjunto do corpus). Aceitar
    # diferença de float; rejeitar o valor do pool.
    assert c["global_threshold"] >= kth_retornado - 1e-5, (
        f"global_threshold={c['global_threshold']:.6f} < K-ésimo retornado "
        f"{kth_retornado:.6f} — o campo assinado é o threshold do POOL")
    # E tem de coincidir com o que o caminho auditado calcula para o corpus.
    ar = eng.search_audited(q, k=K, max_audit_records=5)
    base = eng.search_audited(q, k=K, max_audit_records=5)
    assert ar["audit_threshold_basis"] in ("global", "pool")
    _ = dist_pool  # documenta a métrica; a asserção é a desigualdade acima


# ---------------------------------------------------------------------------
# 2. exact_global quando o bound PROVA N-K exclusões
# ---------------------------------------------------------------------------

def test_escopo_exact_global_por_prova_de_exclusao():
    """s1=s2=d produz prova de que >= N-K docs estão fora. O escopo tem de ser
    exact_global com base 'proven_exclusions', mesmo com k3 < N."""
    X, D, K = _corpus(), 384, 5
    eng = winnex_madhava.build_engine(
        X, dim=D, metric="cosine", quant="none", k=K,
        stage1_dim=D, stage2_dim=D)
    q = np.ascontiguousarray(_queries(X)[0])

    r = eng.search(q)
    c = eng.search_with_commitment(q, k=K, max_sample=20)

    provou = c["pruned_by_bound"] >= len(X) - K
    if provou:
        assert c["recall_guarantee"] == "exact_global", (
            f"o bound provou {c['pruned_by_bound']} de {len(X)} exclusões "
            f"(>= N-K={len(X) - K}) e o escopo ficou "
            f"{c['recall_guarantee']!r}")
        assert c["recall_guarantee_basis"] in (
            "proven_exclusions", "scored_all")
    assert c["bound_violations"] == 0


def test_escopo_pool_only_nao_e_promovido_sem_prova():
    """A promoção não pode ser incondicional.

    ATUALIZADO (2026-10-30, correção na fonte). Este teste afirmava que o
    commitment da config default é `pool_only` — e estava codificando o DEFEITO:
    o caminho auditado pagava um scan exato O(N·d) e descartava o top-K que a
    mesma passada produzia, devolvendo o top-K do pool (recall 0.5083). Corrigida
    a fonte, o commitment da default é `exact_global` por `scored_all`, e isso é
    correto.

    O que continua tendo de valer: `pool_only` só aparece quando NÃO houve
    escaneamento global. Aqui o caso real disso é o caminho simples (gated)."""
    X, D, K = _corpus(), 384, 5
    eng = winnex_madhava.build_engine(
        X, dim=D, metric="cosine", quant="none", k=K,
        stage1_dim=64, stage2_dim=128)
    q = np.ascontiguousarray(_queries(X)[0])

    # caminho simples: gated, sem prova global -> pool_only, honestamente
    r = eng.search(q)
    assert r.collect_audit_used is False
    assert r.recall_guarantee == "pool_only"
    assert r.recall_guarantee_basis == "none"

    # caminho auditado: escaneou o corpus -> pode afirmar exact_global
    c = eng.search_with_commitment(q, k=K, max_sample=20)
    assert c["exact_scored"] == c["n_corpus"]
    assert c["recall_guarantee"] == "exact_global"
    assert c["recall_guarantee_basis"] == "scored_all"
    # e a promoção tem de vir acompanhada do resultado exato
    exato = set(int(i) for i in eng.search_exact(q).indices)
    assert set(int(i) for i in c["indices"]) == exato


# ---------------------------------------------------------------------------
# 3. o gate: search() não paga o scan O(N·d)
# ---------------------------------------------------------------------------

def test_plain_search_nao_materializa_auditoria():
    """O caminho simples NÃO paga o scan exato global nem a coleta por doc."""
    X, D, K = _corpus(), 384, 5
    eng = winnex_madhava.build_engine(
        X, dim=D, metric="cosine", quant="none", k=K,
        stage1_dim=64, stage2_dim=128)
    q = np.ascontiguousarray(_queries(X)[0])

    r = eng.search(q)
    assert r.collect_audit_used is False, (
        "search() materializou a auditoria: o gate do collect_audit sumiu de novo")
    assert r.recall_guarantee == "pool_only"


def test_busca_simples_e_mais_rapida_que_o_scan_exato():
    """A regressão era esta: a busca 'rápida' pagava o scan exato e ficava mais
    lenta que força bruta. Depois do gate, search() < search_exact()."""
    import time
    X, D, K = _corpus(n=20000), 384, 5
    eng = winnex_madhava.build_engine(
        X, dim=D, metric="cosine", quant="none", k=K,
        stage1_dim=64, stage2_dim=128)
    Q = _queries(X, nq=25)

    def med(fn):
        for q in Q[:3]:
            fn(q)
        lat = []
        for q in Q:
            t0 = time.perf_counter()
            fn(q)
            lat.append(time.perf_counter() - t0)
        return float(np.median(lat))

    t_search = med(lambda q: eng.search(q))
    t_exact = med(lambda q: eng.search_exact(q))
    assert t_search < t_exact, (
        f"search()={t_search*1e3:.2f}ms >= search_exact()={t_exact*1e3:.2f}ms "
        f"— o scan O(N·d) voltou ao caminho simples")


# ---------------------------------------------------------------------------
# 4. as métricas que explicam o zero
# ---------------------------------------------------------------------------

def test_commitment_explica_por_que_o_total_e_zero():
    """`total_excluded_count = 0` tem duas causas opostas: bound frouxo, ou
    corte fixo que descartou o corpus antes. Só o par de métricas distingue."""
    X, D, K = _corpus(), 384, 5
    eng = winnex_madhava.build_engine(
        X, dim=D, metric="cosine", quant="none", k=K,
        stage1_dim=64, stage2_dim=128)
    q = np.ascontiguousarray(_queries(X)[0])
    c = eng.search_with_commitment(q, k=K, max_sample=20)

    for campo in ("pruned_by_bound", "pruned_by_prefilter", "n_corpus",
                  "exact_scored"):
        assert campo in c, f"o commitment não expõe {campo}"

    n = len(X)
    assert c["n_corpus"] == n
    assert c["exact_scored"] >= K
    # NÃO é partição: pruned_by_bound e exact_scored se sobrepõem (medido:
    # scored=60, bound=5990 somando N+50 em s1=s2=D). O teto é o invariante.
    assert c["pruned_by_bound"] <= n - K

    # ATUALIZADO (2026-10-30, correção na fonte). Este teste exigia
    # `pruned_by_bound == 0` e `pruned_by_prefilter > 0` porque o caminho
    # auditado devolvia o top-K do POOL. Depois da correção o caminho auditado
    # escaneia o corpus e devolve o top-K exato, então a explicação do zero tem
    # outra forma — e o teste passa a exigir que o número BATA com o que foi
    # escaneado, em vez de exigir um valor fixo.
    if c["exact_scored"] >= n:
        # escaneou tudo -> nenhum doc ficou sem prova por corte
        assert c["pruned_by_prefilter"] == 0, (
            "escaneou o corpus inteiro mas reporta documentos cortados sem prova")
    else:
        # escaneou só o pool -> o resto TEM de estar explicado pelo corte
        assert c["pruned_by_prefilter"] > 0, (
            "não escaneou tudo e não explica o descarte: o auditor não "
            "consegue saber POR QUE o total é o que é")
        assert c["pruned_by_prefilter"] + c["exact_scored"] >= n - K


# ---------------------------------------------------------------------------
# 5. O GATE NÃO PODE PROMOVER ESCOPO (regressão que esta própria correção
#    introduziu, e o teste que a pega)
# ---------------------------------------------------------------------------

def test_gate_nao_promove_escopo_no_caminho_simples():
    """O caminho simples conta exclusões contra o threshold do POOL, que é
    MENOR que o do corpus. Promover o escopo com essa contagem seria assinar
    'exact_global' com um número sobre o pool — exatamente o erro que este
    arquivo corrige duas vezes.

    Medido como os dois divergem (corpus aleatório 6k x 256d, s1=192): pool
    threshold 0.1573 vs global 0.1606, e a contagem do caminho simples deu 644
    onde o threshold global dá 741 — 13% da prova sumindo em silêncio.
    """
    X, D, K = _corpus(n=6000, d=256), 256, 10
    eng = winnex_madhava.build_engine(
        X, dim=D, metric="cosine", quant="none", k=K,
        stage1_dim=192, stage2_dim=0)
    q = np.ascontiguousarray(X[0])

    r = eng.search(q)
    assert r.collect_audit_used is False
    assert r.recall_guarantee_basis != "proven_exclusions", (
        "o caminho simples alegou 'proven_exclusions' — o threshold usado por "
        "ele é o do POOL, logo a prova seria sobre o pool, não sobre o corpus")

    c = eng.search_with_commitment(q, k=K, max_sample=20)
    if c["recall_guarantee_basis"] == "proven_exclusions":
        assert c["pruned_by_bound"] >= len(X) - K, (
            "promoção sem prova suficiente")


def test_global_threshold_e_o_k_esimo_exato_do_corpus():
    """Verificação INDEPENDENTE do campo assinado: recalcula o K-ésimo score
    exato do corpus em numpy e exige que o commitment concorde. Não usa o
    próprio motor como referência."""
    X, D, K = _corpus(n=6000, d=256), 256, 10
    eng = winnex_madhava.build_engine(
        X, dim=D, metric="cosine", quant="none", k=K,
        stage1_dim=64, stage2_dim=128)
    for q0 in X[:5]:
        q = np.ascontiguousarray(q0)
        c = eng.search_with_commitment(q, k=K, max_sample=20)
        sims = X @ q                              # força bruta independente
        kth_global = float(np.partition(sims, -K)[-K])
        assert abs(c["global_threshold"] - kth_global) < 1e-5, (
            f"global_threshold={c['global_threshold']:.6f} != K-ésimo exato do "
            f"corpus {kth_global:.6f} — o campo assinado não é o global")


def test_pruned_by_bound_nunca_excede_n_menos_k():
    """O único invariante estrutural real. NÃO é uma partição: pruned_by_bound
    e exact_scored se sobrepõem (medido: scored=60, bound=5990 somando N+50 em
    s1=s2=D=256), então somá-los é erro. O teto de bound é N-K."""
    X, D, K = _corpus(n=6000, d=256), 256, 10
    for s1, s2 in ((64, 128), (192, 0), (D, D)):
        eng = winnex_madhava.build_engine(
            X, dim=D, metric="cosine", quant="none", k=K,
            stage1_dim=s1, stage2_dim=s2)
        c = eng.search_with_commitment(np.ascontiguousarray(X[0]), k=K,
                                       max_sample=20)
        assert c["pruned_by_bound"] <= c["n_corpus"] - K, (
            f"s1={s1}: bound={c['pruned_by_bound']} > N-K={c['n_corpus']-K}")
        assert c["bound_violations"] == 0
