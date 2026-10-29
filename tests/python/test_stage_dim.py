"""
Regressão: stage1_dim/stage2_dim versus dim (2026-10-29).

Dois defeitos distintos foram encontrados e corrigidos no mesmo dia. Estes
testes existem para que nenhum dos dois volte — inclusive pela mão de quem os
corrigiu.

DEFEITO 1 — stage1_dim > dim invalida o bound (real, corrigido com clamp).
  O bound de Cauchy-Schwarz exige P com linhas ortonormais. Em R^d existem no
  máximo d; rank maior força o MGS a produzir linhas dependentes, o loop do
  build soma pn1 = sum_j <v,P_j>^2 sobre todas as linhas, e(v) fica
  subestimado e UB superestimado. Medido (d=64, audit_exhaustive, 512 docs):
      s1=32 -> viol=0, UB/exato ~ -      (válido)
      s1=64 -> viol=0, UB/exato = 1.002
      s1=128 -> viol=269, UB/exato = 2.008   <- quebrado
  O sintoma é silencioso: os ids continuam corretos, só o certificado mente.

DEFEITO 2 — o clamp de stage2_dim DEGRADAVA o cascade (introduzido e corrigido).
  A primeira versão do fix clampava os dois parâmetros. Com d=64 e os DEFAULTS
  (s1=64, s2=128), clampar s2 para 64 o deixava igual a s1 e matava a segunda
  etapa. Medido no 1.9.20 original: s2=128 > d=64 dá 0 violações e escopo
  exact_global — o build já trata o excesso de stage2. Clampar era "corrigir"
  um caso que não estava quebrado.

Run:  python -m pytest tests/python/test_stage_dim.py -v
"""
import numpy as np
import pytest

import winnex_madhava


def _corpus(d, n=512, seed=7):
    rng = np.random.default_rng(seed)
    k = 32
    centers = rng.normal(size=(k, d))
    X = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        c = centers[i % k]
        X[i] = c + 0.12 * np.linalg.norm(c) / np.sqrt(d) * rng.normal(size=d)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Defeito 2: o cascade NÃO pode degenerar nos defaults
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [32, 64, 96, 127, 128, 256])
def test_defaults_nao_degeneram_o_cascade(d):
    """Com os DEFAULTS do Config (s1=64, s2=128), o cascade precisa sobreviver.

    Este é o teste que pega a regressão que eu mesmo introduzi: clampar
    stage2_dim para d quando d < 128 faz s2 == s1 e a segunda etapa some.
    Para d < 128 o valor correto mantém s2 > s1 — o build trata o excesso.
    """
    cfg = winnex_madhava.Config()
    cfg.dim = d
    eng = winnex_madhava.MadhavaL2(cfg)
    got = eng.config()
    assert got.stage1_dim <= d, (
        f"d={d}: stage1_dim={got.stage1_dim} > d — o bound seria inválido")
    assert got.stage2_dim > got.stage1_dim or got.stage2_dim == 0, (
        f"d={d}: stage2_dim={got.stage2_dim} <= stage1_dim={got.stage1_dim} — "
        f"o cascade degenerou. O clamp de stage2_dim em d é a causa: para "
        f"d < 128 ele colapsa a segunda etapa. stage2 > d é ACEITÁVEL e o "
        f"build trata (medido: 0 violações).")


@pytest.mark.parametrize("d", [64, 128])
def test_stage2_acima_de_dim_continua_valido(d):
    """stage2_dim > dim NÃO quebra o bound — não deve ser clampado.

    Medido no 1.9.20 original com d=64 e s2=128: 0 violações, exact_global.
    Se este teste falhar com violações, então stage2 > dim passou a ser
    problema e o clamp volta a fazer sentido — mas só depois de medir.
    """
    X = _corpus(d)
    q = np.ascontiguousarray(X[0])
    eng = winnex_madhava.build_engine(
        X, dim=d, k=1, quant="none", stage1_dim=32, stage2_dim=128,
        audit_exhaustive=True)
    r = eng.search_with_commitment(q, k=1, max_sample=5)
    assert r["bound_violations"] == 0, (
        f"d={d} com stage2=128 (>{d}): {r['bound_violations']} violações — "
        f"stage2 > dim passou a invalidar o bound")


# ---------------------------------------------------------------------------
# Defeito 1: stage1_dim > dim precisa ser clampado (e o bound, válido)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d,s1_pedido", [(64, 128), (64, 512), (128, 256)])
def test_stage1_acima_de_dim_e_clampado_e_bound_valido(d, s1_pedido):
    """Pedir stage1 > d é clampado em d, e o resultado tem bound válido.

    Sem o clamp, mediu-se UB/exato = 2.008 e 269 violações em d=64/s1=128.
    Com o clamp, o bound é válido e o certificado não mente.
    """
    X = _corpus(d)
    q = np.ascontiguousarray(X[0])
    eng = winnex_madhava.build_engine(
        X, dim=d, k=1, quant="none", stage1_dim=s1_pedido, stage2_dim=0,
        audit_exhaustive=True)
    got = eng.config()
    assert got.stage1_dim <= d, (
        f"d={d}: stage1_dim pedido {s1_pedido} não foi clampado "
        f"(ficou {got.stage1_dim})")
    r = eng.search_with_commitment(q, k=1, max_sample=5)
    assert r["bound_violations"] == 0, (
        f"d={d}, stage1 pedido {s1_pedido}: {r['bound_violations']} violações "
        f"— o bound está inválido")


def test_stage1_em_dim_nao_e_alterado():
    """Controle: stage1 == d é válido e não sofre clamp."""
    X = _corpus(64)
    eng = winnex_madhava.build_engine(
        X, dim=64, k=1, quant="none", stage1_dim=64, stage2_dim=0)
    assert eng.config().stage1_dim == 64
