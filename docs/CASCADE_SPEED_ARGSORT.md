# Cascade B1→B2 como argsort do speed mode — análise e resultado

**Data:** 2026-08-07
**Status:** Análise conclusiva. Testes em `tests/test_cascade_argsort.cpp` e
`tests/test_topk_parallel.cpp`. NENHUMA mudança no core sem aprovação.

---

## A proposta (Cauchy-Schwarz + Speed mode, alta dimensão)

Integrar o cascade stage1→stage2 do Madhava nativo ao speed mode:

- **Stage-1 (s1 dims):** bound B1 = `⟨v_S1,q_S1⟩ + ‖v_R1‖·‖q_R1‖` (barato)
- **Stage-2 (s2 = d):** bound B2 = `⟨v,q⟩` **exato** (subespaço completo ⇒ `‖v_R2‖=0`)
- **Early-exit:** pontuar exato em ordem de B1, parar quando o próximo B2 ≤ kth.

**Exatidão:** com `s2 = d`, `span(P2) = ℝ^d`, logo `B2(v) = ⟨v,q⟩` exatamente
(verificado: `max|B2 − exato| = 7.6e-16`). O early-exit por B2 é matematicamente
exato: para quando o próximo vetor tem score ≤ kth, provadamente fora do top-k.

## Resultado medido

| s1 | corr(B1, exato) | recall top-200 | recall top-1000 | scan até recall 100% |
|----|-----------------|----------------|-----------------|----------------------|
| 16 | 0.47 | 20% | 20% | 100% (early-exit não dispara cedo) |
| 32 | 0.62 | 30% | 70% | ~1000 (1%) |
| 64 | 0.76 | 90% | 100% | ~200-1000 |
| 96 | 0.86 | 90% | 100% | ~200 |
| 128| 1.00 | 100% | 100% | ~10 (só o top-k) |

**Condição para acelerar sem perder garantias:** B1 deve ordenar bem (corr → 1),
o que exige s1 alto (64-128). Nesse regime, o early-exit é exato e examina quase
só o top-k.

## O trade-off que define tudo

`s1 alto ⇒ B1 ≈ exato ⇒ early-exit exato e curto`, MAS a projeção custa `s1·d`
por vetor — que com s1 ≈ d se **aproxima do custo do QKᵀ completo** (`N·d`).

- **GPU:** o QKᵀ (matmul) já é o método exato mais rápido. O cascade adiciona
  overhead sem ganho — o matmul completo já resolve.
- **CPU:** o cascade B1 em s1 moderado (32-64) + early-exit exato dá ganho real
  quando o matmul completo é caro.

## Veredito

O cascade B1→B2 é matematicamente **correto** (exato por construção quando
`s2 = d`) mas o ganho depende de B1 ordenar bem, que exige s1 alto — onde o
custo da projeção se aproxima do scan exato. O speed mode com QKᵀ (matmul GPU)
**já é o método exato mais rápido**; o cascade é o fallback correto para CPU.

**O gargalo real do speed mode GPU** (medido: 74ms batch 1000, N=100k) está no
**kernel topk OpenCL serial** (insertion sort, 1 work-group/query). A otimização
que resolve é **paralelizar o scan** (M work-groups + merge), não o bound CS.

## Próximo passo recomendado

O teste `tests/test_topk_parallel.cpp` mostrou que o divide-and-conquer 2-kernels
foi mais lento (overhead de 2 enqueues). A iteração correta: **1 kernel só**, com
mais work-groups por query e merge intra-kernel — o caminho para os 74ms.
