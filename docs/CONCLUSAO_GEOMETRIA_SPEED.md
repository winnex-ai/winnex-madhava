# Conclusão: substituir QKᵀ pela geometria do Madhava — o resultado definitivo

**Data:** 2026-08-07
**Status:** Investigação completa. Implementação testada e revertida (o QKᵀ permanece).
**Princípio:** o Madhava é um motor geométrico; nenhuma adaptação a dataset.

---

## O que foi implementado e testado

Substituição do QKᵀ (matmul de atenção) pela **geometria do Madhava** no speed mode:
1. Build: projeção de cada vetor em subespaço SVD (s dims) → `v_S`, resíduo `‖v_R‖`.
2. Query: projeta `q` → `q_S`, estima `⟨q,v⟩ ≈ ⟨q_S,v_S⟩`, bound `UB = est + ‖q_R‖·‖v_R‖`.
3. Seleção por bound + re-scoring exato dos candidatos.

Implementado em OpenCL nativo (kernel `build_ladder`, `geo_ub`, `topk_geo`),
compilado JIT como shaders, GPU ativa confirmada.

## Resultados medidos (C++ e Python, dados reais)

### Recall por candidatos M (bound CS, s=32)

| M candidatos | recall top-10 |
|--------------|---------------|
| 64 | 6% |
| 256 | 15% |
| 1024 | 40% |
| 2048 | 60% |

### Recall por subespaço s

| s | M=64 | M=256 |
|---|------|-------|
| 64 | 20% | 50% |
| 96 | 56% | 90% |
| 128 | 100% | 100% |

## A matemática definitiva

**O bound CS em subespaço truncado NÃO captura o top-k em d=128.** A razão é
geométrica e inescapável:
- s=32 captura ~50% da variância, resíduo da query `‖q_R‖≈0.8`.
- O bound `UB = est + ‖q_R‖·‖v_R‖` tem slack grande — o ranking por UB difere
  do ranking exato, então os top-k reais estão fora dos top-M por bound.
- Só com **s=d (subespaço completo)** o bound vira o score exato (recall 100%),
  mas aí a projeção custa `N·d` — **igual ao QKᵀ que queríamos substituir**.

## Por que o QKᵀ permanece

O QKᵀ (matmul) É a operação geométrica exata: computa `⟨q,v⟩` para todos os N.
A "substituição" por projeção truncada aproxima o mesmo produto interno, e a
aproximação perde o top-k em alta dimensão. Para recall exato, o único caminho é
o scan completo — que o QKᵀ já faz de forma otimizada na GPU.

**O QKᵀ não é "softmax"** — o softmax é a normalização opcional do resultado.
O Madhava substitui o softmax (probabilístico) pela **seleção top-k exata**:
os k maiores `⟨q,v⟩`, que é a geometria de proximidade na esfera. Essa substituição
JÁ está implementada (o kernel `topk` seleciona os k maiores scores).

## Veredito

A implementação da geometria do Madhava (projeção + bound CS + re-scoring) foi
correta em estrutura e ativou a GPU, mas **não pode substituir o QKᵀ sem perder
recall** — a maldição da dimensionalidade torna o bound solto demais em d=128.

O gargalo real do speed GPU permanece o **kernel topk OpenCL serial** (insertion
sort, 1 work-group/query). A otimização legítima é paralelizar o scan (M
work-groups + merge), mantendo o QKᵀ exato como base — sem trocar a geometria
exata por uma aproximação que perde recall.
