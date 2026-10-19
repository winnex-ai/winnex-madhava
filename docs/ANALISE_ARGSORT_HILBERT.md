# Análise: recriação do argsort em estrutura de Hilbert (refutada)

**Data:** 2026-08-07
**Status:** Refutação empírica — o teste local (`tests/test_argsort_hilbert.cpp`) prova
que a estrutura de Hilbert não acelera o argsort/top-k do Madhava.
**Princípio:** o Madhava é um motor genérico; nada aqui adapta o core a um dataset.

---

## O que foi testado

Recriar o argsort do speed mode usando a estrutura de espaço de Hilbert do Madhava:

- **Estados**: `|v_i⟩ ∈ ℋ`, normalizados (`‖v‖ = 1`), como embeddings.
- **Produto interno hermitiano**: `⟨a|b⟩`.
- **Medição projetiva**: a query `|q⟩` define um observável; amplitude `a_i = ⟨q|v_i⟩`
  (regra de Born: `p_i = |a_i|²`) → ranking por amplitude.
- **Decomposição espectral**: `|q⟩ = Σ_j β_j |e_j⟩` nos autoestados do operador de
  correlação, com medição coarse→fine e bound espectral de poda.

## Resultado 1 — medição projetiva truncada DIVERGE

| J (autoestados) | variância capturada | recall top-10 no coarse |
|-----------------|---------------------|--------------------------|
| 16 | 32% | **0-10%** |
| 32 | 46% | 10-20% |
| 64 | 67% | 30-90% (só com M=200) |

Para recall 100%, precisaria de J≈128 (subespaço completo) — mas aí o custo
`N·J + M·d` **supera** o argsort exato `N·d`. A medição projetiva truncada é
**aproximada** e perde o top-k em alta dimensão.

## Resultado 2 — o bound exato de Hilbert (com garantia) é fraco

```
J=16: scan 0.4% | recall 20%  | q_R=0.89
J=32: scan 0.4% | recall 20%  | q_R=0.80
J=64: scan 0.4% | recall 90%  | q_R=0.67
```

O bound `UB(v) = coarse + ‖q_R‖·‖v_R‖` precisa de J=64 + escanear 200 candidatos
para 90% recall. O resíduo `‖q_R‖≈0.7-0.9` é grande porque os embeddings preenchem
a esfera em d=128 — nenhum subespaço truncado captura a ordem.

## A matemática da refutação

A regra de Born `p_i = |⟨q|v_i⟩|²` **requer todos os** `⟨q|v_i⟩` para achar o top-k.
Nenhuma decomposição espectral elimina esse custo: os autoestados capturam a
*variância* (estrutura global), mas o *ranking* é uma propriedade de alta dimensão
que só emerge com todas as amplitudes.

**Isso vale para qualquer transformada** (Fourier, wavelet, quaterniônica): o top-k é
uma ordenação dos scores exatos, não uma propriedade espectral. A "estrutura oculta"
que o argsort precisa é a própria ordem — que já está nos scores.

## O que o teste prova (o caminho correto)

O argsort do Madhava é um problema de **seleção parcial**, resolvido por:
- **CPU**: `partial_sort` (heap, `O(N log k)`) ou `nth_element` (quickselect, `O(N)`)
- **GPU**: divide-and-conquer paralelo (M work-groups por query + merge) — o ganho
  vem do **paralelismo no scan**, não de transformada geométrica/espectral.

O arquivo `tests/test_argsort_hilbert.cpp` permanece como evidência da refutação.
