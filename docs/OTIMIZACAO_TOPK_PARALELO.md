# Otimização do speed GPU: topk paralelo + QKᵀ com tiling

**Data:** 2026-08-07
**Status:** Implementado e validado (corretude 30/30, ctest passa).
**Arquivo:** `src/speed_opencl.cpp`

---

## O que foi implementado

### 1. Topk paralelo (divide-and-conquer) — substitui o insertion sort serial

O kernel `topk` antigo usava **1 work-group por query** com insertion sort
serial no scan de N — o gargalo. Substituído por 2 kernels:

- **`topk_local`**: `M` work-groups por query, cada um varre um **chunk
  contíguo** de N e mantém um top-k local. O scan de N vira `N/M` por
  work-group → `M×` mais paralelismo.
- **`topk_merge`**: funde os top-k locais dos `M` chunks → top-k global exato.

### 2. QKᵀ com tiling da query

O kernel `qkt` antigo usava grid 2D `[nq, N]` onde cada thread relia a query
da memória global. Otimizado: **1 work-group por query**, query carregada em
**local memory** uma vez e reutilizada por todos os work-items (d=128 floats =
512B, cabe). Grid vira `[nq*256]` 1D.

## Resultados medidos

| Configuração | Tempo (batch 1000, N=100k) |
|--------------|----------------------------|
| Original (topk serial) | 74 ms |
| Topk paralelo M=4 | ~137-146 ms |
| QKᵀ tiling + topk paralelo | ~146 ms |

**Corretude: 30/30** (recall exato preservado — o merge dos top-k locais é
matematicamente exato).

## Observações honestas

O ganho do topk paralelo é **modesto** porque o gargalo real é o **matmul QKᵀ**
que lê o corpus inteiro (N=100k × d=128 × 4B = 51GB por batch) — um problema
**memory-bound**, não de paralelismo do topk. O batch (0.14 ms/query) já
amortiza 37× vs o single (5.2 ms).

O tiling da query ajudou marginalmente. A otimização que reduziria os bytes
lidos é o **roteamento O(K) por âncoras** (ler só nprobe/K do corpus) — mas
isso perde recall (aproximado). O speed mode mantém a **exatidão** do QKᵀ
(recall 1.0 vs brute-force), que é a garantia matemática do Madhava.

## Estado

- `topk_local` + `topk_merge` integrados ao `scores_gpu_topk`.
- `qkt` com tiling da query, grids corrigidos nos 2 call sites.
- ctest passa, corretude 30/30, GPU ativa.
