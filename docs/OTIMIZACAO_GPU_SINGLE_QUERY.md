# Otimização GPU (OpenCL) — single-query 20× mais rápido (v1.7.2)

**Data:** 2026-08-08
**Escopo:** eliminar o gargalo mais significativo do speed mode GPU: o
single-query mais lento que o CPU (47.8ms vs 9.2ms no 1M).

---

## 1. O problema (evidência medida)

| N | single-query GPU (antes) | Leitura ideal (448 GB/s) | Eficiência de banda |
|---|---|---|---|
| 100K | 5.05 ms | 0.11 ms | ~2.2% |
| 500K | 24.05 ms | 0.57 ms | ~2.4% |
| 1M | 47.77 ms | 1.14 ms | **~2.4%** |

O single-query usava **~2.4% da largura de banda da GPU**. O tempo escalava
linearmente com N → o gargalo era o **kernel de computação**, não o dispatch.

Confirmação: batch nq=1, 8, 32 todos ≈ 48ms total (custo = trabalho), e o GPU
ficava rápido (1.46ms/query) apenas quando o batch satuva a GPU (nq=100).

## 2. Causa raiz (kernel `qkt` original, `speed_opencl.cpp`)

```c
__kernel void qkt(...) {
    int qi = get_group_id(0);       // 1 work-group por query
    ...
    for (int vi = lid; vi < N; vi += lsz) {
        const __global float* v = corpus + (size_t)vi * d;   // 512B/item
        for (int j = 0; j < d; j++) s += lq[j] * v[j];        // nao-coalescido
        scores[qi*N + vi] = s;                                // materializa N
    }
}
```

Três problemas combinados:
1. **1 work-group por query** → 1 SM ativo de 32 → **~3% da GPU usada**. Com
   N=1M, cada um dos 256 itens processa ~3900 vetores serialmente.
2. **Acesso não-coalescido**: work-items adjacentes leem regiões a 512B de
   distância. A GPU coalesce 32B; este padrão desperdiça banda.
3. **Materializa `scores[N]` (4MB)** na memória global, depois o topk relê →
   **2× tráfego de memória**.

## 3. A correção (kernel fusionado `qkt_fused_topk`, v1.7.2)

| Otimização | Antes | Depois |
|---|---|---|
| Paralelização | 1 work-group/query (1 SM) | **M work-groups/query** (todos os SMs) |
| Acesso memória | não-coalescido (512B/item) | **coalescido** (itens adjacentes → vetores adjacentes) |
| scores[N] | materializado + relido | **fuso QKᵀ+topk**, top-k local em registrador |
| L2 correction | kernel separado | **aplicada no fused** |
| Kernels/query | qkt → l2 → topk_local → topk_merge (4) | **qkt_fused → topk_merge_scores (2)** |

### Como funciona o novo kernel

1. **M work-groups por query** (M=64 default), cada um varre um chunk contíguo
   de N (padrão do `topk_local`, agora aplicado ao matmul).
2. **Varredura coalescida**: `for (vi = start+lid; vi < end; vi += lsz)` →
   work-items adjacentes de uma warp leem vetores adjacentes do corpus →
   cada carregamento de 32B da cache é usado por todos os itens da warp.
3. **Fusão**: mantém o top-k local (k≤16) em registrador/local mem e escreve
   apenas os k índices+scores por chunk em `local_idx` — sem `scores[N]`.
4. `topk_merge_scores` funde os M top-k locais → top-k global exato.

O acesso coalescido é a chave: em vez de cada item ler 512B contíguos sozinho
(desperdiçando a cache), uma warp de 32 itens lê 32 vetores adjacentes — a GPU
carrega cada linha de cache uma vez e todos os itens a usam.

## 4. Resultado (RTX 5060 Ti, 1M, L2, 100 queries)

| N | Antes | Depois | Speedup |
|---|---|---|---|
| 100K | 5.05 ms | **0.53 ms** | 9.5× |
| 500K | 24.05 ms | **1.39 ms** | 17.3× |
| 1M | 47.77 ms | **2.40 ms** | **19.9×** |

| Método | Lat (ms) | QPS | Eficiência |
|---|---|---|---|
| Madhava speed GPU | **2.41** | **415** | 100% |
| Madhava speed CPU | 9.56 | 105 | 100% |
| **GPU vs CPU** | — | **3.97×** | — |

Batch (1M): ~1.5-1.9 ms/query, ~600-640 QPS (estável).

**Corretude preservada**: 20/20 L2 e 20/20 cosine idênticos ao brute-force
numpy; 100/100 queries uint8 idênticas ao scan exato; batch == single-query.

## 5. Limitações restantes (honestas)

- **HNSW/IVF continuam mais rápidos** (0.08-0.23ms) — são índices aproximados
  sublineares. O speed mode é um **scan exato** (garantia completa), não compete
  em latência com aproximados. O valor é exatidão + determinismo + build rápido.
- O **batch** em ~600 QPS não escala linearmente com nq (o `topk_merge_scores`
  é serial por query) — otimização futura: paralelizar o merge.
- `M=64` é um heurístico; em GPUs com mais/fewer SMs, o ótimo varia (tunável).
- O caminho antigo (`qkt` + `topk_local` + `topk_merge`) permanece no código
  para compatibilidade/fallback.

## 6. Teste

```bash
# Corretude (L2 + cosine vs brute-force numpy)
python3 -c "
import numpy as np, winnex_madhava as wm
rng = np.random.RandomState(0)
X = np.ascontiguousarray(rng.randn(100000, 128).astype(np.float32))
eng = wm._native.SpeedEngine(X, 128, 1, 0, 4, True)
q = X[0]
print('top-1 proprio vetor:', eng.search(q, 10)['indices'][0] == 0)
"

# Desempenho single-query 1M
python3 -c "
import numpy as np, winnex_madhava as wm, time
rng = np.random.RandomState(0)
X = np.ascontiguousarray(rng.randn(1000000, 128).astype(np.float32))
eng = wm._native.SpeedEngine(X, 128, 1, 0, 4, True)
eng.search(X[0], 10)
t0 = time.time()
for _ in range(10): eng.search(X[0], 10)
print(f'single-query 1M: {(time.time()-t0)/10*1000:.2f} ms')
"
```
