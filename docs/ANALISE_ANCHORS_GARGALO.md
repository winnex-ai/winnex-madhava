# Análise cética: bound Cauchy-Schwarz, leis de âncoras e o gargalo de latência

**Data:** 2026-08-07
**Status:** Documentação de investigação — NENHUMA mudança no core foi feita com base nisso
**Princípio:** o Madhava opera em qualquer dataset; nada aqui adapta o core a um dataset específico.

---

## 1. O que foi testado

Um teste local (`tests/test_anchor_law.py`) compara empiricamente **quatro leis de raios**
para âncoras do bound Cauchy-Schwarz no speed mode:

| Lei | Fórmula | Propriedade geométrica |
|-----|---------|------------------------|
| `π·p` (atual) | `raios = π × primos` | **artefato** — não é espiral, gaps erráticos |
| Logarítmica (auto-similar) | `r = a·e^(bθ)` → `r_k = R·λ^k` | razão constante, self-similar |
| Arquimedes | `r = a + b·k` | espaçamento uniforme |
| Fermat | `r² = a²θ` | áreas iguais por volta |
| Hiperbólica | `r = a/θ` | assintótica |

**Hipótese original:** a progressão geométrica (logarítmica discreta) produziria um
bound CS mais apertado (menor `||v_R||`) que o `π·p` atual.

## 2. Resultado — a hipótese foi REFUTADA

**Todas as quatro leis dão resultado IDÊNTICO.** O resíduo `||v_R||` médio é o mesmo
(0.9730 em d=128) para `π·p`, geométrica, arquimedes e multiescala.

**Por quê:** o resíduo do bound CS depende **apenas do subespaço `span(A)`** das
direções das âncoras, **não da escala dos raios**. `π·p` vs `R·λ^k` vs `a+bk` —
se as direções são as mesmas, `||v_R||` é o mesmo.

## 3. O que de fato aperta o bound (a tabela decisiva)

Projeção no subespaço total das K âncoras (d=64, esfera unitária):

| K (âncoras) | `\|\|v_R\|\|` | poda (res < 0.5) |
|-------------|----------------|-------------------|
| 8  (atual)  | 0.93           | 0.0%              |
| 16           | 0.86           | 0.3%              |
| 32           | 0.70           | 50.2%             |
| 48           | 0.49           | 57.4%             |
| 64           | 0.00           | 100%              |

**Conclusão:** o bound CS só aperta com **mais direções (K)**, não com outra lei de
raios. Mas o custo de projetar em K âncoras é `K·d` FMA/vetor — quando K é grande o
suficiente para podar, o custo da projeção **se aproxima do custo do IP exato** (`d`
FMA/vetor). **O bound CS não paga no regime onde seria útil.**

## 4. O gargalo real (medido)

- GPU OpenCL QKᵀ: 74ms para batch 1000, N=100k, d=128 (1.77× vs CPU)
- O QKᵀ **teórico** é ~0.6ms (12.8 GFLOP a 20 TFLOPS)
- O gargalo é o **topk + copy + overhead de enqueue**, não o matmul

⇒ Trocar a lei de raios das âncoras **não** ataca o gargalo. O ganho real viria de
reduzir o custo do topk/copy pós-QKᵀ, ou de podar células inteiras **antes** do QKᵀ.

## 5. Análise cética da base matemática (QRH / ΨQRH / EinOps)

Analisados com ceticismo: `zenodo.org/records/17171112` e
`github.com/klenioaraujo/-QRH-EinOps`. Foco: a matemática aplicada que poderia
resolver o gargalo.

### 5.1 O einsum 'bhd,bhd->bh' NÃO é atenção (verificado empiricamente)

O código `PSIQRH_PRODUCTION_GRADE.py` (linhas 226-290) computa:
```
q,k,v: [B, T, H, hd]
q_fft = fft(q, dim=1); ...  # O(T log T)
filter = exp(iα·atan(log|freq|))
q_f = q_fft * filter         # O(T)
q_f = q_f.real               # ← descarta o imaginário
attn = einsum('bhd,bhd->bh', q_f, k_f)  # O(T·H·hd)
```

- O einsum `'bhd,bhd->bh'` computa `⟨q_i, k_i⟩` para a **mesma posição i** —
  um produto pontual no tempo, **sem Σ sobre posições**. **Não é atenção.**
- Cada token fica independente; não há mistura entre posições.
- O `.real()` descarta a fase — anulando o "quaterniônico"/SO(4).
- O custo é `O(T·d)` linear, não `O(n log n)` (o FFT existe mas o produto domina).

### 5.2 Irrelevante para o Madhava

O Madhava precisa de `QKᵀ` (matmul) + topk sobre **N vetores**. FFT sobre posições
de uma sequência não é a operação de busca vetorial. Não há como extrair um bound,
poda, ou aceleração de topk disso.

### 5.3 Os "ganhos" reportados

- "2.1× inferência" / "O(n log n)" — contraditos pela análise de custo acima.
- "Energy conservation" — é `norm(x)/norm(output)` normalizado, trivial.
- "Lech lattice" — usa D₂₄, não o Λ₂₄ real (admitido no próprio repo).

## 6. Veredito

**Nenhuma das bases matemáticas analisadas (QRH, ΨQRH, EinOps) fornece uma técnica
para resolver o gargalo de latência do Madhava.** O caminho tecnicamente correto
permanece: atacar o custo do topk/copy pós-QKᵀ, ou poda de células antes do matmul
com bound CS *real* (subespaço de alta dimensão), não com leis de raios cosméticas.

**O teste `test_anchor_law.py` permanece no repo como evidência da refutação** —
o resultado honesto (todas as leis = mesmo resíduo) é mais valioso que a hipótese.
