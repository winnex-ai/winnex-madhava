# Escopo do certificado e gate de auditoria — 2026-10-30

Quatro correções no caminho do commitment, cada uma com a medição que a
justifica. Todas em `tests/python/test_audit_scope_2026_10_30.py`.

---

## 1. `global_threshold` do commitment era o threshold do POOL

`search_with_commitment` usava `exact_score(out.indices[K-1], query)` — o
K-ésimo da lista **retornada**. Com `k3 < N` essa lista é o pool do
post-filter, e o K-ésimo do pool é **menor** que o K-ésimo do corpus. O campo
assinado como `global_threshold` carregava um número do pool.

É o mesmo defeito que o `search()` já havia corrigido em 2026-10-21 ("o
certificado descrevia o pool, não a busca"). O commitment não herdou a
correção, embora `search()` já calcule o valor certo em `audit_threshold`.

**Direção do erro:** threshold menor → o bound exclui **menos** → o bug
*subestimava* a prova. `bound_violations` permanecia 0. Nunca foi atestação
falsa; era um campo assinado que não significava o que o nome dizia.

**Correção:** `float worst = (float)base.audit_threshold;`

---

## 2. `exact_global` era derivado da implementação, não da prova

> ⚠️ **Esta correção precisou de um guard na segunda passada** — ver §5. A rota
> "proven_exclusions" só é sã se o threshold usado na contagem for o do corpus,
> e o caminho simples passou a usar o do pool.

```cpp
// antes
out.recall_guarantee = (pf_eff && out.k3 == N) ? "exact_global" : "pool_only";
```

`k3 == N` é uma afirmação sobre **o que a implementação fez** (o post-filter
por acaso escorou tudo?). Havia uma segunda rota, já calculada e descartada:

> se o bound **provou** `pruned_by_bound >= N-K` documentos fora do top-K,
> então sobram no máximo K sem prova — exatamente os K devolvidos — logo o
> top-K retornado **é** o global top-K, por prova.

Medido (20k × 384d, `s1 = s2 = 384`, 40 queries): recall contra FAISS
**1,0000**, e `(N - pruned_by_bound) - K` ∈ {0, …, 0, 1} — a desigualdade vale
em todas as queries. O rótulo era `pool_only` mesmo assim.

Recusar-se a afirmar uma prova que se tem não é cautela, é contrato perdido:
o comprador lê `pool_only` e conclui que o motor não sabe se acertou.

**Correção:** duas rotas, com a base declarada em `recall_guarantee_basis`
(`scored_all` | `proven_exclusions` | `none`). A rota 2 exige `pf_eff` pelo
mesmo motivo que a rota 1: sem o post-filter exato, os K devolvidos são
ordenados pelo bound e não são os K sobre os quais a prova fala. Exige também
`collect_audit` — ver §5.

---

## 3. O scan O(N·d) do threshold global rodava em TODA `search()`

O comentário no código dizia, desde 2026-10-21:

> That is O(N·d), the price of a certificate that is true about the whole
> corpus. **It runs only when a certificate is requested**
> (`search_audited` / commitment), **never on the plain `search()` path.**

Não rodava. O commit `63d1b6c` (release 1.9.15) removeu o gate ao remover a
coleta por documento, e manteve o comentário. Toda `search()` simples passou a
escorar o corpus inteiro uma segunda vez.

Medido (20k × 384d, k=10, mediana de 25 queries, mesmo processo):

| caminho | antes | depois | |
|---|---:|---:|---|
| `search()` (speed) | 1,03 ms | **0,22 ms** | **4,6× mais rápido** |
| `search_exact()` | 0,69 ms | 0,69 ms | (referência: o mesmo trabalho O(N·d), uma vez) |

Antes da correção, a busca "rápida" era **mais lenta que força bruta** — pagava
o scan exato **e** o cascade de projeção. É exatamente a degradação que o
relatório do Kaggle registrou: `busca_ms_mediana = 7,46 ms` num corpus de 20k,
com `search_exact` disponível a uma fração disso.

**Correção:** `if (out.k3 < N && collect_audit)`. O caminho simples devolve
`collect_audit_used = false`; o caminho auditado continua pagando o preço, que
é o preço que ele existe para pagar.

### Consequência de escopo — declarada, não escondida

Com o gate, os mesmos nomes de campo descrevem escopos diferentes nos dois
caminhos. `search_audited` agora devolve `audit_threshold_basis` e
`pruned_by_bound_basis` (`"global"` | `"pool"`), derivados de `k3` vs `N`:

| campo | `search()` | `search_audited` / commitment |
|---|---|---|
| `audit_threshold` | threshold do pool | threshold do corpus |
| `pruned_by_bound` | exclusões no pool | exclusões no corpus |
| `pruned_by_prefilter` | corpus | corpus |

Comparar `search_audited().audit_excluded` com `search().pruned_by_bound` é
inválido quando os escopos diferem — que passou a ser o caso normal. O teste
de consistência em `test_audit_certificate.py` foi **escopado**, não afrouxado:
a asserção de soundness (`n_viol == 0`) continua idêntica.

---

## 4. O commitment não explicava o próprio zero

`total_excluded_count = 0` tem duas causas opostas:

- o bound não provou nada (bound frouxo), ou
- o bound nem teve a chance, porque um corte fixo descartou o corpus antes.

Só o par distingue. Medido (20k × 384d, `stage1=64`, a config histórica):

```
total_excluded_count  0
pruned_by_bound       0        <- nenhuma exclusão PROVADA
pruned_by_prefilter   19.800   <- todas pelo corte fixo
n_corpus              20.000
exact_scored          200
```

**Campos novos no commitment:** `pruned_by_bound`, `pruned_by_prefilter`,
`n_corpus`, `exact_scored`. O auditor agora vê a partição inteira e sabe *por
que* o selo não carrega prova — em vez de interpretar um zero sem contexto.

---

## Os dois modos, depois das correções

Medido, d=384, k=10, mediana de 25 queries:

| n | **speed** `search()` | **audit** `search_with_commitment()` | overhead |
|---:|---:|---:|---:|
| 20 000 | **0,22 ms** | 1,01 ms | 4,5× |
| 100 000 | **1,07 ms** | 5,58 ms | 5,2× |
| 400 000 | **5,01 ms** | 21,29 ms | 4,3× |

O modo speed agora é busca vetorial de verdade em pequena/média escala (antes
era 1,03 / 5,55 / 21,46 — idêntico ao modo audit, porque os dois faziam o
mesmo trabalho). O modo audit paga ~4,5× por: threshold global exato, prova
por documento, verificação de violação e o commitment assinável.

**O que o modo audit entrega e o speed não:** escopo `exact_global` quando há
prova, `global_threshold` do corpus, `bound_violations` medido, `sampled_records`
determinístico — os campos que o Tracer assina no WORM.

**O que ainda falta para o modo audit provar exclusão em 384d:** nada no
caminho de código — falta a config. Com `stage1=64` a cobertura é 0% porque o
bound é frouxo (resíduo `e(v)=√(1−64/384)≈0,91`). Com `stage1/d ≳ 0,9` ela
passa de 94%. Para auditoria em d=384 até ~400k, `audit_exhaustive=True` é
mais barato **e** prova: `search_exact` é 0,69 ms contra 1,03 ms do cascade.

---

## Suíte

```
53 passed, 2 skipped   (tests/python)
ALL TESTS PASSED       (tests/ C++)
```

6 testes novos em `test_audit_scope_2026_10_30.py`, incluindo a contraprova
(`test_escopo_pool_only_nao_e_promovido_sem_prova`) e a regressão de
performance (`test_busca_simples_e_mais_rapida_que_o_scan_exato`).


---

## 5. O gate quase reintroduziu o bug que ele veio corrigir

Esta correção **introduziu uma regressão na primeira versão**, e ela foi
encontrada só porque o critério foi reexaminado em vez de aceito.

**O defeito.** Com o gate, o caminho simples conta `pruned_by_bound` contra o
threshold do **pool**, que é menor que o do corpus. A rota
`proven_exclusions` (correção 2) decide pela contagem — e eu havia guardado
apenas com `pf_eff && k3 == N`, sem considerar de qual threshold a contagem
veio. Numa config onde a contagem do pool cruza `N-K`, o caminho simples
assinaría `exact_global` **com um número medido sobre o pool** — exatamente o
"pool_only assinado como global" que este arquivo corrige duas vezes.

**A medição que expõe (corpus aleatório 6k × 256d):**

| config | threshold do pool | threshold do corpus | `pruned_by_bound` (plain) | (auditado) |
|---|---:|---:|---:|---:|
| s1=64, s2=128 | 0,1573 | 0,1873 | 0 | 0 |
| s1=128, s2=0 | 0,1795 | 0,1873 | 0 | 0 |
| **s1=192, s2=0** | **0,1844** | **0,1873** | **644** | **741** |

Na terceira linha o caminho simples perde 13% da prova em silêncio. Com o
threshold do pool suficientemente baixo, ele poderia também *cruzar* `N-K` e
promover indevidamente.

**A correção.** O guard passou a ser `collect_audit` — um fato **observável**
sobre qual threshold foi usado — em vez de `k3 < N`, que é um proxy
probabilístico. E `search_audited` / `search_with_commitment` passaram a
pedir `collect_audit=true` **explicitamente**, para que a testemunha não
dependa de um default.

### Uma segunda coisa que eu havia afirmado errado

Eu documentei `exact_scored + pruned_by_bound + pruned_by_prefilter` como uma
"partição" do corpus. **Não é.** Medido em `s1=s2=D=256`, onde o resíduo é 0 e
o bound é exato:

```
exact_scored 60 | pruned_by_bound 5990 | pruned_by_prefilter 0
soma = 6050  (N = 6000)
```

Um sobrevivente do pool que depois perde está nos **dois** conjuntos. O
invariante real é `pruned_by_bound <= N - K`, e `pruned_by_prefilter` é um
resto **clampado**, que pode subdeclarar. Corrigido no header e no teste.

### O que a revisão não mudou

O gate em si continua correto e medido: `search()` 1,03 → 0,22 ms, e a busca
"rápida" deixou de ser mais lenta que `search_exact`. O que faltava era o
escopo explícito e o guard.

**Suíte:** 56 passed, 2 skipped (Python) + ALL TESTS PASSED (C++).
Três testes novos cobrem especificamente o guard (§5), incluindo uma
verificação **independente** de `global_threshold` — o K-ésimo exato do corpus
recalculado em numpy, sem usar o motor como referência.

---

## 6. O gargalo de verdade: o caminho auditado devolvia o top-K do POOL

Esta é a correção que os itens 1–5 estavam orbitando sem tocar.

`search()` paga um scan exato `O(N·d)` sobre o corpus inteiro para achar o
threshold global — e **descartava o top-K que aquela mesma passada tinha
acabado de produzir**. `out.indices` continuava apontando para a lista do
post-filter pool.

Medido (20k × 384d, `s1=64/s2=128`, 40 queries):

| | antes | depois |
|---|---:|---:|
| recall do caminho AUDITADO vs top-K exato | **0,5083** | **1,0000** |
| o caminho auditado devolve o top-K exato | 1 de 8 queries | **40/40** |
| `k3` (docs escaneados pelo post-filter) | 200 | 20 000 |
| `pruned_by_prefilter` | 19 800 | **0** |
| escopo declarado | `pool_only` | `exact_global` |

Ou seja: o caminho que existe para **certificar** respondia 0,51 enquanto
segurava, na mesma passada, a prova de que a resposta era 1,0. E como o
`total_excluded_count` era contado contra um threshold que a lista devolvida
não respeitava, o certificado saía **com prova zero em 100% das consultas** —
que é exatamente o `excluidos_mediana = 0` do relatório do Kaggle.

**A correção não adiciona trabalho.** A passada passa a guardar `(score, id)` e
a usar o argsort que ela já pagava.

### Por que minha recomendação anterior estava errada

Eu havia proposto trocar o braço do benchmark para `stage1_dim=d` "para que o
certificado tenha prova". Era **mascarar**:

| caminho certificado | latência (20k, d=384, k=3) |
|---|---:|
| `search_exact` (o algoritmo exato puro) | **0,69 ms** |
| cascade `s1=64`, commit (após a correção) | 1,03 ms |
| cascade `rank_pleno`, commit | 1,85 ms |

`rank_pleno` é 80% mais lento que o cascade corrigido e converte um corpus de
20k docs de projeções em força bruta disfarçada. A cobertura de 99,98% dele é
**tautológica**: o índice já é exato, então o bound não economiza nada. O
caminho certificado nunca precisou da config cara — precisava devolver o
resultado que já calculava.

O benchmark foi revertido para a config rápida, com o registro do porquê.

### O que continua valendo da análise anterior

A cobertura de exclusão **ainda depende do bound**, e o penhasco de `s1/d`
existe: em 384d com embeddings de texto, `s1=64` dá `pruned_by_bound = 0`
mesmo escaneando o corpus. O motor agora devolve a resposta **exata** nessa
config; o que ele não consegue é **provar exclusão** com esses resíduos. São
duas garantias diferentes:

- **top-K exato**: obtido (recall 1,0, `exact_global` por `scored_all`);
- **prova de exclusão por bound**: zero em `s1=64`, ~100% em `s1≈d`.

Antes da correção, `s1=64` não dava nenhuma das duas.
