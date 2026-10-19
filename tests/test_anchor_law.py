#!/usr/bin/env python3
"""
Teste local — hierarquia auto-similar de âncoras para o bound Cauchy-Schwarz.

Hipótese (análise geométrica):
  O código usa "π·p" (primos × π) como escalas das âncoras, que NÃO é nenhuma
  das quatro espirais polares (logarítmica r=ae^bθ, Fermat r²=a²θ, Arquimedes
  r=a+bθ, hiperbólica r=a/θ). A única auto-similar é a logarítmica, que em
  âncoras discretas = progressão geométrica r_k = R·λ^k.

  Este teste mede EMPIRICAMENTE qual arquitetura de âncoras produz o bound CS
  mais apertado (menor ||v_R|| = menor resíduo) e a maior fração podável.

Metodologia:
  - Corpus sintético na esfera unitária (d-dim), como embeddings reais.
  - Para cada "lei de âncoras", construir K âncoras ortonormais.
  - Medir, para cada query, o resíduo ||v_R|| = ||v - proj_S(v)|| médio sobre o
    corpus. Quanto menor, mais apertado o bound → mais poda antes do QKᵀ.

Leis testadas:
  1. pi-primes : raios = π·p (artefato atual, sem lei angular)
  2. geometric : raios = R·λ^k (progressão geométrica — logarítmica discreta)
  3. multiscale: âncoras com DIREÇÕES VARIADAS por escala (a hipótese)
  4. archimedes: raios = a + b·k (espaçamento uniforme)
"""
import numpy as np
import math

def make_corpus(N, d, seed=42, n_clusters=10):
    """Embeddings sintéticos na esfera unitária com estrutura de clusters."""
    rng = np.random.RandomState(seed)
    # centros de cluster (direções)
    centers = rng.randn(n_clusters, d).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = np.zeros((N, d), dtype=np.float32)
    for i in range(N):
        c = centers[i % n_clusters]
        noise = rng.randn(d).astype(np.float32) * 0.3
        v = c + noise
        n = np.linalg.norm(v)
        X[i] = v / n if n > 1e-9 else c
    return X

def orthonormalize(dirs):
    """Gram-Schmidt sobre um conjunto de direções."""
    A = []
    for v in dirs:
        v = v.copy()
        for a in A:
            v -= np.dot(v, a) * a
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            A.append(v / nv)
        else:
            # direção aleatória ortogonal
            r = np.random.RandomState(len(A)).randn(v.size).astype(np.float32)
            for a in A:
                r -= np.dot(r, a) * a
            r /= np.linalg.norm(r)
            A.append(r)
    return np.array(A)

def build_anchors(X, K, law, d, seed=42):
    """
    Constrói K âncoras ortonormais segundo uma lei de raios.
    As DIREÇÕES vêm de SVD (eixos principais) — mesmas direções para todas as
    leis — exceto 'multiscale', que varia direções por escala.
    """
    rng = np.random.RandomState(seed)
    # direções base: top-K eixos SVD da amostra
    sample = X[:min(5000, len(X))]
    _, s, Vt = np.linalg.svd(sample - sample.mean(0), full_matrices=False)
    dirs = Vt[:K].astype(np.float32)
    A = orthonormalize(dirs)

    # raios pela lei
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    if law == 'pi':
        rad = np.array([math.pi * p for p in primes[:K]], dtype=np.float32)
    elif law == 'geometric':
        lam = 1.5
        rad = np.array([lam ** k for k in range(K)], dtype=np.float32)
    elif law == 'archimedes':
        rad = np.array([1.0 + 1.0 * k for k in range(K)], dtype=np.float32)
    elif law == 'multiscale':
        # DIREÇÕES VARIADAS por escala: cada escala usa um bloco de direções
        # distintas (eixos SVD + perturbados), não o mesmo eixo escalado.
        A_all = []
        scale_dirs = K // 4 + 1
        for s in range(4):  # 4 escalas
            base = Vt[s * scale_dirs:(s + 1) * scale_dirs].astype(np.float32)
            if len(base) == 0:
                base = Vt[:scale_dirs].astype(np.float32)
            A_all.extend(orthonormalize(base))
        A = orthonormalize(np.array(A_all[:K]))
        lam = 1.5
        rad = np.array([lam ** k for k in range(K)], dtype=np.float32)
    else:
        raise ValueError(law)
    return A, rad

def measure_residual(X, A, n_anch_active=8):
    """
    Mede o resíduo ||v_R|| médio quando projetamos v no subespaço das âncoras.
    Para um bound CS, o resíduo da projeção na MELHOR âncora (mais alinhada)
    determina o aperto: res = sqrt(1 - max_a <v,a>²).
    """
    proj = X @ A.T  # (N, K) — similaridade com cada âncora
    best = np.max(np.abs(proj), axis=1)
    res = np.sqrt(np.clip(1.0 - best ** 2, 0, 1))
    return float(res.mean()), float(np.mean(res < 0.5)), float(np.mean(res < 0.3))

def main():
    print("=" * 78)
    print("TESTE LOCAL — hierarquia de âncoras para o bound Cauchy-Schwarz")
    print("=" * 78)
    print("Hipótese: π·p não é espiral; a auto-similar é r=R·λ^k (logarítmica).")
    print()

    for N, d in [(20000, 64), (50000, 128)]:
        print(f"\n{'='*78}")
        print(f"CORPUS: N={N}, d={d} (esfera unitária, 10 clusters)")
        print(f"{'='*78}")
        X = make_corpus(N, d)
        print(f"\n  Lei de âncoras | ||v_R|| médio | poda 50% | poda 30% | cobertura")
        print(f"  {'-'*80}")
        results = []
        for law in ['pi', 'geometric', 'archimedes', 'multiscale']:
            A, rad = build_anchors(X, 8, law, d)
            avg_res, frac50, frac30 = measure_residual(X, A)
            # cobertura = quão bem as K âncoras capturam a variância
            cov = float(np.mean(np.sum((X @ A.T) ** 2, axis=1)))
            results.append((law, avg_res, frac50, frac30, cov))
            print(f"  {law:>12} | {avg_res:.4f}       | {frac50*100:5.1f}% | {frac30*100:5.1f}% | {cov:.3f}")

        best = min(results, key=lambda x: x[1])
        print(f"\n  MELHOR (menor resíduo): {best[0]} — ||v_R||={best[1]:.4f}")
        print(f"  Melhor cobertura: {max(results, key=lambda x: x[4])[0]} — var capturada={max(r[4] for r in results):.3f}")

    print("\n" + "=" * 78)
    print("INTERPRETAÇÃO")
    print("=" * 78)
    print("""
    - ||v_R|| médio: o resíduo após projetar na melhor âncora. Quanto menor,
      mais apertado o bound CS → mais vetores podados antes do QKᵀ.
    - 'poda 50%': fração de vetores com ||v_R|| < 0.5 (bound potencialmente
      forte o suficiente para descartar muitos).
    - 'cobertura': variância capturada pelas K âncoras (⟨v,a⟩² somado).

    Se a progressão geométrica / multiescala reduzir ||v_R|| vs π·p, isso
    confirma que a lei auto-similar é a arquitetura correta para o bound.
    Se for igual, o resíduo é dominado pelas DIREÇÕES, não pelos raios —
    e o ganho precisa vir de mais âncoras, não de outra lei de raios.
    """)

if __name__ == '__main__':
    main()
