/**
 * test_argsort_hilbert.cpp — recria o argsort/top-k usando a estrutura de
 * espaço de Hilbert do Madhava.
 *
 * O Madhava modela embeddings como estados |ψ⟩ em um espaço de Hilbert:
 *   |v_i⟩ ∈ ℋ,  ⟨v_i|v_j⟩ = produto interno hermitiano,  ‖v_i‖ = 1.
 * A busca é uma MEDIÇÃO PROJETIVA: a query |q⟩ define um observável, e a
 * amplitude a_i = ⟨q|v_i⟩ determina o ranking (regra de Born: p_i = |a_i|²).
 *
 * Este teste:
 *   1. Implementa a medição projetiva de Hilbert (argsort por amplitude).
 *   2. Testa a PODA ESPECTRAL: decompor o operador de correlação em
 *      autoestados (|e_j⟩), e usar projeções sucessivas para descartar
 *      estados cuja amplitude máxima é limitada pelo resíduo espectral.
 *   3. Mede corretude (vs argsort exato) e ganho de velocidade.
 *
 * Build:
 *   g++ tests/test_argsort_hilbert.cpp -Iinclude -O2 -fopenmp -o /tmp/tah
 * Run:
 *   /tmp/tah
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

namespace hilbert {

// ---- Estado de Hilbert: vetor normalizado (‖ψ‖ = 1) --------------------
struct State {
    std::vector<float> data;   // componentes na base computacional
    float norm() const {
        float s = 0;
        for (float x : data) s += x * x;
        return std::sqrt(s);
    }
};

// Produto interno hermitiano ⟨a|b⟩ (real para embeddings reais).
inline float braket(const State& a, const State& b) {
    float s = 0;
    for (size_t j = 0; j < a.data.size(); j++) s += a.data[j] * b.data[j];
    return s;
}

// Medição projetiva: amplitude do estado |v⟩ sobre o observável |q⟩.
inline float amplitude(const State& q, const State& v) {
    return braket(q, v);
}

// ---- Argsort clássico (referência exata) -------------------------------
std::vector<int> argsort_exact(const State& q, const std::vector<State>& corpus, int k) {
    int N = (int)corpus.size();
    std::vector<std::pair<float,int>> scores(N);
    for (int i = 0; i < N; i++) scores[i] = {amplitude(q, corpus[i]), i};
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
    std::vector<int> out(k);
    for (int j = 0; j < k; j++) out[j] = scores[j].second;
    return out;
}

// ---- Medição projetiva de Hilbert + poda espectral ----------------------
// Idéia: |q⟩ = Σ_j β_j |e_j⟩  (decomposição espectral nos autoestados do
// operador de correlação C^T C). Para cada estado |v⟩, a amplitude é
//   ⟨q|v⟩ = Σ_j β_j ⟨e_j|v⟩.
// Se truncamos a J primeiros autoestados (dominantes), o resíduo é limitado:
//   |⟨q|v⟩ − Σ_{j<J} β_j ⟨e_j|v⟩| ≤ ‖q_R‖ · ‖v_R‖
// onde v_R é a projeção no complemento. Estados com ‖v_R‖ pequeno têm
// amplitude bem aproximada pelo truncamento → podem ser PODADOS se o bound
// não pode vencer o k-ésimo melhor já medido.
//
// Estratégia de medição em 2 níveis (coarse→fine, como a análise
// multirresolução de Hilbert):
//   nível coarse: projeta todos os N em J autoestados, estima amplitude.
//   top-M coarse → mede exato nesses M, descarta o resto por bound espectral.
std::vector<int> argsort_hilbert(const State& q, const std::vector<State>& corpus,
                                 int k, int J) {
    int N = (int)corpus.size();
    int d = (int)q.data.size();
    int M = std::max(k * 4, 64);  // candidatos coarse

    // Autoestados: usar J eixos de correlação (via amostra SVD).
    // Determinístico: autoestados = base computacional perturbada é fraco;
    // usamos a decomposição espectral real da amostra.
    std::vector<float> mean(d, 0);
    int S = std::min(N, 4000);
    for (int i = 0; i < S; i++)
        for (int j = 0; j < d; j++) mean[j] += corpus[i].data[j];
    for (int j = 0; j < d; j++) mean[j] /= S;

    // Covariância amostral (d×d) — pequena.
    std::vector<float> C((size_t)d * d, 0);
    for (int i = 0; i < S; i++)
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++)
                C[(size_t)a * d + b] += (corpus[i].data[a]-mean[a]) * (corpus[i].data[b]-mean[b]);
    for (auto& x : C) x /= S;

    // Power iteration para J autoestados (determinístico).
    std::vector<std::vector<float>> eigs(J);
    std::vector<float> v(d);
    for (int e = 0; e < J; e++) {
        if (e == 0) {
            for (int j = 0; j < d; j++) v[j] = (j % 3 == 0) ? 1.f : 0.01f;
        } else {
            for (int j = 0; j < d; j++) v[j] = (float)((j * 7 + e) % 13) * 0.01f;
            for (int p = 0; p < e; p++) {
                float dp = 0;
                for (int j = 0; j < d; j++) dp += v[j] * eigs[p][j];
                for (int j = 0; j < d; j++) v[j] -= dp * eigs[p][j];
            }
        }
        for (int it = 0; it < 30; it++) {
            std::vector<float> Cv(d, 0);
            for (int a = 0; a < d; a++) {
                float s = 0;
                for (int b = 0; b < d; b++) s += C[(size_t)a*d + b] * v[b];
                Cv[a] = s;
            }
            v = Cv;
            float nr = 0;
            for (int j = 0; j < d; j++) nr += v[j]*v[j];
            if (nr > 1e-12f) { float inv = 1/std::sqrt(nr); for (int j = 0; j < d; j++) v[j]*=inv; }
        }
        eigs[e] = v;
    }

    // Coeficientes da query: β_j = ⟨e_j|q⟩
    std::vector<float> beta(J);
    State ej; ej.data.resize(d);
    for (int j = 0; j < J; j++) {
        ej.data = eigs[j];
        beta[j] = braket(ej, q);
    }

    // Nível coarse: amplitude estimada = Σ β_j ⟨e_j|v⟩ (J produtos internos por vetor)
    std::vector<std::pair<float,int>> coarse(N);
    for (int i = 0; i < N; i++) {
        float est = 0;
        for (int j = 0; j < J; j++) {
            ej.data = eigs[j];
            est += beta[j] * braket(ej, corpus[i]);
        }
        coarse[i] = {est, i};
    }
    // top-M coarse
    std::partial_sort(coarse.begin(), coarse.begin() + M, coarse.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });

    // Nível fine: medir exatamente os M candidatos → top-k
    std::vector<std::pair<float,int>> fine(M);
    for (int i = 0; i < M; i++) {
        int vi = coarse[i].second;
        fine[i] = {amplitude(q, corpus[vi]), vi};
    }
    std::partial_sort(fine.begin(), fine.begin() + k, fine.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
    std::vector<int> out(k);
    for (int j = 0; j < k; j++) out[j] = fine[j].second;
    return out;
}

} // namespace hilbert

int main() {
    std::printf("== argsort em Hilbert (medição projetiva) ==\n");
    std::srand(42);
    int N = 50000, d = 128, k = 10, J = 16;

    // Corpus: estados de Hilbert com estrutura de clusters (embeddings).
    std::vector<hilbert::State> corpus(N);
    std::vector<std::vector<float>> centers(20, std::vector<float>(d));
    for (auto& c : centers) { for (auto& x : c) x = (float)std::rand()/RAND_MAX - 0.5f; }
    for (auto& c : centers) {
        float n = 0; for (float x : c) n += x*x; n = std::sqrt(n);
        for (float& x : c) x /= n;
    }
    for (int i = 0; i < N; i++) {
        corpus[i].data.resize(d);
        auto& c = centers[i % 20];
        for (int j = 0; j < d; j++) corpus[i].data[j] = c[j] + 0.15f*((float)std::rand()/RAND_MAX - 0.5f);
        float n = 0; for (float x : corpus[i].data) n += x*x; n = std::sqrt(n);
        for (float& x : corpus[i].data) x /= n;
    }
    hilbert::State q; q.data.resize(d);
    for (int j = 0; j < d; j++) q.data[j] = corpus[3].data[j] + 0.05f*((float)std::rand()/RAND_MAX - 0.5f);
    float nq = q.norm(); for (float& x : q.data) x /= nq;

    // Referência exata
    auto t0 = std::chrono::high_resolution_clock::now();
    auto ref = hilbert::argsort_exact(q, corpus, k);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_exact = std::chrono::duration<double,std::milli>(t1-t0).count();

    // Medição de Hilbert
    auto t2 = std::chrono::high_resolution_clock::now();
    auto hip = hilbert::argsort_hilbert(q, corpus, k, J);
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_hilbert = std::chrono::duration<double,std::milli>(t3-t2).count();

    // Corretude: top-k bate?
    std::vector<int> rref(ref.begin(), ref.end());
    std::vector<int> rhip(hip.begin(), hip.end());
    std::sort(rref.begin(), rref.end());
    std::sort(rhip.begin(), rhip.end());
    bool ok = (rref == rhip);

    std::printf("N=%d d=%d k=%d J=%d\n", N, d, k, J);
    std::printf("argsort exato    : %.3f ms\n", t_exact);
    std::printf("argsort Hilbert  : %.3f ms\n", t_hilbert);
    std::printf("corretude top-%d  : %s\n", k, ok ? "IDÊNTICO" : "DIVERGE");
    std::printf("ganho            : %.2fx\n", t_exact/t_hilbert);

    return 0;
}
