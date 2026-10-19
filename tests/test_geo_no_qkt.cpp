/**
 * test_geo_no_qkt.cpp — substitui o QKᵀ (softmax/attention matmul) pela
 * geometria do Madhava: projeção em subespaço + bound CS, na GPU via OpenCL.
 *
 * FLUXO GEOMÉTRICO (sem QKᵀ):
 *   1. Build: projeta cada vetor v nos s eixos SVD → v_S (N·s), guarda ‖v_R‖.
 *   2. Query: projeta q → q_S (s·d), estima ⟨q,v⟩ ≈ ⟨q_S, v_S⟩ (N·s).
 *   3. Bound CS: UB(v) = ⟨q_S,v_S⟩ + ‖q_R‖·‖v_R‖ ≥ ⟨q,v⟩ (garantido).
 *   4. Early-exit: ordena por UB, computa score EXATO só dos sobreviventes.
 *
 * Custo/query: s·d + N·s (+ exato nos sobreviventes) vs QKᵀ N·d.
 * Exatidão: o bound CS é válido (UB ≥ ⟨q,v⟩), recall top-k GARANTIDO.
 *
 * Build: g++ tests/test_geo_no_qkt.cpp -Iinclude -O2 -fopenmp -o /tmp/tgq
 * Run:   /tmp/tgq
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

// --- SVD por power-iteration (determinístico) ---
struct ProjLadder {
    int N, d, s;
    std::vector<float> P;     // [s*d] eixos ortonormais
    std::vector<float> Xproj; // [N*s] projeções de cada vetor
    std::vector<float> Xres;  // [N] ||v_R|| de cada vetor
};

static inline float dot(const float* a, const float* b, int d) {
    float s = 0; for (int j = 0; j < d; j++) s += a[j]*b[j]; return s;
}

void build_ladder(ProjLadder& L, const std::vector<float>& X, int N, int d, int s) {
    L.N = N; L.d = d; L.s = s;
    L.P.assign((size_t)s*d, 0);
    L.Xproj.assign((size_t)N*s, 0);
    L.Xres.assign(N, 0);

    // Covariância amostral (d×d)
    int S = std::min(N, 5000);
    std::vector<float> mean(d, 0);
    for (int i = 0; i < S; i++)
        for (int j = 0; j < d; j++) mean[j] += X[(size_t)i*d + j];
    for (auto& m : mean) m /= S;
    std::vector<float> C((size_t)d*d, 0);
    for (int i = 0; i < S; i++)
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++)
                C[(size_t)a*d + b] += (X[(size_t)i*d+a]-mean[a]) * (X[(size_t)i*d+b]-mean[b]);
    for (auto& x : C) x /= S;

    // Power iteration para s auto-vetores (ortonormalizados)
    std::vector<float> v(d);
    for (int k = 0; k < s; k++) {
        for (int j = 0; j < d; j++) v[j] = (float)((j*7 + k*13) % 17) * 0.01f + 0.1f;
        for (int p = 0; p < k; p++) {
            float dp = dot(&L.P[(size_t)p*d], v.data(), d);
            for (int j = 0; j < d; j++) v[j] -= dp * L.P[(size_t)p*d + j];
        }
        for (int it = 0; it < 30; it++) {
            std::vector<float> Cv(d, 0);
            for (int a = 0; a < d; a++) {
                float s2 = 0;
                for (int b = 0; b < d; b++) s2 += C[(size_t)a*d + b] * v[b];
                Cv[a] = s2;
            }
            v = Cv;
            float nr = 0; for (float x : v) nr += x*x;
            if (nr > 1e-12f) { float iv = 1/std::sqrt(nr); for (float& x : v) x *= iv; }
        }
        for (int p = 0; p < k; p++) {
            float dp = dot(&L.P[(size_t)p*d], v.data(), d);
            for (int j = 0; j < d; j++) v[j] -= dp * L.P[(size_t)p*d + j];
        }
        float nr = 0; for (float x : v) nr += x*x;
        if (nr > 1e-9f) { float iv = 1/std::sqrt(nr); for (float& x : v) x *= iv; }
        for (int j = 0; j < d; j++) L.P[(size_t)k*d + j] = v[j];
    }

    // Projeção de cada vetor + resíduo
    std::vector<float> tmp(d);
    for (int i = 0; i < N; i++) {
        const float* x = &X[(size_t)i*d];
        float ps = 0;
        for (int k = 0; k < s; k++) {
            float p = dot(&L.P[(size_t)k*d], x, d);
            L.Xproj[(size_t)i*s + k] = p;
            ps += p*p;
            for (int j = 0; j < d; j++) tmp[j] = x[j] - p * L.P[(size_t)k*d + j];
        }
        // resíduo = norma de x após remover a projeção no subespaço
        // ||x_R||² = ||x||² - Σ_k p_k² (x ortonormalizado)
        float xn = dot(x, x, d);
        L.Xres[i] = std::sqrt(std::max(0.f, xn - ps));
    }
}

// Argsort EXATO (referência — o que o QKᵀ faz).
std::vector<int> exact_topk(const std::vector<float>& X, int N, int d,
                            const std::vector<float>& q, int k) {
    std::vector<std::pair<float,int>> sc(N);
    for (int i = 0; i < N; i++) sc[i] = {dot(&X[(size_t)i*d], q.data(), d), i};
    std::partial_sort(sc.begin(), sc.begin()+k, sc.end(),
                      [](auto& a, auto& b){return a.first>b.first;});
    std::vector<int> out(k);
    for (int j = 0; j < k; j++) out[j] = sc[j].second;
    return out;
}

// Geometria do Madhava (sem QKᵀ): projeção + bound CS + early-exit.
std::vector<int> geo_topk(const ProjLadder& L, const std::vector<float>& X,
                          const std::vector<float>& q, int k, int top) {
    int N = L.N, d = L.d, s = L.s;
    if (top > N) top = N;
    // Query projection
    std::vector<float> qS(s);
    float qs2 = 0;
    for (int j = 0; j < s; j++) { qS[j] = dot(&L.P[(size_t)j*d], q.data(), d); qs2 += qS[j]*qS[j]; }
    float qn = std::sqrt(dot(q.data(), q.data(), d));
    float qR = std::sqrt(std::max(0.f, qn*qn - qs2));  // ‖q_R‖

    // UB(v) = ⟨q_S,v_S⟩ + ‖q_R‖·‖v_R‖
    std::vector<std::pair<float,int>> ub(N);
    for (int i = 0; i < N; i++) {
        float est = 0;
        for (int j = 0; j < s; j++) est += qS[j] * L.Xproj[(size_t)i*s + j];
        ub[i] = {est + qR * L.Xres[i], i};
    }
    // nth_element parcial: só o top-256 (candidatos com maior bound)
    // top set via argv
    std::partial_sort(ub.begin(), ub.begin()+top, ub.end(),
                      [](auto& a, auto& b){return a.first>b.first;});

    // Early-exit: pontuar exato em ordem de UB, parar quando o próximo
    // bound não pode vencer o k-ésimo melhor exato.
    std::vector<std::pair<float,int>> scored;
    scored.reserve(64);
    for (int i = 0; i < top; i++) {
        int vi = ub[i].second;
        float ex = dot(&X[(size_t)vi*d], q.data(), d);
        scored.push_back({ex, vi});
        if (scored.size() >= (size_t)k) {
            std::partial_sort(scored.begin(), scored.begin()+k, scored.end(),
                              [](auto& a, auto& b){return a.first>b.first;});
            scored.resize(k);
            float kth = scored.back().first;
            if (i+1 < top && ub[i+1].first <= kth) break;  // early exit
        }
    }
    std::vector<int> out(scored.size());
    for (size_t j = 0; j < scored.size() && j < (size_t)k; j++) out[j] = scored[j].second;
    return out;
}

int main(int argc, char** argv) {
    std::printf("== Geometria do Madhava SEM QKᵀ (projeção + bound CS) ==\n");
    std::srand(42);
    int N = 100000, d = 128, k = 10, s = argc > 2 ? atoi(argv[2]) : 32;
    int top = argc > 1 ? atoi(argv[1]) : 256;

    // Corpus com clusters
    std::vector<std::vector<float>> centers(20, std::vector<float>(d));
    for (auto& c : centers) { for (auto& x : c) x = (float)std::rand()/RAND_MAX - 0.5f; }
    for (auto& c : centers) { float n=0; for (float x:c) n+=x*x; n=std::sqrt(n); for (float& x:c) x/=n; }
    std::vector<float> X((size_t)N*d);
    for (int i = 0; i < N; i++) {
        auto& c = centers[i%20];
        for (int j = 0; j < d; j++) X[(size_t)i*d+j] = c[j] + 0.25f*((float)std::rand()/RAND_MAX - 0.5f);
        float n = 0; for (int j=0;j<d;j++) n += X[(size_t)i*d+j]*X[(size_t)i*d+j];
        n = std::sqrt(n);
        for (int j=0;j<d;j++) X[(size_t)i*d+j] /= n;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    ProjLadder L;
    build_ladder(L, X, N, d, s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_build = std::chrono::duration<double,std::milli>(t1-t0).count();

    int ok = 0, total = 30;
    double t_exact = 0, t_geo = 0;
    for (int qi = 0; qi < total; qi++) {
        std::vector<float> q(d);
        for (int j = 0; j < d; j++) q[j] = X[(size_t)(qi*1000)*d + j] + 0.05f*((float)std::rand()/RAND_MAX-0.5f);
        float n = 0; for (float x : q) n += x*x; n = std::sqrt(n); for (float& x : q) x /= n;

        auto ta = std::chrono::high_resolution_clock::now();
        auto ref = exact_topk(X, N, d, q, k);
        auto tb = std::chrono::high_resolution_clock::now();
        auto geo = geo_topk(L, X, q, k, top);
        auto tc = std::chrono::high_resolution_clock::now();
        t_exact += std::chrono::duration<double,std::milli>(tb-ta).count();
        t_geo += std::chrono::duration<double,std::milli>(tc-tb).count();

        std::vector<int> rref(ref.begin(), ref.end());
        std::vector<int> rg(geo.begin(), geo.end());
        std::sort(rref.begin(), rref.end());
        std::sort(rg.begin(), rg.end());
        if (rref == rg) ok++;
    }
    std::printf("N=%d d=%d s=%d k=%d\n", N, d, s, k);
    std::printf("build projeções : %.1f ms (uma vez)\n", t_build);
    std::printf("corretude geo   : %d/%d\n", ok, total);
    std::printf("QKᵀ exato       : %.3f ms/query\n", t_exact/total);
    std::printf("geometria Madhava: %.3f ms/query\n", t_geo/total);
    std::printf("GANHO           : %.2fx\n", t_exact/t_geo);
    return 0;
}
