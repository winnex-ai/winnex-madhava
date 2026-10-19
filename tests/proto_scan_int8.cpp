// proto_scan_int8.cpp — Protótipo: validar o ganho de quantizar as PROJEÇÕES
// para int8 no scan do bound (corpus float32 mantido p/ e(v)/PCA).
//
// O scan atual (float32) relê pr1_f_ (N×s1×4B) do DRAM a cada query.
// A hipótese: quantizar pr1 → int8 (N×s1×1B) com escala per-axis dá ~4× menos
// bandwidth → scan ~4× mais rápido em N grande, com erro de bound coberto
// pela margem (como o caminho int8 do motor já faz com qm).
//
// Este protótipo MEDE (sem tocar no motor):
//   1. scan float32 (dot_f32 sobre N×s1 float) vs int8 (dot_int8_scaled)
//   2. o erro do bound int8 vs float32 (quantos docs mudariam de lado)
//
// Compila: g++ -O3 -march=native -fopenmp tests/proto_scan_int8.cpp -o /tmp/proto_scan
// Roda:   /tmp/proto_scan [N] [s1] [nq]

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <random>

// ---- cópia do dot_f32 / dot_int8_scaled do motor (mesma matemática) ----
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
inline float dot_f32(const float* a, const float* b, int d) {
    __m256 s = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8)
        s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
    float o[8];
    _mm256_storeu_ps(o, s);
    float r = o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7];
    for (; i < d; i++) r += a[i] * b[i];
    return r;
}
inline float dot_int8_scaled(const int8_t* a, const float* scale, const float* b, int d) {
    __m256 s = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m128i u8lo = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i u8hi = _mm_srli_si128(u8lo, 8);
        __m256i i32lo = _mm256_cvtepi8_epi32(u8lo);
        __m256i i32hi = _mm256_cvtepi8_epi32(u8hi);
        __m256 flo = _mm256_cvtepi32_ps(i32lo);
        __m256 fhi = _mm256_cvtepi32_ps(i32hi);
        __m256 slo = _mm256_loadu_ps(scale + i);
        __m256 shi = _mm256_loadu_ps(scale + i + 8);
        __m256 blo = _mm256_loadu_ps(b + i);
        __m256 bhi = _mm256_loadu_ps(b + i + 8);
        s = _mm256_fmadd_ps(flo, _mm256_mul_ps(slo, blo), s);
        s = _mm256_fmadd_ps(fhi, _mm256_mul_ps(shi, bhi), s);
    }
    float o[8];
    _mm256_storeu_ps(o, s);
    float r = o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7];
    for (; i < d; i++) r += (float)a[i] * scale[i] * b[i];
    return r;
}
#else
inline float dot_f32(const float* a, const float* b, int d) {
    float s = 0; for (int i = 0; i < d; i++) s += a[i]*b[i]; return s;
}
inline float dot_int8_scaled(const int8_t* a, const float* scale, const float* b, int d) {
    float s = 0; for (int i = 0; i < d; i++) s += (float)a[i]*scale[i]*b[i]; return s;
}
#endif

// quantização (cópia do motor: escala per-axis, round-half-away)
inline void quantize(const float* src, int8_t* dst, const float* scale, int dims) {
    for (int j = 0; j < dims; j++) {
        int qi = (int)(src[j] / scale[j] + (src[j] >= 0 ? 0.5f : -0.5f));
        if (qi > 127) qi = 127;
        if (qi < -128) qi = -128;
        dst[j] = (int8_t)qi;
    }
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 100000;
    int s1 = argc > 2 ? atoi(argv[2]) : 192;
    int nq = argc > 3 ? atoi(argv[3]) : 20;
    int K = 10;

    // Gera projeções realistas (manifold + ruído, ~como Pv de embeddings).
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> pr_f((size_t)N * s1);
    for (auto& x : pr_f) x = nd(rng) * 0.3f;   // projeções ~N(0, 0.3)
    // força um manifold: algumas colunas dominantes (como PCA real)
    for (int j = 0; j < std::min(s1, 16); j++)
        for (int i = 0; i < N; i++) pr_f[(size_t)i*s1 + j] += nd(rng) * 2.0f;

    // Escalas per-axis (calibração do motor: max abs por coluna * 1.05 / 127)
    std::vector<float> scale(s1);
    for (int j = 0; j < s1; j++) {
        float ma = 0;
        for (int i = 0; i < N; i++) ma = std::max(ma, std::fabs(pr_f[(size_t)i*s1 + j]));
        scale[j] = ma / 127.0f * 1.05f;
    }

    // Quantiza para int8
    std::vector<int8_t> pr_i((size_t)N * s1);
    for (int i = 0; i < N; i++) quantize(&pr_f[(size_t)i*s1], &pr_i[(size_t)i*s1], scale.data(), s1);

    // Queries (projeções de query, float32)
    std::vector<std::vector<float>> qs(nq);
    std::vector<float> qr(nq), qm(nq);
    for (int qi = 0; qi < nq; qi++) {
        qs[qi].resize(s1);
        for (auto& x : qs[qi]) x = nd(rng) * 0.3f;
        for (int j = 0; j < std::min(s1,16); j++) qs[qi][j] += nd(rng) * 2.0f;
        // residual da query + margem int8 (como o motor)
        float q1s = 0; for (int j=0;j<s1;j++) q1s += qs[qi][j]*qs[qi][j];
        qr[qi] = std::sqrt(std::max(0.0f, 1.0f - q1s));
        float m = 0; for (int j=0;j<s1;j++) m += 0.5f*scale[j]*std::fabs(qs[qi][j]);
        qm[qi] = m;
    }
    // residuals dos docs e(v): ~sqrt(1 - ||Pv||²) (para o bound completo)
    std::vector<float> e1(N);
    for (int i = 0; i < N; i++) {
        float pn = 0;
        for (int j=0;j<s1;j++) pn += pr_f[(size_t)i*s1+j]*pr_f[(size_t)i*s1+j];
        e1[i] = std::sqrt(std::max(0.0f, 1.0f - pn));
    }

    // ---- BENCH: scan float32 vs int8 (a operação do motor) ----
    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    auto ms = [&](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };

    // warmup + bench float32
    volatile float sink = 0;
    auto t0 = now();
    for (int qi = 0; qi < nq; qi++) {
        const float* pq = qs[qi].data();
        for (int i = 0; i < N; i++) {
            float ub = dot_f32(&pr_f[(size_t)i*s1], pq, s1) + e1[i]*qr[qi] + 1e-4f;
            sink += ub;
        }
    }
    double t_f32 = ms(t0, now());

    // bench int8
    t0 = now();
    for (int qi = 0; qi < nq; qi++) {
        const float* pq = qs[qi].data();
        for (int i = 0; i < N; i++) {
            float ub = dot_int8_scaled(&pr_i[(size_t)i*s1], scale.data(), pq, s1)
                       + e1[i]*qr[qi] + qm[qi] + 1e-4f;
            sink += ub;
        }
    }
    double t_i8 = ms(t0, now());
    (void)sink;

    // ---- PRECISÃO: o bound int8 inverte decisões vs float32? ----
    // Compara UB float32 vs UB int8 para o MESMO threshold (top-10 do float32).
    // Conta quantos docs mudariam de "excluído" p/ "mantido" (falso negativo do
    // int8) e vice-versa — o que importa é NÃO perder um vizinho verdadeiro.
    int flips_excl_to_keep = 0, flips_keep_to_excl = 0;
    double max_err = 0;
    for (int qi = 0; qi < std::min(nq, 5); qi++) {
        const float* pq = qs[qi].data();
        // threshold = score do 10º melhor por bound float32 (aproximado: top-10 por ub f32)
        std::vector<float> ub_f32(N), ub_i8(N);
        for (int i = 0; i < N; i++) {
            ub_f32[i] = dot_f32(&pr_f[(size_t)i*s1], pq, s1) + e1[i]*qr[qi] + 1e-4f;
            ub_i8[i]  = dot_int8_scaled(&pr_i[(size_t)i*s1], scale.data(), pq, s1)
                        + e1[i]*qr[qi] + qm[qi] + 1e-4f;
            max_err = std::max(max_err, (double)std::fabs(ub_f32[i] - ub_i8[i]));
        }
        // threshold do float32 (o "verdadeiro")
        std::vector<float> srt = ub_f32;
        std::nth_element(srt.begin(), srt.begin()+K-1, srt.end(), std::greater<float>());
        float thr = srt[K-1];
        for (int i = 0; i < N; i++) {
            bool excl_f32 = ub_f32[i] < thr;
            bool excl_i8  = ub_i8[i]  < thr;
            if (excl_f32 && !excl_i8) flips_excl_to_keep++;  // int8 mantém o que f32 excluiu (seguro p/ recall mas +trabalho)
            if (!excl_f32 && excl_i8) flips_keep_to_excl++;  // int8 exclui o que f32 manteve (PERIGOSO: pode perder vizinho)
        }
    }

    printf("N=%d s1=%d nq=%d\n", N, s1, nq);
    printf("scan float32: %.3f ms  (%d queries)  = %.3f ms/query\n", t_f32, nq, t_f32/nq);
    printf("scan int8   : %.3f ms  (%d queries)  = %.3f ms/query\n", t_i8, nq, t_i8/nq);
    printf("GANHO int8  : %.2fx\n", t_f32/t_i8);
    printf("erro max UB int8 vs float32: %.2e\n", max_err);
    printf("flips excl->keep (seguro, +trabalho): %d\n", flips_excl_to_keep);
    printf("flips keep->excl (PERIGOSO, perde vizinho): %d\n", flips_keep_to_excl);
    return 0;
}
