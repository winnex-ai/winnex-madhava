/**
 * speed_cpu.cpp — native speed mode, CPU backend (OpenMP + AVX2)
 * =================================================================
 * The QKᵀ matmul (scores = Q @ corpus.T) and the per-row topk, implemented
 * natively with OpenMP parallelism and the same AVX2 dot-product kernels the
 * bound engine uses. This is the always-available path; the CUDA path is
 * layered on top when a CUDA toolkit is present at build time.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#include "winnex_madhava/speed_engine.hpp"
#include "winnex_madhava/winnex_madhava.hpp"  // for Metric

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace winnex_madhava {

// The CUDA enable helper is defined in speed_gpu.cu (which has CUDA). In
// this CPU TU we provide a no-op stub so the constructors compile and
// use_gpu_ stays false — the native CPU backend is always correct.
namespace {
void _enable_gpu(class SpeedEngine& /*self*/, const std::vector<float>& /*corpus*/,
                 int /*n*/, int /*dim*/) {}
}

namespace {

inline float dot_f32(const float* a, const float* b, int d) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8)
        s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
    float o[8];
    _mm256_storeu_ps(o, s);
    float r = o[0] + o[1] + o[2] + o[3] + o[4] + o[5] + o[6] + o[7];
    for (; i < d; i++) r += a[i] * b[i];
    return r;
#else
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
#endif
}

} // namespace

// --- Construction ---------------------------------------------------------
SpeedEngine::SpeedEngine(const float* corpus, int n, int dim, Metric metric)
    : n_(n), dim_(dim), is_cosine_(metric == Metric::Cosine) {
    corpus_f32_.resize((size_t)n * dim);
    norms_.resize((size_t)n);
    if (is_cosine_) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            const float* v = corpus + (size_t)i * dim;
            float* dst = corpus_f32_.data() + (size_t)i * dim;
            float norm = 0;
            for (int j = 0; j < dim; j++) { dst[j] = v[j]; norm += v[j] * v[j]; }
            norm = std::sqrt(norm);
            if (norm > 1e-12f) {
                float inv = 1.0f / norm;
                for (int j = 0; j < dim; j++) dst[j] *= inv;
            }
            norms_[i] = 1.0f;
        }
    } else {
        std::memcpy(corpus_f32_.data(), corpus, (size_t)n * dim * sizeof(float));
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            const float* v = corpus_f32_.data() + (size_t)i * dim;
            float nrm = 0;
            for (int j = 0; j < dim; j++) nrm += v[j] * v[j];
            norms_[i] = nrm;  // ||v||²
        }
    }
    _enable_gpu(*this, corpus_f32_, n, dim);
}

SpeedEngine::SpeedEngine(const uint8_t* corpus, int n, int dim, Metric metric)
    : n_(n), dim_(dim), is_cosine_(metric == Metric::Cosine) {
    corpus_f32_.resize((size_t)n * dim);
    norms_.resize((size_t)n);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const uint8_t* v = corpus + (size_t)i * dim;
        float* dst = corpus_f32_.data() + (size_t)i * dim;
        float norm = 0;
        for (int j = 0; j < dim; j++) { dst[j] = (float)v[j]; norm += dst[j] * dst[j]; }
        if (is_cosine_) {
            norm = std::sqrt(norm);
            if (norm > 1e-12f) {
                float inv = 1.0f / norm;
                for (int j = 0; j < dim; j++) dst[j] *= inv;
            }
            norms_[i] = 1.0f;
        } else {
            norms_[i] = norm;  // ||v||²
        }
    }
    _enable_gpu(*this, corpus_f32_, n, dim);
}

SpeedEngine::~SpeedEngine() {
    // GPU buffers are freed by the CUDA backend in speed_gpu.cu.
}

// Stub for the CUDA scores matmul (defined in speed_gpu.cu when CUDA is
// present). Never called when use_gpu_ is false; exists only so the CPU TU
// links cleanly without a CUDA toolkit.
void SpeedEngine::scores_gpu(const float*, int, float*) const {
    // unreachable without CUDA
}

// --- Single-query search (CPU) --------------------------------------------
SpeedResult SpeedEngine::search_cpu(const float* query, int k) const {
    SpeedResult out;
    auto t0 = std::chrono::high_resolution_clock::now();
    int N = n_, d = dim_;
    k = std::min(k, N);

    // Normalize the query for cosine.
    std::vector<float> qbuf(d);
    const float* q = query;
    if (is_cosine_) {
        float qn = 0;
        for (int j = 0; j < d; j++) qn += query[j] * query[j];
        qn = std::sqrt(qn);
        float inv = qn > 1e-12f ? 1.0f / qn : 0.0f;
        for (int j = 0; j < d; j++) qbuf[j] = query[j] * inv;
        q = qbuf.data();
    }
    float qn2 = 0;
    for (int j = 0; j < d; j++) qn2 += q[j] * q[j];

    std::vector<std::pair<float, int>> scores(N);
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        const float* v = corpus_f32_.data() + (size_t)i * d;
        float ip = dot_f32(v, q, d);
        // Cosine: higher ip is better. L2²: ||v||² + ||q||² - 2·<v,q>;
        // ordering by -(L2²) ≡ ordering by 2·ip - ||v||².
        // Use double for the L2 combination to avoid fp32 cancellation when
        // 2·ip and ||v||² are both ~1e6 (uint8 corpora) — reordering near-ties.
        float score = is_cosine_ ? ip : (float)(2.0 * (double)ip - (double)norms_[i]);
        scores[i] = {score, i};
    }
    out.bound_pairs = N;
    // score = ip (cosine: higher is better) OR ||q||² - L2² (L2: higher is
    // better because it means smaller L2²). So BOTH sort descending.
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    for (int i = 0; i < k; i++) out.indices.push_back(scores[i].second);

    out.latency_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

// --- Batch search (CPU) ----------------------------------------------------
SpeedResult SpeedEngine::search_batch_cpu(const float* queries, int nq, int k) const {
    SpeedResult out;
    auto t0 = std::chrono::high_resolution_clock::now();
    int N = n_, d = dim_;
    k = std::min(k, N);
    out.bound_pairs = (long long)N * nq;

    // Normalize all queries.
    std::vector<float> qnorm((size_t)nq * d);
    if (is_cosine_) {
#pragma omp parallel for
        for (int qi = 0; qi < nq; qi++) {
            const float* q = queries + (size_t)qi * d;
            float* qd = qnorm.data() + (size_t)qi * d;
            float qn = 0;
            for (int j = 0; j < d; j++) qn += q[j] * q[j];
            float inv = qn > 1e-12f ? 1.0f / std::sqrt(qn) : 0.0f;
            for (int j = 0; j < d; j++) qd[j] = q[j] * inv;
        }
    } else {
        std::memcpy(qnorm.data(), queries, (size_t)nq * d * sizeof(float));
    }

    std::vector<std::pair<float, int>> scores(N);
    for (int qi = 0; qi < nq; qi++) {
        const float* q = qnorm.data() + (size_t)qi * d;
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            const float* v = corpus_f32_.data() + (size_t)i * d;
            float ip = dot_f32(v, q, d);
            float score = is_cosine_ ? ip : (2.0f * ip - norms_[i]);
            scores[i] = {score, i};
        }
        // score = ip (cosine: higher better) OR ||q||²-L2² (L2: higher =
        // smaller L2² = better). Both sort descending.
        std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        for (int i = 0; i < k; i++) out.indices.push_back(scores[i].second);
    }
    out.latency_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

// --- Public dispatch (CUDA if available, else CPU) ------------------------
// The GPU path uses cuBLAS SGEMM for the QKᵀ matmul + host topk. It is
// enabled only when the wheel was built with CUDA and a device is present.

namespace {
SpeedResult _topk_host(const std::vector<float>& scores_row, int N, int k,
                       const std::vector<float>& norms, bool cosine) {
    SpeedResult out;
    k = std::min(k, N);
    std::vector<std::pair<float, int>> s(N);
    for (int i = 0; i < N; i++) {
        float ip = scores_row[i];
        float score = cosine ? ip : (float)(2.0 * (double)ip - (double)norms[i]);
        s[i] = {score, i};
    }
    // score = ip (cosine: higher better) OR ||q||²-L2² (L2: higher = smaller
    // L2² = better). Both sort descending.
    std::partial_sort(s.begin(), s.begin() + k, s.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    for (int i = 0; i < k; i++) out.indices.push_back(s[i].second);
    out.bound_pairs = N;
    return out;
}
} // namespace

SpeedResult SpeedEngine::search(const float* query, int k) const {
    if (use_gpu_) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<float> scores((size_t)n_);
        scores_gpu(query, 1, scores.data());
        SpeedResult out = _topk_host(scores, n_, k, norms_, is_cosine_);
        out.latency_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return out;
    }
    return search_cpu(query, k);
}

SpeedResult SpeedEngine::search_batch(const float* queries, int nq, int k) const {
    if (use_gpu_) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<float> all_scores((size_t)nq * n_);
        scores_gpu(queries, nq, all_scores.data());
        SpeedResult out;
        out.bound_pairs = (long long)nq * n_;
        for (int qi = 0; qi < nq; qi++) {
            std::vector<float> row(all_scores.begin() + (size_t)qi * n_,
                                   all_scores.begin() + (size_t)(qi + 1) * n_);
            SpeedResult r = _topk_host(row, n_, k, norms_, is_cosine_);
            out.indices.insert(out.indices.end(), r.indices.begin(), r.indices.end());
        }
        out.latency_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return out;
    }
    return search_batch_cpu(queries, nq, k);
}

} // namespace winnex_madhava
