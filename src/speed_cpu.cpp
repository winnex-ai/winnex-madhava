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
#include <cstdio>
#include <cstring>
#include <stdexcept>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace winnex_madhava {

#if !defined(MADHAVA_HAS_CUDA) && !defined(MADHAVA_HAS_OPENCL)
// GPU enable: the real implementation lives in speed_gpu.cu (CUDA) or
// speed_opencl.cpp (OpenCL). This CPU TU provides the fallback: the GPU
// cannot be enabled here, so we log why and (when require_gpu is set) throw
// instead of silently degrading.
void SpeedEngine::enable_gpu(const std::vector<float>& /*corpus*/, int /*n*/, int /*dim*/) {
    gpu_reason_ = "build without a GPU backend (neither CUDA nor OpenCL)";
    if (require_gpu_) {
        throw std::runtime_error(
            "SpeedEngine: GPU required but this build has no GPU backend. "
            "Build with OpenCL or CUDA, or pass require_gpu=False to fall "
            "back to the CPU backend.");
    }
    std::fprintf(stderr,
                 "[Winnex Madhava] SpeedEngine: GPU unavailable — falling back "
                 "to CPU backend. Reason: %s\n",
                 gpu_reason_.c_str());
}
#endif

#if !defined(MADHAVA_HAS_CUDA) && !defined(MADHAVA_HAS_OPENCL)
// Static GPU hooks — CPU-only stubs (real impls live in the GPU backend).
void SpeedEngine::init_gpu_impl() {}
void SpeedEngine::free_gpu_impl() {}
bool SpeedEngine::gpu_available() { return false; }
#endif

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
SpeedEngine::SpeedEngine(const float* corpus, int n, int dim, Metric metric,
                         int n_anchors, int nprobe, bool require_gpu,
                         const std::string& opencl_lib)
    : n_(n), dim_(dim), is_cosine_(metric == Metric::Cosine),
      require_gpu_(require_gpu), opencl_lib_(opencl_lib) {
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
    n_anchors_ = n_anchors; nprobe_ = nprobe;
    if (n_anchors_ >= 2) { build_anchors(); }
    enable_gpu(corpus_f32_, n, dim);
}

SpeedEngine::SpeedEngine(const uint8_t* corpus, int n, int dim, Metric metric,
                         int n_anchors, int nprobe, bool require_gpu,
                         const std::string& opencl_lib)
    : n_(n), dim_(dim), is_cosine_(metric == Metric::Cosine),
      require_gpu_(require_gpu), opencl_lib_(opencl_lib) {
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
    n_anchors_ = n_anchors; nprobe_ = nprobe;
    if (n_anchors_ >= 2) { build_anchors(); }
    enable_gpu(corpus_f32_, n, dim);
}

SpeedEngine::~SpeedEngine() {
    // GPU buffers are freed by the CUDA backend in speed_gpu.cu.
}

// --- PiPrime anchors: SVD + Gram-Schmidt + Voronoi cell assignment --------
// K anchors are orthonormal vectors that partition the sphere. Each corpus
// vector is assigned to its nearest anchor (Voronoi). A query finds its
// top-nprobe cells by O(K·d) navigation, then QKᵀ runs only over those cells.
void SpeedEngine::build_anchors() {
    int n = n_, d = dim_, K = n_anchors_;
    if (K > n) K = n;
    if (K < 2) return;
    use_anchors_ = true;

    // --- Compute K anchors via SVD (offline, once) ---
    // Seeds: mean + top-K principal components (like PiPrime), then GS-orthogonalize.
    std::vector<float> mean(d, 0.0f);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) mean[j] += corpus_f32_[(size_t)i * d + j];
    for (int j = 0; j < d; j++) mean[j] /= (float)n;

    // SVD on the (small) covariance is expensive for large N; use a sample.
    int sample = std::min(n, 10000);
    std::vector<float> A((size_t)sample * d);
    for (int i = 0; i < sample; i++)
        for (int j = 0; j < d; j++)
            A[(size_t)i * d + j] = corpus_f32_[(size_t)i * d + j] - mean[j];

    // Covariance (d x d), small: d^2.
    std::vector<float> C((size_t)d * d, 0.0f);
    for (int i = 0; i < sample; i++)
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++)
                C[(size_t)a * d + b] += A[(size_t)i * d + a] * A[(size_t)i * d + b];
    for (int j = 0; j < (size_t)d * d; j++) C[j] /= (float)sample;

    // Power iteration for the top-K eigenvectors of C (deterministic seeds).
    anchors_.assign((size_t)K * d, 0.0f);
    std::vector<float> v(d, 0.0f);
    for (int k = 0; k < K; k++) {
        if (k == 0) {
            for (int j = 0; j < d; j++) v[j] = mean[j];  // direction of mean
        } else {
            for (int j = 0; j < d; j++) v[j] = (float)((j % 7) + 1) * 0.01f;  // perturbed seed
            // orthogonalize against previous anchors
            for (int p = 0; p < k; p++) {
                float dp = 0;
                for (int j = 0; j < d; j++) dp += v[j] * anchors_[(size_t)p * d + j];
                for (int j = 0; j < d; j++) v[j] -= dp * anchors_[(size_t)p * d + j];
            }
        }
        // Power iterations on C
        for (int it = 0; it < 20; it++) {
            std::vector<float> Cv(d, 0.0f);
            for (int a = 0; a < d; a++) {
                float s = 0;
                for (int b = 0; b < d; b++) s += C[(size_t)a * d + b] * v[b];
                Cv[a] = s;
            }
            v = Cv;
            float nr = 0;
            for (int j = 0; j < d; j++) nr += v[j] * v[j];
            if (nr > 1e-12f) {
                float inv = 1.0f / std::sqrt(nr);
                for (int j = 0; j < d; j++) v[j] *= inv;
            }
        }
        // Gram-Schmidt orthogonalize v against all previous anchors
        for (int p = 0; p < k; p++) {
            float dp = 0;
            for (int j = 0; j < d; j++) dp += v[j] * anchors_[(size_t)p * d + j];
            for (int j = 0; j < d; j++) v[j] -= dp * anchors_[(size_t)p * d + j];
        }
        float nr = 0;
        for (int j = 0; j < d; j++) nr += v[j] * v[j];
        if (nr > 1e-9f) {
            float inv = 1.0f / std::sqrt(nr);
            for (int j = 0; j < d; j++) anchors_[(size_t)k * d + j] = v[j] * inv;
        } else {
            anchors_[(size_t)k * d + 0] = 1.0f;  // fallback
        }
    }

    // --- Voronoi assignment: cell_of_[i] = argmax_k <e_i, a_k> ---
    cell_of_.resize((size_t)n);
    cells_.assign((size_t)K, {});
    cell_radius_.assign((size_t)K, 0.0f);
    for (int i = 0; i < n; i++) {
        const float* e = corpus_f32_.data() + (size_t)i * d;
        float best = -1e30f; int bk = 0;
        for (int k = 0; k < K; k++) {
            const float* a = anchors_.data() + (size_t)k * d;
            float s = 0;
            for (int j = 0; j < d; j++) s += e[j] * a[j];
            if (s > best) { best = s; bk = k; }
        }
        cell_of_[i] = bk;
        cells_[bk].push_back(i);
        // radius: ||e - a_k||
        const float* a = anchors_.data() + (size_t)bk * d;
        float rsq = 0;
        for (int j = 0; j < d; j++) { float t = e[j] - a[j]; rsq += t * t; }
        if (rsq > cell_radius_[bk]) cell_radius_[bk] = rsq;
    }
    for (int k = 0; k < K; k++) cell_radius_[k] = std::sqrt(cell_radius_[k]);
}

#if !defined(MADHAVA_HAS_CUDA) && !defined(MADHAVA_HAS_OPENCL)
// Stub for the GPU scores matmul (defined in speed_gpu.cu / speed_opencl.cpp
// when a GPU backend is present). Never called when use_gpu_ is false; exists
// only so the CPU TU links cleanly without any GPU backend.
void SpeedEngine::scores_gpu(const float*, int, float*) const {
    // unreachable without a GPU backend
}

// Stub for the optimized GPU topk (defined in speed_gpu.cu / speed_opencl.cpp
// when a GPU backend is present). Never called when use_gpu_ is false.
SpeedResult SpeedEngine::scores_gpu_topk(const float*, int, int,
                                         std::vector<int>&) const {
    SpeedResult r;  // unreachable without a GPU backend
    return r;
}

// Stub for the Phase-2 bound Stage-1 scan (defined in speed_opencl.cpp when a
// GPU backend is present). Never called when use_gpu_ is false.
void SpeedEngine::bound_stage1_gpu(const float*, const float*, const float*,
                                   const float*, const float*, int, int, int,
                                   int, float*) const {
    // unreachable without a GPU backend
}
#endif

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

    // --- O(K) anchor navigation: route the query to the nprobe most-similar
    // anchor cells, then score ONLY those cells (sublinear, not brute force).
    std::vector<std::pair<float, int>> scores;
    long long bound_pairs = 0;
    if (use_anchors_) {
        int K = n_anchors_;
        int np = std::min(nprobe_, K);
        // O(K·d): score the query against the K anchors.
        std::vector<std::pair<float, int>> anchor_scores(K);
        for (int k = 0; k < K; k++) {
            const float* a = anchors_.data() + (size_t)k * d;
            float s = dot_f32(a, q, d);
            anchor_scores[k] = {s, k};
        }
        // Top-nprobe anchors (cells).
        std::partial_sort(anchor_scores.begin(), anchor_scores.begin() + np,
                          anchor_scores.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        // Score members of the selected cells (parallel across cells).
        int total = 0;
        for (int ci = 0; ci < np; ci++) total += (int)cells_[anchor_scores[ci].second].size();
        scores.resize((size_t)total);
#pragma omp parallel for
        for (int ci = 0; ci < np; ci++) {
            int k = anchor_scores[ci].second;
            const auto& members = cells_[k];
            int base = 0;
            for (int cj = 0; cj < ci; cj++) base += (int)cells_[anchor_scores[cj].second].size();
            for (size_t mi = 0; mi < members.size(); mi++) {
                int vi = members[mi];
                const float* v = corpus_f32_.data() + (size_t)vi * d;
                float ip = dot_f32(v, q, d);
                float score = is_cosine_ ? ip
                    : (float)(2.0 * (double)ip - (double)norms_[vi]);
                scores[base + (int)mi] = {score, vi};
            }
        }
        out.bound_pairs = total;
    } else {
        // Brute-force exact scan (full QKᵀ).
        scores.resize((size_t)N);
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            const float* v = corpus_f32_.data() + (size_t)i * d;
            float ip = dot_f32(v, q, d);
            float score = is_cosine_ ? ip
                : (float)(2.0 * (double)ip - (double)norms_[i]);
            scores[i] = {score, i};
        }
        out.bound_pairs = N;
    }
    // score = ip (cosine: higher is better) OR ||q||² - L2² (L2: higher is
    // better because it means smaller L2²). So BOTH sort descending.
    int nscored = (int)scores.size();
    k = std::min(k, nscored);
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
        std::vector<int> idx;
        SpeedResult out = scores_gpu_topk(query, 1, k, idx);
        out.indices = idx;
        out.latency_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return out;
    }
    return search_cpu(query, k);
}

SpeedResult SpeedEngine::search_batch(const float* queries, int nq, int k) const {
    if (use_gpu_) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<int> idx;
        SpeedResult out = scores_gpu_topk(queries, nq, k, idx);
        out.indices = idx;
        out.latency_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return out;
    }
    return search_batch_cpu(queries, nq, k);
}

} // namespace winnex_madhava
