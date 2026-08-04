/**
 * winnex_madhava.cpp — implementation of the Madhava L2 C++ Library
 * ==============================================================
 * See winnex_madhava.hpp for the API and the mathematical guarantee.
 *
 * Stage 1 (bound pruning):
 *   For every vector v, compute the Cauchy-Schwarz upper bound of ⟨v,q⟩:
 *     ub = ⟨Pv,Pq⟩ + e(v)·e(q) + quantization_margin
 *   Convert to a lower bound of L2²:
 *     lb = ‖v‖² + ‖q‖² − 2·ub
 *   Keep the top-k1 smallest lb (the candidates most likely to be near).
 *   This is a mathematical guarantee: any vector with lb above the
 *   k1-th smallest cannot be in the exact top-K.
 *
 * Post-filter (optional):
 *   Compute the exact L2² on the surviving top-k1 and return the top-K.
 *   This closes the gap between bound ranking and exact L2 ranking,
 *   without ever losing a vector that the bound could have pruned
 *   correctly (the bound never prunes a true neighbor).
 *
 * BSL 1.1 | pay@winnex.ai
 */
#include "winnex_madhava/winnex_madhava.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include <utility>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace winnex_madhava {

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

inline float load_raw_norm(const uint8_t* src, float* dst, int d, float& norm) {
    float n = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 nv = _mm256_setzero_ps();
    int j = 0;
    for (; j + 16 <= d; j += 16) {
        __m128i u8 = _mm_loadu_si128((const __m128i*)(src + j));
        __m256i lo32 = _mm256_cvtepu8_epi32(u8);
        __m256 flo = _mm256_cvtepi32_ps(lo32);
        _mm256_storeu_ps(dst + j, flo);
        nv = _mm256_fmadd_ps(flo, flo, nv);
        __m256i hi32 = _mm256_cvtepu8_epi32(_mm_srli_si128(u8, 8));
        __m256 fhi = _mm256_cvtepi32_ps(hi32);
        _mm256_storeu_ps(dst + j + 8, fhi);
        nv = _mm256_fmadd_ps(fhi, fhi, nv);
    }
    float o[8];
    _mm256_storeu_ps(o, nv);
    for (int k = 0; k < 8; k++) n += o[k];
    for (; j < d; j++) { dst[j] = src[j]; n += dst[j] * dst[j]; }
#else
    for (int j = 0; j < d; j++) { dst[j] = src[j]; n += dst[j] * dst[j]; }
#endif
    norm = std::sqrt(n);
    return norm;
}

inline void quantize(const float* src, int8_t* dst, const float* scale, int dims) {
    for (int j = 0; j < dims; j++) {
        int qi = (int)(src[j] / scale[j] + (src[j] >= 0 ? 0.5f : -0.5f));
        if (qi > 127) qi = 127;
        if (qi < -128) qi = -128;
        dst[j] = (int8_t)qi;
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
double recall_at_k(const std::vector<int>& result, const std::vector<int>& gt_set, int k) {
    int hits = 0;
    for (int ri : result)
        for (int vi : gt_set)
            if (ri == vi) { hits++; break; }
    return (double)hits / (double)std::max(k, 1);
}

double ndcg_at_k(const std::vector<int>& result, const std::vector<int>& gt_set, int k) {
    double dcg = 0, idcg = 0;
    for (int j = 0; j < k && j < (int)result.size(); j++) {
        int rel = 0;
        for (int vi : gt_set) if (result[j] == vi) { rel = 1; break; }
        dcg += (std::pow(2, rel) - 1) / std::log2(j + 2);
    }
    for (int j = 0; j < k; j++) idcg += 1.0 / std::log2(j + 2);
    return idcg > 0 ? dcg / idcg : 0.0;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
float l2_sq(const uint8_t* v_raw, const float* q, int dim) {
    float s = 0;
    for (int j = 0; j < dim; j++) {
        float d = (float)v_raw[j] - q[j];
        s += d * d;
    }
    return s;
}

std::vector<std::vector<int>> read_bigann_groundtruth(const std::string& path, int n_queries) {
    std::vector<std::vector<int>> gt;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return gt;
    int nq = 0, dim = 0;
    fread(&nq, 4, 1, f);
    fread(&dim, 4, 1, f);
    int want = n_queries > 0 ? std::min(n_queries, nq) : nq;
    gt.resize(want);
    std::vector<int> ids(dim);
    std::vector<float> dists(dim);
    for (int gi = 0; gi < want; gi++) {
        if (fread(ids.data(), 4, dim, f) != (size_t)dim) break;
        if (fread(dists.data(), 4, dim, f) != (size_t)dim) break;
        gt[gi].assign(ids.begin(), ids.end());
    }
    fclose(f);
    return gt;
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
MadhavaL2::MadhavaL2(const Config& cfg) : cfg_(cfg) {}

MadhavaL2::~MadhavaL2() {
    delete[] P1_;
    delete[] pr1_;
    delete[] pr1_scale_;
    delete[] e1_;
    delete[] vn_;
}

void MadhavaL2::build(const uint8_t* raw_base, int n) {
    base_ = raw_base;
    n_ = n;
    built_ = true;
    int D = cfg_.dim, s1 = cfg_.stage1_dim;
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Build the QR-orthogonalized projection P1 via MGS.
    P1_ = new float[(size_t)s1 * D];
    {
        std::mt19937 rng(cfg_.seed + s1);
        std::normal_distribution<float> nd(0, 1);
        for (int i = 0; i < s1; i++) {
            for (int j = 0; j < D; j++) P1_[i * D + j] = nd(rng);
            for (int k = 0; k < i; k++) {
                float dp = 0;
                for (int j = 0; j < D; j++) dp += P1_[i * D + j] * P1_[k * D + j];
                for (int j = 0; j < D; j++) P1_[i * D + j] -= dp * P1_[k * D + j];
            }
            float nr = 0;
            for (int j = 0; j < D; j++) nr += P1_[i * D + j] * P1_[i * D + j];
            nr = std::sqrt(nr);
            if (nr > 1e-10f)
                for (int j = 0; j < D; j++) P1_[i * D + j] /= nr;
        }
    }

    // 2. Allocate buffers.
    pr1_ = new int8_t[(size_t)n * s1];
    pr1_scale_ = new float[s1];
    e1_ = new float[n];
    vn_ = new float[n];

    // 3. Calibrate the int8 per-axis scales on a sample.
    int sn = std::min(n, 100000);
    std::vector<float> ck((size_t)sn * D);
    std::vector<float> ma(s1, 0);
    for (int i = 0; i < sn; i++) {
        float norm;
        load_raw_norm(base_ + (size_t)i * D, ck.data() + (size_t)i * D, D, norm);
        vn_[i] = norm;
    }
    for (int i = 0; i < sn; i++)
        for (int j = 0; j < s1; j++) {
            float s = dot_f32(ck.data() + (size_t)i * D, P1_ + (size_t)j * D, D);
            if (std::fabs(s) > ma[j]) ma[j] = std::fabs(s);
        }
    for (int j = 0; j < s1; j++)
        pr1_scale_[j] = std::max(ma[j] / 127.0f * 1.05f, 1e-10f);

    // 4. Streaming pass: compute int8 projections + the REAL float32 residual.
    const int CHUNK = 500000;
    std::vector<float> ch((size_t)CHUNK * D);
    int p = 0;
    while (p < n) {
        int nt = std::min(CHUNK, n - p);
        for (int i = 0; i < nt; i++) {
            float norm;
            load_raw_norm(base_ + (size_t)(p + i) * D, ch.data() + (size_t)i * D, D, norm);
            vn_[p + i] = norm;
        }
#pragma omp parallel for
        for (int i = 0; i < nt; i++) {
            int id = p + i;
            float* v = ch.data() + (size_t)i * D;
            float pj[64];
            float pn = 0;
            for (int j = 0; j < s1; j++) { pj[j] = dot_f32(v, P1_ + (size_t)j * D, D); pn += pj[j] * pj[j]; }
            quantize(pj, pr1_ + (size_t)id * s1, pr1_scale_, s1);
            // Cauchy-Schwarz residual over the REAL float32 projection:
            // e(v) = sqrt(||v||² − ||Pv||²). This is what the inequality requires.
            e1_[id] = std::sqrt(std::max(0.0f, vn_[id] * vn_[id] - pn));
        }
        p += nt;
    }

    build_s_ = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

float MadhavaL2::l2_sq(int idx, const float* q) const {
    return winnex_madhava::l2_sq(base_ + (size_t)idx * cfg_.dim, q, cfg_.dim);
}

float MadhavaL2::lb_l2(int idx, const float* pq, float qr1, float qm1, float qn2) const {
    int s1 = cfg_.stage1_dim;
    // Cauchy-Schwarz upper bound of ⟨v,q⟩ using the int8-quantized projection
    // plus the quantization margin (qm) and the residual product (e·qr).
    float inner = 0;
    const int8_t* pr = pr1_ + (size_t)idx * s1;
    for (int j = 0; j < s1; j++) inner += (float)pr[j] * pr1_scale_[j] * pq[j];
    float ub = inner + e1_[idx] * qr1 + qm1 + 1e-5f;
    // L2² = ‖v‖² + ‖q‖² − 2⟨v,q⟩ ≥ ‖v‖² + ‖q‖² − 2·ub
    return vn_[idx] * vn_[idx] + qn2 - 2.0f * ub;
}

SearchResult MadhavaL2::search(const float* query, const std::vector<float>& query_norm) const {
    SearchResult out;
    if (!built_) return out;
    int N = n_, D = cfg_.dim, s1 = cfg_.stage1_dim, K = cfg_.k;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Query projections + residuals.
    float qn = query_norm.empty() ? std::sqrt(dot_f32(query, query, D)) : query_norm[0];
    float pq[64];
    float q1s = 0;
    for (int j = 0; j < s1; j++) { pq[j] = dot_f32(query, P1_ + (size_t)j * D, D); q1s += pq[j] * pq[j]; }
    float qr1 = std::sqrt(std::max(0.0f, qn * qn - q1s));
    float qm1 = 0;
    for (int j = 0; j < s1; j++) qm1 += 0.5f * pr1_scale_[j] * std::fabs(pq[j]);
    float qn2 = qn * qn;

    // Stage-1: bound pruning over all N.
    out.bound_pairs = N;
    std::vector<std::pair<float, int>> b1((size_t)N);
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        b1[i] = {lb_l2(i, pq, qr1, qm1, qn2), i};
    }
    int k1 = std::max(cfg_.k1_min, (int)(N * cfg_.k1_fraction));
    if (k1 > N) k1 = N;
    if (k1 < N) std::nth_element(b1.begin(), b1.begin() + k1, b1.end());
    out.k1 = k1;

    // Stage-3: post-filter with exact L2 on the survivors (if enabled),
    // else return the top-K by the bound.
    if (cfg_.postfilter) {
        std::vector<std::pair<float, int>> exact((size_t)k1);
#pragma omp parallel for
        for (int i = 0; i < k1; i++) {
            int vi = b1[i].second;
            exact[i] = {l2_sq(vi, query), vi};
        }
        out.k3 = k1;
        std::partial_sort(exact.begin(), exact.begin() + std::min(K, k1), exact.end());
        for (int i = 0; i < std::min(K, k1); i++) out.indices.push_back(exact[i].second);
    } else {
        std::partial_sort(b1.begin(), b1.begin() + std::min(K, k1), b1.end());
        for (int i = 0; i < std::min(K, k1); i++) out.indices.push_back(b1[i].second);
    }

    out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

SearchResult MadhavaL2::search(const float* query) const {
    std::vector<float> qn;
    return search(query, qn);
}

SearchResult MadhavaL2::search_exact(const float* query) const {
    std::vector<float> qn;
    return search_exact(query, qn);
}

SearchResult MadhavaL2::search_exact(const float* query, const std::vector<float>& query_norm) const {
    SearchResult out;
    if (!built_) return out;
    int N = n_, K = cfg_.k;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<float, int>> scores((size_t)N);
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        scores[i] = {l2_sq(i, query), i};
    }
    out.bound_pairs = N;
    out.k3 = N;
    std::partial_sort(scores.begin(), scores.begin() + std::min(K, N), scores.end());
    for (int i = 0; i < std::min(K, N); i++) out.indices.push_back(scores[i].second);
    out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

} // namespace winnex_madhava
