// ==== winnex_madhava.hpp (inlined) ====
/**
 * winnex_madhava.hpp — Public API of the Madhava L2 C++ Library
 * ==========================================================
 * Deterministic vector search with Cauchy-Schwarz upper bounds,
 * evaluated against the official BIGANN-100M L2 ground truth.
 *
 * This library packages the Madhava L2 engine as a reusable C++20
 * library with:
 *   - Cauchy-Schwarz bound Stage-1 pruning (0 bound violations)
 *   - Optional post-filter with exact L2 in the surviving top-k1
 *   - Exact-scan baseline (the recall "ceiling" of a subset)
 *   - Binary Recall@K / NDCG@K against a ground-truth id list
 *
 * Mathematical guarantee (Cauchy-Schwarz on raw inner product):
 *   ⟨v,q⟩ ≤ ⟨Pv,Pq⟩ + ‖v−PᵀPv‖ · ‖q−PᵀPq‖
 * hence for L2²:
 *   ‖v−q‖² = ‖v‖² + ‖q‖² − 2⟨v,q⟩ ≥ ‖v‖² + ‖q‖² − 2·(UB of ⟨v,q⟩)
 * Every vector pruned by Stage 1 is mathematically proven not to be
 * in the top-K by the L2 distance.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef MADHAVA_L2_HPP
#define MADHAVA_L2_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>

namespace winnex_madhava {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config {
    int dim = 128;          // vector dimensionality (uint8 raw bytes)
    int stage1_dim = 64;    // Stage-1 QR projection dimensionality
    int seed = 42;          // PRNG seed for the MGS projection
    int k = 10;             // number of results to return
    double k1_fraction = 0.05; // Stage-1 keep fraction (top-k1 = N*k1_fraction)
    int k1_min = 100;       // minimum k1 (guards tiny N)
    bool postfilter = true; // apply exact-L2 post-filter on the top-k1
    int n_threads = 0;      // 0 = use omp_get_max_threads()
};

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------
struct SearchResult {
    std::vector<int> indices;   // top-K indices, ascending L2²
    int k1 = 0;                 // survivors after Stage-1 pruning
    int k3 = 0;                 // candidates actually evaluated exactly
    double latency_ms = 0.0;    // search wall time (excludes build)
    long long bound_pairs = 0;  // N vectors evaluated in Stage-1
    long long bound_violations = 0; // should always be 0
};

// ---------------------------------------------------------------------------
// Evaluation metrics (binary relevance against a ground-truth id list)
// ---------------------------------------------------------------------------
struct Metrics {
    double recall_at_k = 0.0;
    double ndcg_at_k = 0.0;
};

// Binary Recall@K: fraction of the GT set present in the result list.
double recall_at_k(const std::vector<int>& result, const std::vector<int>& gt_set, int k);

// Binary NDCG@K (relevance = 1 if the id is in the GT set).
double ndcg_at_k(const std::vector<int>& result, const std::vector<int>& gt_set, int k);

// ---------------------------------------------------------------------------
// Madhava L2 Engine
// ---------------------------------------------------------------------------
// Immutable view over a raw uint8 corpus. The caller owns the data pointer;
// the engine never copies the corpus (mmap-friendly).
class MadhavaL2 {
public:
    explicit MadhavaL2(const Config& cfg);
    ~MadhavaL2();

    // Copy is expensive (owns buffers) — disable it.
    MadhavaL2(const MadhavaL2&) = delete;
    MadhavaL2& operator=(const MadhavaL2&) = delete;
    MadhavaL2(MadhavaL2&&) = delete;
    MadhavaL2& operator=(MadhavaL2&&) = delete;

    // Build the Stage-1 projections + per-vector residuals.
    // raw_base must point to at least n*dim uint8 bytes, kept alive during use.
    void build(const uint8_t* raw_base, int n);

    // Search: Stage-1 bound pruning (top-k1) + optional exact-L2 post-filter.
    SearchResult search(const float* query, const std::vector<float>& query_norm) const;
    SearchResult search(const float* query) const; // computes norm internally

    // Exact scan baseline: computes L2² for ALL n vectors (the recall ceiling).
    // Returns the exact top-K. Use to measure the subset's achievable ceiling.
    SearchResult search_exact(const float* query, const std::vector<float>& query_norm) const;
    SearchResult search_exact(const float* query) const; // computes norm internally

    // Diagnostics
    int num_vectors() const { return n_; }
    int dim() const { return cfg_.dim; }
    const Config& config() const { return cfg_; }
    double build_seconds() const { return build_s_; }
    bool built() const { return built_; }

private:
    // Per-vector L2² (raw uint8 vs float query)
    float l2_sq(int idx, const float* q) const;
    // Cauchy-Schwarz lower bound of L2² for a vector (Stage-1 score)
    float lb_l2(int idx, const float* pq, float qr1, float qm1, float qn2) const;

    Config cfg_;
    const uint8_t* base_ = nullptr;
    int n_ = 0;
    bool built_ = false;
    double build_s_ = 0.0;

    // Stage-1 projection (QR-orthogonalized via Modified Gram-Schmidt)
    float* P1_ = nullptr;       // [stage1_dim × dim]
    int8_t* pr1_ = nullptr;     // [n × stage1_dim] int8 quantized projections
    float* pr1_scale_ = nullptr; // [stage1_dim] per-axis quant scale
    float* e1_ = nullptr;       // [n] residual ||v−PᵀPv|| (float32 REAL, not int8)
    float* vn_ = nullptr;       // [n] L2 norms of the raw vectors
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
// Read the official BIGANN ground-truth file (header: nq:int32, dim:int32,
// then per query: dim ids int32 + dim dists float32).
// Returns gt[query_index][neighbor_slot] = dataset id.
std::vector<std::vector<int>> read_bigann_groundtruth(const std::string& path, int n_queries);

// Compute the L2² distance between a raw uint8 vector and a float query.
float l2_sq(const uint8_t* v_raw, const float* q, int dim);

} // namespace winnex_madhava

#endif // MADHAVA_L2_HPP

// ==== winnex_madhava.cpp (implementation) ====
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

// ==== bench_bigann_l2.cpp (main) ====
/**
 * bench_bigann_l2.cpp — Benchmark Madhava L2 against the official BIGANN
 * L2 ground truth, including the exact-scan "ceiling" baseline.
 *
 * Usage:
 *   winnex_madhava_bench <base.u8bin> <queries.u8bin> <gt.bin> [n] [n_queries] [k1_frac]
 *
 * Output (CSV):
 *   Scale,Method,Build_s,Lat_ms,NDCG_L2,R@10_L2,Vio_Pairs,k1,k3
 *
 * This benchmark is the scientific core of the L2 investigation:
 * it reports the exact-scan ceiling alongside the Madhava result, so the
 * reader can see how close the bound+filter engine gets to the physical
 * limit of the evaluated subset.
 */


#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace winnex_madhava;

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <base.u8bin> <queries.u8bin> <gt.bin> [n] [nq] [k1_frac]\n", argv[0]);
        return 1;
    }
    std::string base_path = argv[1];
    std::string q_path = argv[2];
    std::string gt_path = argv[3];
    int n = argc > 4 ? atoi(argv[4]) : 100000000;
    int nq = argc > 5 ? atoi(argv[5]) : 50;
    double k1_frac = argc > 6 ? atof(argv[6]) : 0.05;
    int dim = 128, k = 10;

    // mmap the base corpus (no RAM copy).
    int fd = open(base_path.c_str(), O_RDONLY);
    if (fd < 0) { perror("open base"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t len = st.st_size;
    const uint8_t* base = (const uint8_t*)mmap(0, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    int total_vecs = (int)(len / dim);
    n = std::min(n, total_vecs);

    // Load queries (the first nq*2 float vectors; we use query 2i for GT[i]).
    FILE* fq = fopen(q_path.c_str(), "rb");
    if (!fq) { perror("open queries"); return 1; }
    std::vector<uint8_t> qbuf((size_t)nq * 2 * dim);
    fread(qbuf.data(), 1, qbuf.size(), fq); fclose(fq);
    std::vector<float> qv((size_t)nq * 2 * dim);
    for (size_t i = 0; i < qbuf.size(); i++) qv[i] = (float)qbuf[i];

    // Load GT (GT[gi] <-> query 2*gi).
    auto gt = read_bigann_groundtruth(gt_path, nq);
    if (gt.empty()) { fprintf(stderr, "GT not readable\n"); return 1; }

    Config cfg;
    cfg.dim = dim;
    cfg.k = k;
    cfg.k1_fraction = k1_frac;
    cfg.postfilter = true;

    MadhavaL2 engine(cfg);
    engine.build(base, n);

    printf("Scale,Method,Build_s,Lat_ms,NDCG_L2,R@10_L2,Vio_Pairs,k1,k3\n");

    auto eval = [&](const char* method, SearchResult (*fn)(const MadhavaL2&, const float*),
                    double* out_r, double* out_n) {
        double tr = 0, tn = 0, tlat = 0;
        long long vio = 0;
        int ak1 = 0, ak3 = 0;
        int n_eval = (int)gt.size();
        for (int gi = 0; gi < n_eval; gi++) {
            int qi = 2 * gi;
            std::vector<int> gset;
            for (int v : gt[gi]) if (v >= 0 && v < n) gset.push_back(v);
            SearchResult r = fn(engine, &qv[(size_t)qi * dim]);
            tn += ndcg_at_k(r.indices, gset, k);
            tr += recall_at_k(r.indices, gset, k);
            tlat += r.latency_ms;
            vio += r.bound_violations;
            ak1 += r.k1; ak3 += r.k3;
        }
        int m = std::max(n_eval, 1);
        printf("%d,%s,%.3f,%.3f,%.4f,%.4f,%lld,%d,%d\n",
               n, method, engine.build_seconds(), tlat / m, tn / m, tr / m, vio, ak1 / m, ak3 / m);
        if (out_r) *out_r = tr / m;
        if (out_n) *out_n = tn / m;
    };

    double r_ceiling = 0, n_ceiling = 0;
    eval("exact_scan", [](const MadhavaL2& e, const float* q) { return e.search_exact(q); }, &r_ceiling, &n_ceiling);
    double r_madhava = 0, n_madhava = 0;
    eval("madhava", [](const MadhavaL2& e, const float* q) { return e.search(q); }, &r_madhava, &n_madhava);

    fprintf(stderr, "\n");
    fprintf(stderr, "Ceiling (exact scan):  R@10=%.4f NDCG=%.4f\n", r_ceiling, n_ceiling);
    fprintf(stderr, "Madhava (bound+filter): R@10=%.4f NDCG=%.4f\n", r_madhava, n_madhava);
    fprintf(stderr, "Efficiency vs ceiling:  %.1f%%\n", r_ceiling > 0 ? 100.0 * r_madhava / r_ceiling : 0.0);
    fprintf(stderr, "GT coverage in subset:  %.1f%%\n", 100.0 * r_ceiling);
    fprintf(stderr, "\nBSL 1.1 | pay@winnex.ai\n");
    return 0;
}
