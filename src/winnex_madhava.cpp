/**
 * winnex_madhava.cpp — implementation of the Winnex Madhava Engine
 * =================================================================
 * Deterministic vector search with Cauchy-Schwarz bounds, parametrizable
 * across the Winnex stack:
 *
 *   - Metric 'cosine' (stack default): normalizes each raw uint8 vector to
 *     unit norm at build time, then works in inner-product space. This is
 *     what v17, Madhava-Sec, and the HMC v7 notebook do.
 *   - Metric 'l2': works on raw uint8, converts the inner-product upper bound
 *     into a lower bound of L2² (BIGANN-style).
 *   - Cascade [stage1, stage2]: Stage-1 wide bound B1 prunes to k1; if a
 *     Stage-2 projection is configured, a tighter bound B2 prunes to k2.
 *     Pruning ALWAYS uses the tightest available bound (stack FIX(1)).
 *   - Modulation: ranks survivors by B1 + α·(B2−B1) with
 *     α = sigmoid((e1−e2)/mean(e1)) — the error-backpropagation refinement.
 *     Never used for pruning; only for ranking (stack invariant).
 *   - Quantization: 'int8' keeps per-axis scales + margins; 'none' uses the
 *     float32 projection exactly (no quant margin).
 *   - Post-filter: exact metric re-score on the surviving set.
 *
 * BSL 1.1 | pay@winnex.ai
 */
#include "winnex_madhava/winnex_madhava.hpp"
#include "winnex_madhava/speed_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace winnex_madhava {

namespace {

// OpenMP thread helpers with a no-OpenMP fallback (compile without -fopenmp).
inline int wm_omp_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int wm_omp_thread_num() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

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

// M4 (v1.8.0): AVX2 dot of int8-quantized projections with per-axis float scale.
//   inner = Σ (int8[j] * scale[j] * pq[j])
// Processes 16 int8 (two AVX2 256-bit lanes) per iteration — the bound-scan
// hot loop. Falls back to scalar when AVX2/FMA is unavailable.
inline float dot_int8_scaled(const int8_t* a, const float* scale, const float* b, int d) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        // 16 int8 -> two groups of 8, sign-extended to 32-bit int, then float.
        __m128i u8lo = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i u8hi = _mm_srli_si128(u8lo, 8);
        __m256i i32lo = _mm256_cvtepi8_epi32(u8lo);
        __m256i i32hi = _mm256_cvtepi8_epi32(u8hi);
        __m256 flo = _mm256_cvtepi32_ps(i32lo);
        __m256 fhi = _mm256_cvtepi32_ps(i32hi);
        // flo * scale[i..i+8] * b[i..i+8]
        __m256 slo = _mm256_loadu_ps(scale + i);
        __m256 shi = _mm256_loadu_ps(scale + i + 8);
        __m256 blo = _mm256_loadu_ps(b + i);
        __m256 bhi = _mm256_loadu_ps(b + i + 8);
        s = _mm256_fmadd_ps(flo, _mm256_mul_ps(slo, blo), s);
        s = _mm256_fmadd_ps(fhi, _mm256_mul_ps(shi, bhi), s);
    }
    float o[8];
    _mm256_storeu_ps(o, s);
    float r = o[0] + o[1] + o[2] + o[3] + o[4] + o[5] + o[6] + o[7];
    for (; i < d; i++) r += (float)a[i] * scale[i] * b[i];
    return r;
#else
    float r = 0;
    for (int i = 0; i < d; i++) r += (float)a[i] * scale[i] * b[i];
    return r;
#endif
}

// M4 (v1.8.0): AVX2 L2² between a raw uint8 vector and a float32 query.
//   s = Σ (v[j] - q[j])²
// Processes 16 uint8 (two lanes) per iteration. Fallback to scalar.
inline float l2_sq_avx(const uint8_t* v_raw, const float* q, int dim) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s = _mm256_setzero_ps();
    int j = 0;
    for (; j + 16 <= dim; j += 16) {
        __m128i u8 = _mm_loadu_si128((const __m128i*)(v_raw + j));
        __m256i lo32 = _mm256_cvtepu8_epi32(u8);
        __m256i hi32 = _mm256_cvtepu8_epi32(_mm_srli_si128(u8, 8));
        __m256 flo = _mm256_cvtepi32_ps(lo32);
        __m256 fhi = _mm256_cvtepi32_ps(hi32);
        __m256 qlo = _mm256_loadu_ps(q + j);
        __m256 qhi = _mm256_loadu_ps(q + j + 8);
        __m256 dlo = _mm256_sub_ps(flo, qlo);
        __m256 dhi = _mm256_sub_ps(fhi, qhi);
        s = _mm256_fmadd_ps(dlo, dlo, s);
        s = _mm256_fmadd_ps(dhi, dhi, s);
    }
    float o[8];
    _mm256_storeu_ps(o, s);
    float r = o[0] + o[1] + o[2] + o[3] + o[4] + o[5] + o[6] + o[7];
    for (; j < dim; j++) { float d = (float)v_raw[j] - q[j]; r += d * d; }
    return r;
#else
    float s = 0;
    for (int j = 0; j < dim; j++) { float d = (float)v_raw[j] - q[j]; s += d * d; }
    return s;
#endif
}

// Load a raw uint8 vector into float32, computing its L2 norm.
// Optionally normalizes to unit norm (cosine metric).
inline float load_raw(const uint8_t* src, float* dst, int d, bool normalize) {
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
    n = std::sqrt(n);
    if (normalize && n > 1e-10f) {
        float inv = 1.0f / n;
        for (int j = 0; j < d; j++) dst[j] *= inv;
    }
    return n;
}

inline void quantize(const float* src, int8_t* dst, const float* scale, int dims) {
    for (int j = 0; j < dims; j++) {
        int qi = (int)(src[j] / scale[j] + (src[j] >= 0 ? 0.5f : -0.5f));
        if (qi > 127) qi = 127;
        if (qi < -128) qi = -128;
        dst[j] = (int8_t)qi;
    }
}

// Build an orthonormal projection (s rows × D cols) aligned to the principal
// directions of the corpus — the "UB Width" mode (BasisMode::PCACorpus).
//
// The bound tightness is driven by the residual e(v) = sqrt(||v||^2 − ||P v||^2).
// A random projection (the default) leaves e(v) ≈ sqrt(1 − s/D), which is large
// at D = 1536 and degenerates the bound to exhaustive search. Aligning P to the
// corpus' top-s principal directions shrinks e(v) to the manifold residual,
// restoring pruning power at high dimension — while remaining orthonormal, so
// the Cauchy-Schwarz bound stays valid in the ORIGINAL space (0 violations by
// construction). This is the same math the X-Factor applies to embed_tokens,
// applied here to the search corpus.
//
// Returns true on success; false when the covariance is degenerate (in which
// case the caller falls back to a random basis).
//
// base is either uint8 (default mode) or float32 (the UB-Width/float32 path).
// A uint8 quantized corpus destroys the fine manifold structure (eigenvalues
// collapse to ~1e-4, and the principal basis no longer reflects the embedding
// geometry). For embeddings (Qwen/BERT/DeepSeek, d~1536) the caller MUST use
// the float32 path so the PCA basis aligns with the real data manifold.
bool build_pca_basis(const float* base_f32, int n, int D, int s, int seed,
                     int sample_cap, bool normalize, int iterations,
                     float* P /* s×D row-major */) {
    if (s <= 0 || s > D) return false;
    int sample = std::min(n, sample_cap);
    if (sample < 1) return false;
    if (iterations < 1) iterations = 1;

    // 1. Load a subsample of the corpus, optionally L2-normalized.
    //    Subsample read: the original code gathered base_f32[rng()%n] — a random
    //    cache-hostile gather over the full corpus (61MB at n=10k,d=1536).
    //    Reading CONTIGUOUS blocks (the first `sample` rows) is cache-friendly
    //    and is an i.i.d. subsample when the corpus is arbitrarily ordered
    //    (embeddings are order-independent for the covariance). Determinism:
    //    same seed/corpus -> same first `sample` rows -> same basis.
    std::mt19937 rng(seed);
    std::vector<float> A((size_t)sample * D);
    for (int i = 0; i < sample; i++) {
        const float* v = base_f32 + (size_t)i * D;
        float* dst = A.data() + (size_t)i * D;
        float nn = 0.0f;
        for (int j = 0; j < D; j++) { dst[j] = v[j]; nn += dst[j] * dst[j]; }
        if (normalize && nn > 1e-10f) {
            float inv = 1.0f / std::sqrt(nn);
            for (int j = 0; j < D; j++) dst[j] *= inv;
        }
    }

    // 2. Empirical covariance, NON-CENTERED: C = (1/sample)·Σ a·aᵀ.
    //    The bound e(v) = sqrt(1 − ‖Pv‖²) needs v and P in the same space;
    //    centering removes the corpus' dominant direction and widens the bound.
    //    NOTE (1.9.5 benchmark): a matrix-free Aᵀ(A·v) form was tried and
    //    REVERTED — it regresses low/mid dim (d=128: 0.5s -> 11.4s, measured on
    //    the Kaggle runtime) because O(2·sample·D·s·iters) >> O(D²·sample) when
    //    sample=10k > D. The direct covariance is the right choice across the
    //    supported dim range (0.2s at d=128, 4.1s at d=1536 on Kaggle).
    std::vector<float> C((size_t)D * D, 0.0f);
    for (int i = 0; i < sample; i++) {
        const float* a = A.data() + (size_t)i * D;
        for (int r = 0; r < D; r++) {
            float ar = a[r];
            float* Cr = C.data() + (size_t)r * D;
            for (int c = 0; c < D; c++) Cr[c] += ar * a[c];
        }
    }
    float invs = 1.0f / (float)sample;
    for (size_t x = 0; x < (size_t)D * D; x++) C[x] *= invs;
    double trace = 0.0;
    for (int j = 0; j < D; j++) trace += (double)C[(size_t)j * D + j];
    if (trace < 1e-12) return false;

    // 3. Top-s eigenvectors by POWER ITERATION with deflation + MGS
    //    re-orthogonalization (O(D²·s·iters)) — the X-Factor's method.
    //    Iteration cap (1.9.5): the power iteration converges in ~10-30 steps
    //    for the dominant directions (subspace sim = 1.0000 to the 200-step
    //    result, measured); the remaining steps only refine the individual
    //    eigenvector without changing the bound e(v) = sqrt(1 - ||Pv||²).
    std::vector<float> W(C);   // working copy, deflated in place
    std::vector<float> basis((size_t)s * D, 0.0f);  // row-major [s][D]
    std::mt19937 rng_pow(seed + 0x5EED);
    std::normal_distribution<float> nd(0, 1);
    for (int k = 0; k < s; ++k) {
        // Deterministic random start (component in every direction) so the
        // iteration cannot be trapped in a null subspace.
        std::vector<float> v(D, 0.0f);
        float n0 = 0.0f;
        for (int j = 0; j < D; ++j) { float x = nd(rng_pow); v[j] = x; n0 += x * x; }
        n0 = std::sqrt(n0) + 1e-12f;
        for (float& x : v) x /= n0;

        double lambda = 0.0;
        for (int it = 0; it < iterations; ++it) {
            // w = C·v
            std::vector<float> work(D, 0.0f);
            for (int i = 0; i < D; ++i) {
                float s_ = 0.0f;
                const float* Ci = W.data() + (size_t)i * D;
                for (int j = 0; j < D; ++j) s_ += Ci[j] * v[j];
                work[i] = s_;
            }
            // MGS re-orthogonalization against previously found directions.
            // Serial: parallelizing this O(k·D) inner loop with the environment's
            // OMP_NUM_THREADS spawns a 28-thread team per 1536-iteration loop
            // and the spawn overhead dominates at high thread counts.
            for (int j = 0; j < k; ++j) {
                const float* uj = basis.data() + (size_t)j * D;
                float dot = 0.0f;
                for (int i = 0; i < D; ++i) dot += work[i] * uj[i];
                for (int i = 0; i < D; ++i) work[i] -= dot * uj[i];
            }
            // Rayleigh quotient λ = vᵀCv (vᵀv = 1) — the bound.
            double new_lambda = 0.0;
            for (int i = 0; i < D; ++i) new_lambda += (double)v[i] * work[i];
            // Normalize.
            float nw = 0.0f;
            for (int i = 0; i < D; ++i) nw += work[i] * work[i];
            nw = std::sqrt(nw) + 1e-12f;
            if (nw < 1e-10f) break;  // remaining direction exhausted
            for (int i = 0; i < D; ++i) v[i] = work[i] / nw;
            if (std::fabs(new_lambda - lambda) < 1e-8 * (1.0 + std::fabs(new_lambda))) {
                lambda = new_lambda;
                break;
            }
            lambda = new_lambda;
        }
        // Store eigenvector.
        for (int i = 0; i < D; ++i) basis[(size_t)k * D + i] = v[i];
        // Deflate: W ← W − λ·v·vᵀ.
        for (int i = 0; i < D; ++i) {
            float* Wi = W.data() + (size_t)i * D;
            const float vi = v[i];
            for (int j = 0; j < D; ++j) Wi[j] -= (float)(lambda * vi * v[j]);
        }
    }

    // 4. Emit the top-s eigenvectors as s orthonormal rows (already
    //    orthonormal by the MGS re-orthogonalization).
    for (int r = 0; r < s; ++r)
        for (int j = 0; j < D; ++j) P[(size_t)r * D + j] = basis[(size_t)r * D + j];
    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
double recall_at_k(const std::vector<int>& result, const std::vector<int>& gt_set, int k) {
    // Recall@K robusto: intersecta com TODO o conjunto relevante (gt_set, que
    // o chamador já filtrou para o subset), e divide por min(k, |gt_set|).
    // Assim, se o GT tem menos de k relevantes no subset, um scan exato perfeito
    // ainda atinge recall=1.0 (não é penalizado por |gt_set| < k).
    int hits = 0;
    for (int ri : result)
        for (int vi : gt_set)
            if (ri == vi) { hits++; break; }
    int denom = std::min(k, (int)gt_set.size());
    return denom > 0 ? (double)hits / (double)denom : 0.0;
}

double ndcg_at_k(const std::vector<int>& result, const std::vector<int>& gt_set, int k) {
    // NDCG@K robusto: o denominador (idcg) usa min(k, |gt_set|) — o mesmo
    // princípio do recall — para não penalizar queries com poucos relevantes.
    double dcg = 0, idcg = 0;
    int top = std::min(k, (int)gt_set.size());
    for (int j = 0; j < top && j < (int)result.size(); j++) {
        int rel = 0;
        for (int vi : gt_set) if (result[j] == vi) { rel = 1; break; }
        dcg += (std::pow(2, rel) - 1) / std::log2(j + 2);
    }
    for (int j = 0; j < top; j++) idcg += 1.0 / std::log2(j + 2);
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
    if (fread(&nq, 4, 1, f) != 1 || fread(&dim, 4, 1, f) != 1) {
        fclose(f); return gt;
    }
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
MadhavaL2::MadhavaL2(const Config& cfg) : cfg_(cfg) {
    // Sanity: stage2_dim must be > stage1_dim for a meaningful cascade.
    if (cfg_.stage2_dim > 0 && cfg_.stage2_dim <= cfg_.stage1_dim) {
        // Keep the engine functional: promote stage2 to stage1+1 (bounded by dim).
        cfg_.stage2_dim = std::min(cfg_.stage1_dim + 1, cfg_.dim);
    }
}

void MadhavaL2::set_basis(const float* P1, const float* P2) {
    if (!built_ || !P1) return;
    const int D = cfg_.dim, s1 = cfg_.stage1_dim, s2 = cfg_.stage2_dim;
    std::memcpy(P1_, P1, (size_t)s1 * D * sizeof(float));
    if (s2 > 0 && P2) {
        std::memcpy(P2_, P2, (size_t)s2 * D * sizeof(float));
    }
    // The bound uses BOTH the projection basis (P1_/P2_) AND the cached
    // per-vector projections (pr1_f_/pr2_f_) computed at build time. When the
    // basis changes, the cached projections MUST be recomputed — otherwise the
    // bound ub_raw uses projections from the OLD basis while the query projects
    // onto the NEW one, producing a garbage bound (measured recall 0.05).
    // Recompute projections + residuals consistently over the REAL float32
    // corpus (corpus_f32_ when available, else the uint8 base re-normalized).
    const bool normalize = (cfg_.metric == Metric::Cosine) && cfg_.normalize_input;
    std::vector<float> buf((size_t)1 * D);
    // SCAN INT8: a basis change changes the projection magnitudes. The int8
    // scale (calibrated on the OLD basis during build) may saturate/clip on the
    // NEW basis — measured: pca_corpus + scan_int8 dropped recall to 0.85. So
    // when scan_int8 is on, FIRST recalibrate the per-axis scale on the new
    // basis (max |projection| over the corpus), THEN quantize.
    bool recalib_scale = cfg_.scan_int8 && pr1_ != nullptr;
    std::vector<float> ma1(recalib_scale ? s1 : 0, 0.0f);
    std::vector<float> ma2(recalib_scale && s2 > 0 ? s2 : 0, 0.0f);
    if (recalib_scale) {
        // Pass 1: project all docs on the NEW basis, find per-axis max |.|.
        for (int i = 0; i < n_; i++) {
            const float* base_v;
            if (corpus_f32_ != nullptr) base_v = corpus_f32_ + (size_t)i * D;
            else { for (int j = 0; j < D; j++) buf[j] = (float)base_[(size_t)i * D + j]; base_v = buf.data(); }
            for (int j = 0; j < s1; j++) {
                float d = dot_f32(base_v, P1_ + (size_t)j * D, D);
                if (std::fabs(d) > ma1[j]) ma1[j] = std::fabs(d);
            }
            if (s2 > 0) for (int j = 0; j < s2; j++) {
                float d = dot_f32(base_v, P2_ + (size_t)j * D, D);
                if (std::fabs(d) > ma2[j]) ma2[j] = std::fabs(d);
            }
        }
        for (int j = 0; j < s1; j++) pr1_scale_[j] = std::max(ma1[j] / 127.0f * 1.05f, 1e-10f);
        if (s2 > 0) for (int j = 0; j < s2; j++) pr2_scale_[j] = std::max(ma2[j] / 127.0f * 1.05f, 1e-10f);
    }
    // Pass 2: recompute projections + residuals, quantize int8 with the fresh scale.
    for (int i = 0; i < n_; i++) {
        const float* base_v;
        if (corpus_f32_ != nullptr) base_v = corpus_f32_ + (size_t)i * D;
        else { for (int j = 0; j < D; j++) buf[j] = (float)base_[(size_t)i * D + j]; base_v = buf.data(); }
        float nn = 0.0f;
        for (int j = 0; j < D; j++) nn += base_v[j] * base_v[j];
        float vn2 = normalize ? 1.0f : nn;
        // recompute Stage-1 projection + residual
        float pn1 = 0.0f;
        float pj1[256];
        for (int j = 0; j < s1; j++) {
            float d = dot_f32(base_v, P1_ + (size_t)j * D, D);
            pr1_f_[(size_t)i * s1 + j] = d;
            pj1[j] = d;
            pn1 += d * d;
        }
        e1_[i] = std::sqrt(std::max(0.0f, vn2 - pn1));
        // SCAN INT8: also refresh the int8 projections after a basis change.
        if (cfg_.scan_int8 && pr1_ != nullptr && pr1_scale_ != nullptr) {
            quantize(pj1, pr1_ + (size_t)i * s1, pr1_scale_, s1);
        }
        if (s2 > 0) {
            float pn2 = 0.0f;
            float pj2[256];
            for (int j = 0; j < s2; j++) {
                float d = dot_f32(base_v, P2_ + (size_t)j * D, D);
                pr2_f_[(size_t)i * s2 + j] = d;
                pj2[j] = d;
                pn2 += d * d;
            }
            e2_[i] = std::sqrt(std::max(0.0f, vn2 - pn2));
            if (cfg_.scan_int8 && pr2_ != nullptr && pr2_scale_ != nullptr) {
                quantize(pj2, pr2_ + (size_t)i * s2, pr2_scale_, s2);
            }
        }
    }
}

void MadhavaL2::build_float32(const float* raw_base, int n) {
    // Keep the float32 buffer alive for the PCA path (and to satisfy the
    // caller-owned-data contract, the caller keeps the array; we copy so the
    // basis construction below can use a stable pointer even for uint8 builds).
    const int D = cfg_.dim;
    if (corpus_f32_ == nullptr) {
        corpus_f32_ = new float[(size_t)n * D];
    }
    std::memcpy(corpus_f32_, raw_base, (size_t)n * D * sizeof(float));

    // The build() path expects uint8 for the metric evaluation (vn_, exact
    // cosine over uint8). For the float32 path we re-quantize to uint8 using
    // the CORRECT affine map for unit-norm embeddings:
    //
    //     uint8 = clamp( (v + 1) * 127.5 , 0, 255 )        (v ∈ [-1, 1])
    //
    // A raw truncation `(uint8_t)v` (or clamp(v, 0, 255)) is WRONG: a unit-norm
    // embedding has values in [-1, 1], and truncation maps nearly all of them
    // to 0 — the motor then sees all-zero vectors and recall collapses to 0
    // (measured). The affine map preserves the geometry. The PCA basis,
    // however, uses the REAL float32 in corpus_f32_ — that is the point of
    // this path.
    //
    // IMPORTANT: the uint8 buffer must outlive the engine (base_ references it
    // for search). Keep it as an owned member.
    delete[] corpus_u8_owned_;
    corpus_u8_owned_ = new uint8_t[(size_t)n * D];
    for (size_t i = 0; i < (size_t)n * D; i++) {
        float v = raw_base[i];
        float q = (v + 1.0f) * 127.5f;
        if (q < 0.0f) q = 0.0f;
        if (q > 255.0f) q = 255.0f;
        corpus_u8_owned_[i] = (uint8_t)q;
    }
    build(corpus_u8_owned_, n);
}

MadhavaL2::~MadhavaL2() {
    delete[] P1_;
    delete[] P2_;
    delete[] pr1_;
    delete[] pr2_;
    delete[] pr1_f_;
    delete[] pr2_f_;
    delete[] pr1_scale_;
    delete[] pr2_scale_;
    delete[] e1_;
    delete[] e2_;
    delete[] vn_;
    delete[] vn_eff_;
    delete[] corpus_f32_;
    delete[] corpus_u8_owned_;
    // Phase-3: release the GPU keepalive (frees the OpenCL context ref + the
    // upload-once projection buffers).
    if (gpu_keepalive_) {
        SpeedEngine* se = static_cast<SpeedEngine*>(gpu_keepalive_);
        delete se;
        gpu_keepalive_ = nullptr;
    }
    gpu_stage1_ready_ = false;
}

// Phase-3 GPU Stage-1 enable (see header). Must run AFTER build (pr1_f_/e1_/
// vn_eff_ exist). Constructs a keepalive SpeedEngine to bring up the OpenCL
// context, then uploads the projection buffers once via madhava_gpu_stage1_init.
bool MadhavaL2::enable_gpu_stage1(const std::string& opencl_lib) {
    if (!built_ || !pr1_f_ || !e1_ || !vn_eff_) return false;
    if (gpu_stage1_ready_) return true;   // already enabled
    int s1 = cfg_.stage1_dim;
    // Construct a minimal SpeedEngine GPU to initialize the process-global
    // OpenCL context/kernels (init_gpu_impl). Its "corpus" is irrelevant (1x1);
    // it only keeps the context alive (gpu_keepalive_).
    try {
        std::vector<float> dummy(1, 0.0f);
        SpeedEngine* se = new SpeedEngine(dummy.data(), 1, 1,
                                          Metric::Cosine, 0, 4, true, opencl_lib);
        if (!se->has_gpu()) { delete se; return false; }
        gpu_keepalive_ = se;
    } catch (...) {
        return false;
    }
    // Upload the projection buffers once.
    bool ok = madhava_gpu_stage1_init(pr1_f_, e1_, vn_eff_, n_, s1,
                                      opencl_lib.c_str());
    if (!ok) {
        delete static_cast<SpeedEngine*>(gpu_keepalive_);
        gpu_keepalive_ = nullptr;
        return false;
    }
    gpu_stage1_ready_ = true;
    return true;
}

void MadhavaL2::build(const uint8_t* raw_base, int n) {
    base_ = raw_base;
    n_ = n;
    built_ = true;
    int D = cfg_.dim, s1 = cfg_.stage1_dim, s2 = cfg_.stage2_dim;
    bool normalize = (cfg_.metric == Metric::Cosine) && cfg_.normalize_input;
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Build the projection(s). Default: QR-orthogonalized random Gaussian
    //    via MGS (the historical behavior). With BasisMode::PCACorpus ("UB
    //    Width"): align the projection to the corpus' principal directions, so
    //    the residual e(v) shrinks to the manifold residual and the bound stays
    //    tight at high dimension (the validated fix for the √d bottleneck).
    //
    //    The PCA basis MUST be computed on the float32 embeddings, not the uint8
    //    bytes: a uint8 quantized corpus collapses the eigenvalues to ~1e-4 and
    //    the principal basis stops reflecting the manifold (measured at d=1536).
    //    So when PCACorpus is requested over a uint8 corpus, we materialize the
    //    float32 copy once and compute the basis from it.
    if (cfg_.basis == BasisMode::PCACorpus && corpus_f32_ == nullptr) {
        corpus_f32_ = new float[(size_t)n * D];
        for (size_t i = 0; i < (size_t)n * D; i++) corpus_f32_[i] = (float)base_[i];
    }
    P1_ = new float[(size_t)s1 * D];
    bool pca_ok1 = false;
    if (cfg_.basis == BasisMode::PCACorpus) {
        pca_ok1 = build_pca_basis(corpus_f32_, n, D, s1, cfg_.seed + s1,
                                  cfg_.pca_sample, cfg_.normalize_input,
                                  cfg_.pca_iterations, P1_);
    }
    if (!pca_ok1) {
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
    if (s2 > 0) {
        P2_ = new float[(size_t)s2 * D];
        bool pca_ok2 = false;
        if (cfg_.basis == BasisMode::PCACorpus) {
            pca_ok2 = build_pca_basis(corpus_f32_, n, D, s2, cfg_.seed + s2,
                                      cfg_.pca_sample, cfg_.normalize_input,
                                      cfg_.pca_iterations, P2_);
        }
        if (!pca_ok2) {
            std::mt19937 rng2(cfg_.seed + s2);
            std::normal_distribution<float> nd(0, 1);
            for (int i = 0; i < s2; i++) {
                for (int j = 0; j < D; j++) P2_[i * D + j] = nd(rng2);
                for (int k = 0; k < i; k++) {
                    float dp = 0;
                    for (int j = 0; j < D; j++) dp += P2_[i * D + j] * P2_[k * D + j];
                    for (int j = 0; j < D; j++) P2_[i * D + j] -= dp * P2_[k * D + j];
                }
                float nr = 0;
                for (int j = 0; j < D; j++) nr += P2_[i * D + j] * P2_[i * D + j];
                nr = std::sqrt(nr);
                if (nr > 1e-10f)
                    for (int j = 0; j < D; j++) P2_[i * D + j] /= nr;
            }
        }
    }

    // 2. Allocate buffers.
    // Memory-aware: with int8 quantization we store ONLY the int8 projections
    // (no float32 copy) — this is the memory-light path used at 100M scale.
    vn_ = new float[n];
    vn_eff_ = new float[n];
    e1_ = new float[n];
    pr1_scale_ = new float[s1];
    if (cfg_.quant == QuantMode::Int8) {
        pr1_ = new int8_t[(size_t)n * s1];
    } else {
        pr1_f_ = new float[(size_t)n * s1];
        // SCAN INT8 (2026-09-03): opt-in fast scan for float32 corpora. Keep
        // pr1_f_ (needed for e(v)/exact) AND allocate the int8 pr1_ so ub_raw
        // can scan with dot_int8_scaled (4× less DRAM traffic, ~2× faster at
        // large N). pr1_scale_ was already allocated above.
        if (cfg_.scan_int8) {
            pr1_ = new int8_t[(size_t)n * s1];
        }
    }
    if (s2 > 0) {
        e2_ = new float[n];
        pr2_scale_ = new float[s2];
        if (cfg_.quant == QuantMode::Int8) {
            pr2_ = new int8_t[(size_t)n * s2];
        } else {
            pr2_f_ = new float[(size_t)n * s2];
            if (cfg_.scan_int8) {
                pr2_ = new int8_t[(size_t)n * s2];
            }
        }
    }

    // 3. Calibrate the int8 per-axis scales on a sample.
    int sn = std::min(n, 100000);
    std::vector<float> ck((size_t)sn * D);
    std::vector<float> ma1(s1, 0), ma2(s2 > 0 ? s2 : 0);
    for (int i = 0; i < sn; i++) {
        float raw_norm;
        load_raw(base_ + (size_t)i * D, ck.data() + (size_t)i * D, D, normalize);
        vn_[i] = raw_norm;
        vn_eff_[i] = normalize ? 1.0f : raw_norm;
    }
    for (int i = 0; i < sn; i++) {
        float* v = ck.data() + (size_t)i * D;
        for (int j = 0; j < s1; j++) {
            float s = dot_f32(v, P1_ + (size_t)j * D, D);
            if (std::fabs(s) > ma1[j]) ma1[j] = std::fabs(s);
        }
        if (s2 > 0) for (int j = 0; j < s2; j++) {
            float s = dot_f32(v, P2_ + (size_t)j * D, D);
            if (std::fabs(s) > ma2[j]) ma2[j] = std::fabs(s);
        }
    }
    for (int j = 0; j < s1; j++)
        pr1_scale_[j] = std::max(ma1[j] / 127.0f * 1.05f, 1e-10f);
    if (s2 > 0) for (int j = 0; j < s2; j++)
        pr2_scale_[j] = std::max(ma2[j] / 127.0f * 1.05f, 1e-10f);

    // 4. Streaming pass: compute projections + residuals.
    //    When a float32 corpus is available, the residual must use it (the same
    //    space as the basis); the uint8 re-normalized base_ is a different space
    //    and would collapse the projection energy to ~0 (e(v)→1.0).
    const int CHUNK = 500000;
    const bool use_f32 = (corpus_f32_ != nullptr) && (cfg_.quant == QuantMode::None);
    int p = 0;
    while (p < n) {
        int nt = std::min(CHUNK, n - p);
        std::vector<float> ch((size_t)nt * D);
        if (use_f32) {
            // copy the real float32 rows (same space as the PCA basis)
            std::memcpy(ch.data(), corpus_f32_ + (size_t)p * D,
                        (size_t)nt * D * sizeof(float));
            for (int i = 0; i < nt; i++) {
                float nn = 0.0f;
                for (int j = 0; j < D; j++) nn += ch[(size_t)i * D + j] * ch[(size_t)i * D + j];
                vn_[p + i] = std::sqrt(nn);
                vn_eff_[p + i] = normalize ? 1.0f : vn_[p + i];
                // SPACE MISMATCH FIX (2026-08-15): the residual e(v) =
                // sqrt(||v||^2 - ||P v||^2) assumes v is unit-norm when
                // normalize=true (vn_eff_=1.0). But corpus_f32_ may hold RAW
                // values (e.g. uint8 0-255 cast to float) that are NOT
                // unit-norm. Without normalizing v here, pn1 = ||P v||^2 >> 1
                // and e(v) collapses to 0 — the BIGANN space mismatch. So when
                // normalize=true, divide v by its norm so vn_eff_=1.0 is exact.
                if (normalize && nn > 1e-10f) {
                    float inv = 1.0f / std::sqrt(nn);
                    for (int j = 0; j < D; j++) ch[(size_t)i * D + j] *= inv;
                }
            }
        } else {
            for (int i = 0; i < nt; i++) {
                float raw_norm;
                load_raw(base_ + (size_t)(p + i) * D, ch.data() + (size_t)i * D, D, normalize);
                vn_[p + i] = raw_norm;
                vn_eff_[p + i] = normalize ? 1.0f : raw_norm;
            }
        }
#pragma omp parallel for
        for (int i = 0; i < nt; i++) {
            int id = p + i;
            float* v = ch.data() + (size_t)i * D;
            // Stage-1
            float pj1[256];
            float pn1 = 0;
            for (int j = 0; j < s1; j++) { pj1[j] = dot_f32(v, P1_ + (size_t)j * D, D); pn1 += pj1[j] * pj1[j]; }
            if (cfg_.quant == QuantMode::None) {
                for (int j = 0; j < s1; j++) pr1_f_[(size_t)id * s1 + j] = pj1[j];
                // SCAN INT8: also quantize the projection so ub_raw can use the
                // fast int8 scan. The scale was calibrated on a sample above.
                if (cfg_.scan_int8 && pr1_ != nullptr) {
                    quantize(pj1, pr1_ + (size_t)id * s1, pr1_scale_, s1);
                }
            } else {
                quantize(pj1, pr1_ + (size_t)id * s1, pr1_scale_, s1);
            }
            // Cauchy-Schwarz residual over the REAL float32 projection:
            // e(v) = sqrt(||v||² − ||Pv||²). This is what the inequality requires.
            float vn2 = vn_eff_[id] * vn_eff_[id];
            e1_[id] = std::sqrt(std::max(0.0f, vn2 - pn1));
            // Stage-2
            if (s2 > 0) {
                float pj2[256];
                float pn2 = 0;
                for (int j = 0; j < s2; j++) { pj2[j] = dot_f32(v, P2_ + (size_t)j * D, D); pn2 += pj2[j] * pj2[j]; }
                if (cfg_.quant == QuantMode::None) {
                    for (int j = 0; j < s2; j++) pr2_f_[(size_t)id * s2 + j] = pj2[j];
                    if (cfg_.scan_int8 && pr2_ != nullptr) {
                        quantize(pj2, pr2_ + (size_t)id * s2, pr2_scale_, s2);
                    }
                } else {
                    quantize(pj2, pr2_ + (size_t)id * s2, pr2_scale_, s2);
                }
                e2_[id] = std::sqrt(std::max(0.0f, vn2 - pn2));
            }
        }
        p += nt;
    }

    build_s_ = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

// Cauchy-Schwarz upper bound of ⟨v,q⟩ for a vector at a given layer.
// Uses either the int8-quantized projection (with quant margin) or the
// float32 projection exactly.
float MadhavaL2::ub_raw(int idx, int layer, const float* pq, float qr, float qm) const {
    int s = (layer == 1) ? cfg_.stage1_dim : cfg_.stage2_dim;
    if (cfg_.quant == QuantMode::Int8) {
        const int8_t* pr = (layer == 1) ? pr1_ + (size_t)idx * s : pr2_ + (size_t)idx * s;
        const float* scale = (layer == 1) ? pr1_scale_ : pr2_scale_;
        // M4: AVX2 dot int8×scale×pq (16 int8/iter) — o hot loop do Stage-1.
        float inner = dot_int8_scaled(pr, scale, pq, s);
        float e = (layer == 1) ? e1_[idx] : e2_[idx];
        return inner + e * qr + qm + 1e-5f;
    } else {
        // SCAN INT8 (2026-09-03): when scan_int8 is on, the float32 path ALSO
        // has the int8 projections (pr1_/pr2_ + pr1_scale_). Use them for the
        // scan (dot_int8_scaled: 1B/doc/dim vs 4B) — ~2× faster at large N,
        // validated safe (the int8 bound with the qm quant margin never excludes
        // a doc the float32 bound keeps). pr1_f_ is retained for e(v)/exact.
        if (pr1_ != nullptr && pr1_scale_ != nullptr) {
            const int8_t* pri = (layer == 1) ? pr1_ + (size_t)idx * s : pr2_ + (size_t)idx * s;
            const float* scale = (layer == 1) ? pr1_scale_ : pr2_scale_;
            float inner = dot_int8_scaled(pri, scale, pq, s);
            float e = (layer == 1) ? e1_[idx] : e2_[idx];
            // The caller (search) passes qm (the int8 quant margin for the
            // query) when quant==Int8; for the scan_int8 path on a float32
            // corpus, search must also pass qm (it does — see the pq1/qm1
            // computation which now includes the scan_int8 margin). Here we add
            // the per-doc margin via the same qm the caller supplied + the
            // float32 orthonormality margin. dot_int8_scaled already includes
            // scale; qm (query side) + 1e-4 (numeric) cover the rest.
            return inner + e * qr + qm + 1e-4f;
        }
        const float* prf = (layer == 1) ? pr1_f_ + (size_t)idx * s : pr2_f_ + (size_t)idx * s;
        float inner = dot_f32(prf, pq, s);
        float e = (layer == 1) ? e1_[idx] : e2_[idx];
        // FIX(1.8.0): margem de segurança para o caminho float32.
        // O bound Cauchy-Schwarz exige que P tenha linhas ortonormais; o MGS
        // acumula erro numérico (~1e-7..1e-14 conforme a dimensão) que pode
        // subestimar o residual real ‖v−PᵀPv‖. Sem margem, o bound pode ser
        // MENOR que ⟨v,q⟩ real → lb de L2² maior que o L2² real → poda um
        // vizinho verdadeiro (violação da garantia). A margem cobre o erro
        // de ortonormalidade + o arredondamento do produto interno.
        return inner + e * qr + 1e-4f;
    }
}

// Exact metric score between a raw vector and a float query.
float MadhavaL2::exact_score(int idx, const float* q) const {
    if (cfg_.metric == Metric::L2) {
        // M4: AVX2 L2² (16 uint8/iter) — o hot loop do post-filter / scan exato.
        return l2_sq_avx(base_ + (size_t)idx * cfg_.dim, q, cfg_.dim);
    } else {
        // Cosine. BUG B FIX (2026-08-15): when a float32 corpus is available
        // (corpus_f32_), the exact score MUST use it — the same space as the
        // PCA basis and the query. Using the uint8 re-normalized base_ puts v
        // in a different space (e(v)→1.0, wrong worst/top-K). For a uint8
        // corpus (BIGANN), corpus_f32_ is null and we fall back to base_.
        float dot = 0, vn2 = 0, qn2 = 0;
        if (corpus_f32_ != nullptr && cfg_.quant == QuantMode::None) {
            const float* vf = corpus_f32_ + (size_t)idx * cfg_.dim;
            for (int j = 0; j < cfg_.dim; j++) { dot += vf[j] * q[j]; vn2 += vf[j] * vf[j]; }
        } else {
#if defined(__AVX2__) && defined(__FMA__)
            __m256 sd = _mm256_setzero_ps();
            __m256 sv = _mm256_setzero_ps();
            int j = 0;
            for (; j + 16 <= cfg_.dim; j += 16) {
                const uint8_t* v8 = base_ + (size_t)idx * cfg_.dim + j;
                __m128i u8 = _mm_loadu_si128((const __m128i*)v8);
                __m256i lo32 = _mm256_cvtepu8_epi32(u8);
                __m256i hi32 = _mm256_cvtepu8_epi32(_mm_srli_si128(u8, 8));
                __m256 flo = _mm256_cvtepi32_ps(lo32);
                __m256 fhi = _mm256_cvtepi32_ps(hi32);
                __m256 qlo = _mm256_loadu_ps(q + j);
                __m256 qhi = _mm256_loadu_ps(q + j + 8);
                sd = _mm256_fmadd_ps(flo, qlo, sd);
                sd = _mm256_fmadd_ps(fhi, qhi, sd);
                sv = _mm256_fmadd_ps(flo, flo, sv);
                sv = _mm256_fmadd_ps(fhi, fhi, sv);
            }
            float od[8], ov[8];
            _mm256_storeu_ps(od, sd);
            _mm256_storeu_ps(ov, sv);
            for (int k = 0; k < 8; k++) { dot += od[k]; vn2 += ov[k]; }
            for (; j < cfg_.dim; j++) {
                float v = (float)base_[(size_t)idx * cfg_.dim + j];
                dot += v * q[j];
                vn2 += v * v;
            }
#else
            for (int j = 0; j < cfg_.dim; j++) {
                float v = (float)base_[(size_t)idx * cfg_.dim + j];
                dot += v * q[j];
                vn2 += v * v;
            }
#endif
        }
        for (int j = 0; j < cfg_.dim; j++) qn2 += q[j] * q[j];
        float denom = std::sqrt(vn2 * qn2);
        return denom > 1e-10f ? dot / denom : 0.0f;
    }
}

SearchResult MadhavaL2::search(const float* query, const std::vector<float>& query_norm) const {
    SearchResult out;
    if (!built_) return out;
    int N = n_, D = cfg_.dim, s1 = cfg_.stage1_dim, s2 = cfg_.stage2_dim, K = cfg_.k;
    bool is_l2 = (cfg_.metric == Metric::L2);
    bool normalize = (cfg_.metric == Metric::Cosine) && cfg_.normalize_input;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Query norm (effective for the chosen metric).
    float qn = query_norm.empty() ? std::sqrt(dot_f32(query, query, D)) : query_norm[0];
    float qn_eff = normalize ? 1.0f : qn;

    // For cosine + normalize_input, the vectors were unit-normalized at build;
    // the query must be normalized the same way so projections match.
    std::vector<float> qbuf;
    const float* qproj = query;
    if (normalize && qn > 1e-10f) {
        qbuf.resize((size_t)D);
        float inv = 1.0f / qn;
        for (int j = 0; j < D; j++) qbuf[j] = query[j] * inv;
        qproj = qbuf.data();
    }

    // Query projections + residuals.
    float pq1[256];
    float q1s = 0;
    for (int j = 0; j < s1; j++) { pq1[j] = dot_f32(qproj, P1_ + (size_t)j * D, D); q1s += pq1[j] * pq1[j]; }
    float qr1 = std::sqrt(std::max(0.0f, qn_eff * qn_eff - q1s));
    // SCAN INT8: the quant margin qm is needed when ub_raw uses the int8
    // projections — native Int8 quant OR the opt-in scan_int8 on a float32
    // corpus (which keeps pr1_ int8 alongside pr1_f_).
    const bool scan_uses_int8 = (cfg_.quant == QuantMode::Int8)
        || (cfg_.scan_int8 && pr1_ != nullptr && pr1_scale_ != nullptr);
    float qm1 = 0;
    if (scan_uses_int8)
        for (int j = 0; j < s1; j++) qm1 += 0.5f * pr1_scale_[j] * std::fabs(pq1[j]);
    float qn2 = qn_eff * qn_eff;

    // Stage-1: bound pruning over all N. This pass computes UB(v,q) for all N
    // and stores each value in ub_local (the per-query buffer below) so the
    // audit hook can reuse it — the Fusion-B win (drop the second O(N·s1) pass
    // that recomputed ub_raw after the top-K) WITHOUT any shared engine state.
    out.bound_pairs = N;
    std::vector<std::pair<float, int>> b1((size_t)N);
    // Per-query UB buffer (2026-09-04). The audit hook below needs the Stage-1
    // UB of all N docs to certify exclusion against the global K-th threshold.
    // This buffer is LOCAL to this call — not a shared mutable member — so
    // concurrent search() calls on the same engine (search_batch / OpenMP /
    // GPU, Phase 2) never race on it. Pre-sized + disjoint index writes are
    // thread-safe under the parallel-for below. This keeps the Fusion-B
    // compute win without the shared-state data race of the 1.9.12 mutable
    // buffer.
    std::vector<float> ub_local((size_t)N);
    // Phase-3 GPU Stage-1 (2026-09-04): when the upload-once GPU buffers are
    // ready, run the O(N) bound scan on the GPU (madhava_gpu_stage1_scan) and
    // materialize b1/ub_local from the returned sortable scores. The GPU
    // computes the SAME ub_raw per doc (parity-validated), so the k1 survivor
    // set and the audit are bit-identical to the CPU path. Serialized by the
    // mutex (OpenCL queue is not thread-safe for concurrent enqueue); each call
    // has its own scores buffer, so search() stays thread-safe.
    bool gpu_scan = false;
    if (gpu_stage1_ready_) {
        std::lock_guard<std::mutex> lock(gpu_mu_);
        std::vector<float> gpu_scores((size_t)N);
        if (madhava_gpu_stage1_scan(pq1, qr1, qm1, qn2, N, s1, is_l2 ? 1 : 0,
                                    gpu_scores.data())) {
            // The GPU returned the SORTABLE scores (cosine: -ub; L2:
            // vn²+qn2-2·ub). Recover ub for the audit and fill b1/ub_local.
            // This fill must be PARALLEL (disjoint writes) — a serial loop over
            // N here cost ~77ms at N=500k and wiped out the GPU scan win.
#pragma omp parallel for
            for (int i = 0; i < N; i++) {
                float score = gpu_scores[(size_t)i];
                //   cosine: score = -ub            → ub = -score
                //   L2:     score = vn² + qn2 - 2·ub → ub = (vn² + qn2 - score)/2
                float ub = is_l2
                    ? (vn_eff_[i] * vn_eff_[i] + qn2 - score) * 0.5f
                    : -score;
                ub_local[(size_t)i] = ub;
                b1[i] = {score, i};
            }
            gpu_scan = true;
        }
    }
    if (!gpu_scan) {
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            float ub = ub_raw(i, 1, pq1, qr1, qm1);
            ub_local[(size_t)i] = ub;   // disjoint index writes: thread-safe
            // Score for sorting:
            //   L2:    lb = ‖v‖² + ‖q‖² − 2·ub   (ascending = best)
            //   Cosine: ub (descending = best) — sort ascending on −ub
            float score = is_l2 ? (vn_eff_[i] * vn_eff_[i] + qn2 - 2.0f * ub) : -ub;
            b1[i] = {score, i};
        }
    }
    // EXHAUSTIVE AUDIT (2026-09-03): audit_exhaustive forces the pool to cover
    // the ENTIRE corpus so the exact post-filter is global. We keep the b1
    // bound-sort (harmless) but skip the nth_element cut so every doc reaches
    // the exact re-score below. With k1 = N the pool is the whole corpus and
    // recall_guarantee below is "exact_global".
    int k1;
    if (cfg_.audit_exhaustive) {
        k1 = N;
    } else {
        k1 = std::max(cfg_.k1_min, (int)(N * cfg_.k1_fraction));
        if (k1 > N) k1 = N;
        if (k1 < N) std::nth_element(b1.begin(), b1.begin() + k1, b1.end());
    }
    out.k1 = k1;

    // Stage-2 (optional): tighter bound on the k1 survivors.
    int k2 = k1;
    const std::vector<std::pair<float, int>>* survivors = &b1;
    std::vector<std::pair<float, int>> b2;
    if (s2 > 0) {
        float pq2[256];
        float q2s = 0;
        for (int j = 0; j < s2; j++) { pq2[j] = dot_f32(qproj, P2_ + (size_t)j * D, D); q2s += pq2[j] * pq2[j]; }
        float qr2 = std::sqrt(std::max(0.0f, qn_eff * qn_eff - q2s));
        float qm2 = 0;
        if (scan_uses_int8)
            for (int j = 0; j < s2; j++) qm2 += 0.5f * pr2_scale_[j] * std::fabs(pq2[j]);

        b2.resize((size_t)k1);
#pragma omp parallel for
        for (int i = 0; i < k1; i++) {
            int vi = b1[i].second;
            float ub2 = ub_raw(vi, 2, pq2, qr2, qm2);
            float score2 = is_l2 ? (vn_eff_[vi] * vn_eff_[vi] + qn2 - 2.0f * ub2) : -ub2;
            b2[i] = {score2, vi};
        }
        // EXHAUSTIVE AUDIT: skip the k2 cut so every doc reaches the exact
        // post-filter (the global guarantee needs k2 = N, not the k2_max cap).
        int k2t;
        if (cfg_.audit_exhaustive) {
            k2t = N;
        } else {
            k2t = std::max(cfg_.k2_min, (int)(N * cfg_.k2_fraction));
            if (k2t > k1) k2t = k1;
            // Cap Stage-2 survivors (streaming/100M: limit post-filter cost).
            if (k2t > cfg_.k2_max) k2t = cfg_.k2_max;
        }
        if (k2t < k2) std::nth_element(b2.begin(), b2.begin() + k2t, b2.end());
        k2 = k2t;
        survivors = &b2;
    }
    out.k2 = k2;

    // Rank the survivors:
    //   - If modulation is on, rank by modulated = B1 + α(B2−B1)
    //   - Else rank by the (already computed) bound score.
    //
    // FUSION A (2026-09-04): when the postfilter runs WITHOUT early_exit (the
    // default), every k2 survivor is re-scored by exact_score and the final
    // ranking is by partial_sort of the exact score — the `ranked` order from
    // the modulation does NOT affect the result. So the modulation (which
    // recomputes B1/B2 and sweeps e1_sum over N) is unnecessary: we skip the
    // block and feed the survivors (bound order) straight through. The
    // modulation_gain becomes 0 (a metric — the result is identical). The
    // modulation only runs when it matters: postfilter=false (ranking by bound)
    // OR early_exit=true (evaluation order affects where early_exit stops).
    const bool pf_eff_A = cfg_.postfilter || cfg_.audit_exhaustive;
    const bool ee_eff_A = cfg_.early_exit && !cfg_.audit_exhaustive;
    const bool modulation_needed = cfg_.modulation && s2 > 0
        && (!pf_eff_A || ee_eff_A);
    std::vector<std::pair<float, int>> ranked(k2);
    double mod_gain = 0.0;
    if (modulation_needed) {
        // Recompute B1/B2 for the k2 survivors to build the modulated score.
        float pq1m[256], pq2m[256];
        for (int j = 0; j < s1; j++) pq1m[j] = pq1[j];
        if (s2 > 0) for (int j = 0; j < s2; j++) pq2m[j] = 0.0f; // recompute below
        // (pq2 already available in this scope? recompute to be safe)
        float q1sm = 0, q2sm = 0;
        for (int j = 0; j < s1; j++) q1sm += pq1m[j] * pq1m[j];
        for (int j = 0; j < s2; j++) { float v = dot_f32(qproj, P2_ + (size_t)j * D, D); pq2m[j] = v; q2sm += v * v; }
        float qr1m = std::sqrt(std::max(0.0f, qn_eff * qn_eff - q1sm));
        float qr2m = std::sqrt(std::max(0.0f, qn_eff * qn_eff - q2sm));
        float qm1m = 0, qm2m = 0;
        if (scan_uses_int8) {
            for (int j = 0; j < s1; j++) qm1m += 0.5f * pr1_scale_[j] * std::fabs(pq1m[j]);
            for (int j = 0; j < s2; j++) qm2m += 0.5f * pr2_scale_[j] * std::fabs(pq2m[j]);
        }
        // mean(e1) over all docs for the sigmoid (stack uses mean over corpus)
        double e1_sum = 0;
        for (int i = 0; i < N; i++) e1_sum += e1_[i];
        float mu = (float)(e1_sum / std::max(N, 1));

#pragma omp parallel for reduction(+:mod_gain)
        for (int i = 0; i < k2; i++) {
            int vi = survivors->at(i).second;
            float B1 = ub_raw(vi, 1, pq1m, qr1m, qm1m);
            float B2 = ub_raw(vi, 2, pq2m, qr2m, qm2m);
            float delta_e = (e1_[vi] - e2_[vi]) / std::max(mu, 1e-9f);
            float alpha = 1.0f / (1.0f + std::exp(-delta_e * 0.5f));
            float modulated = B1 + alpha * (B2 - B1);
            // Convert to a sortable score.
            float score = is_l2 ? (vn_eff_[vi] * vn_eff_[vi] + qn2 - 2.0f * modulated) : -modulated;
            ranked[i] = {score, vi};
            mod_gain += std::fabs(score - survivors->at(i).first);
        }
        mod_gain /= std::max(k2, 1);
    } else {
        for (int i = 0; i < k2; i++) ranked[i] = survivors->at(i);
    }
    out.modulation_gain = mod_gain;

    // Final: post-filter with exact metric on the survivors, else top-K by score.
    // Early-exit (bigann_stream V17 optimization): score survivors in bound order
    // and stop once the *bound* of the next candidate cannot beat the current
    // K-th best exact score. This drives k3 well below k2_max.
    //
    // ranked[i].first holds the bound score:
    //   L2:    lower bound of L2²  (ascending = better; exact L2² >= lb)
    //   cosine: -UB(similarity)     (ascending = better; exact cos <= UB)
    // So:
    //   L2:    stop when lb_next >= worst_exact_L2 (exact L2² >= lb >= worst → can't be top-K)
    //   cosine: stop when -UB_next <= worst_exact (ub_next >= -worst? no: -UB_next <= worst
    //           means UB_next >= -worst, and exact cos <= UB_next — not a valid stop!)
    // CORRECT cosine stop: UB_next <= worst_exact, i.e. -ranked[i+1].first <= worst_exact.
    // EXHAUSTIVE AUDIT: the exact post-filter must cover the WHOLE corpus and
    // run to completion. We force the effective flags locally (cfg_ is const):
    //   - postfilter must be ON (the global top-K is decided by exact re-score),
    //   - early_exit must be OFF (it could stop before scoring all N, leaving
    //     k3 < N and thus recall_guarantee != "exact_global").
    const bool pf_eff = cfg_.postfilter || cfg_.audit_exhaustive;
    const bool ee_eff = cfg_.early_exit && !cfg_.audit_exhaustive;
    if (pf_eff) {
        // Sort ALL survivors by bound score ascending (best-first). The early-exit
        // walks the list in bound order, so the whole list must be sorted — a
        // partial_sort would leave the tail unordered and break the pruning bound.
        std::sort(ranked.begin(), ranked.end());
        std::vector<std::pair<float, int>> exact(k2);
        int n_exact = 0;
        std::vector<std::pair<float, int>> heap; // holds exact scores; "worst" is heap[0]
        auto worse_than = [&](float a, float b) { return is_l2 ? a > b : a < b; };
        auto push_candidate = [&](int vi, float score) {
            if ((int)heap.size() < K) {
                heap.push_back({score, vi});
                if ((int)heap.size() == K)
                    std::make_heap(heap.begin(), heap.end(),
                                   [&](auto& a, auto& b) { return worse_than(a.first, b.first); });
            } else if (worse_than(heap[0].first, score)) {
                std::pop_heap(heap.begin(), heap.end(),
                              [&](auto& a, auto& b) { return worse_than(a.first, b.first); });
                heap.back() = {score, vi};
                std::push_heap(heap.begin(), heap.end(),
                               [&](auto& a, auto& b) { return worse_than(a.first, b.first); });
            }
        };
        if (ee_eff) {
            int i = 0;
            for (; i < k2; i++) {
                int vi = ranked[i].second;
                float score = exact_score(vi, query);
                exact[i] = {score, vi};
                n_exact++;
                push_candidate(vi, score);
                if ((int)heap.size() == K && i + 1 < k2) {
                    float worst = heap[0].first;              // current K-th best exact score
                    float next_bound = ranked[i + 1].first;   // bound score of next survivor
                    bool cannot_beat = is_l2 ? next_bound >= worst
                                             : -next_bound <= worst;
                    if (cannot_beat)
                        break;
                }
            }
        } else {
#pragma omp parallel for
            for (int i = 0; i < k2; i++) {
                int vi = ranked[i].second;
                float score = exact_score(vi, query);
                exact[i] = {score, vi};
            }
            n_exact = k2;
            // Build a SEPARATE heap for the honest pruned_by_bound count — the
            // K best exact scores among the evaluated survivors. This does NOT
            // affect the returned indices (which come from the exact sort
            // below); it only provides the K-th best exact score (worst) so the
            // proof-based pruning can be measured.
            for (int i = 0; i < n_exact; i++) push_candidate(exact[i].second, exact[i].first);
        }
        out.k3 = n_exact;

        // HONEST PRUNING BREAKDOWN (2026-08-15).
        //   exact_evals     = how many were scored exactly (n_exact).
        //   pruned_by_bound = the number of the N vectors whose Cauchy-Schwarz
        //                     bound is PROVEN below the K-th best exact score:
        //                     UB(v,q) < worst. This is the proof-based pruning
        //                     (0 violations by construction). With a wide bound
        //                     (e(v)≈1) it is ~0 — the truth, not the "95%" of
        //                     the fixed k1_fraction cutoff.
        //   pruned_by_prefilter = N − exact_evals − pruned_by_bound (the fixed
        //                     k1_fraction/k2_max cutoffs that are NOT proven by
        //                     a per-vector certificate).
        //
        // We recompute UB(v,q) for ALL N in Stage-1 order (the query
        // projections pq1/qr1 are already available) and count how many fall
        // below the worst (K-th best exact) score. This measures the REAL
        // bound-driven pruning against the whole corpus, independent of the
        // fixed cutoff.
        out.exact_evals = n_exact;
        // (the honest pruned_by_bound + audit hook are computed AFTER the
        //  final top-K is known, using the GLOBAL K-th threshold — see below.)

        if (ee_eff && (int)heap.size() == K) {
            // early_exit: the heap holds the top-K by exact score in the
            // walk order — this IS the correct result.
            std::sort(heap.begin(), heap.end(),
                      [&](auto& a, auto& b) { return !worse_than(a.first, b.first); });
            for (int i = 0; i < K; i++) out.indices.push_back(heap[i].second);
        } else {
            // Fallback (early_exit off or partial): sort the exact scores.
            if (is_l2)
                std::partial_sort(exact.begin(), exact.begin() + std::min(K, n_exact), exact.end());
            else
                std::partial_sort(exact.begin(), exact.begin() + std::min(K, n_exact), exact.end(),
                                  [](auto& a, auto& b) { return a.first > b.first; });
            for (int i = 0; i < std::min(K, n_exact); i++) out.indices.push_back(exact[i].second);
        }
    } else {
        std::partial_sort(ranked.begin(), ranked.begin() + std::min(K, k2), ranked.end());
        for (int i = 0; i < std::min(K, k2); i++) out.indices.push_back(ranked[i].second);
    }

    // RECALL GUARANTEE (2026-09-03): derived from OBSERVED state, not a promise.
    //   - With the exact post-filter ON and all N scored (k3 == N), the returned
    //     top-K IS the exact global top-K → "exact_global".
    //   - Otherwise the result is the best WITHIN the post-filter pool (docs cut
    //     by k1_fraction/k2_max were discarded without a proof) → "pool_only".
    //   - Without a post-filter (cfg_.postfilter=false) the ranking is by the
    //     bound/modulated score, never the exact top-K → "pool_only".
    // This makes the honest statement the README already makes ("bound_violations
    // == 0 does not mean the returned top-K is the true top-K") machine-readable.
    out.recall_guarantee = (pf_eff && out.k3 == N) ? "exact_global" : "pool_only";

    // HONEST PRUNING BREAKDOWN + AUDIT HOOK (computed against the GLOBAL
    // K-th threshold, not the post-filter pool threshold).
    //
    // 1.9.0 BUG FIX: the old code counted pruned_by_bound against the heap[0]
    // of the post-filter pool (the K-th best exact score AMONG THE k1_fraction
    // CANDIDATES). When that pool threshold exceeds the true global K-th
    // score, docs that ARE in the real top-K got marked "provably excluded" —
    // a contradiction (measured: 9 of the top-10 were in audit_ids). The
    // certificate must be a WITNESS of the motor's decision, and the decision
    // is the GLOBAL top-K. So the threshold is exact_score of the K-th result.
    out.pruned_by_bound = 0;
    out.audit_ids.clear();
    out.audit_ubs.clear();
    out.audit_residuals.clear();
    out.audit_l2_lbs.clear();
    if ((int)out.indices.size() >= K && K > 0) {
        float worst = exact_score(out.indices[K - 1], query);  // GLOBAL K-th
        out.audit_threshold = worst;
        out.audit_ids.reserve(std::min(N, 4096));
        out.audit_ubs.reserve(std::min(N, 4096));
        out.audit_residuals.reserve(std::min(N, 4096));
        if (is_l2) out.audit_l2_lbs.reserve(std::min(N, 4096));
        // Parallel O(N) bound scan (Gargalo #2): the audit hook runs on EVERY
        // search, so this loop must not serialize the query path. Each thread
        // keeps its own buffers (ub_raw/query projections are read-only and
        // thread-safe after build); the per-thread lists are concatenated in
        // thread order, preserving the deterministic ascending doc_id order.
        long long pruned = 0;
        {
            const int nthreads = std::max(1, wm_omp_max_threads());
            std::vector<std::vector<int>> t_ids(nthreads);
            std::vector<std::vector<float>> t_ubs(nthreads);
            std::vector<std::vector<float>> t_res(nthreads);
            std::vector<std::vector<float>> t_lbs(nthreads);
            std::vector<long long> t_count(nthreads, 0);
#pragma omp parallel for
            for (int i = 0; i < N; i++) {
                const int t = wm_omp_thread_num();
                // (2026-09-04) Reuses the UB already computed in pass 1
                // (ub_local, the per-query buffer) instead of recomputing
                // ub_raw — the Fusion-B win, now without shared query state.
                float ub = ub_local[(size_t)i];
                float resid = e1_[i] * qr1;
                if (is_l2) {
                    float lb = vn_eff_[i] * vn_eff_[i] + qn2 - 2.0f * ub;
                    if (lb > worst) {
                        t_ids[t].push_back(i);
                        t_ubs[t].push_back(ub);
                        t_res[t].push_back(resid);
                        t_lbs[t].push_back(lb);
                        t_count[t]++;
                    }
                } else {
                    if (ub < worst) {
                        t_ids[t].push_back(i);
                        t_ubs[t].push_back(ub);
                        t_res[t].push_back(resid);
                        t_count[t]++;
                    }
                }
            }
            for (int t = 0; t < nthreads; t++) {
                pruned += t_count[t];
                out.audit_ids.insert(out.audit_ids.end(),
                                     t_ids[t].begin(), t_ids[t].end());
                out.audit_ubs.insert(out.audit_ubs.end(),
                                     t_ubs[t].begin(), t_ubs[t].end());
                out.audit_residuals.insert(out.audit_residuals.end(),
                                           t_res[t].begin(), t_res[t].end());
                if (is_l2)
                    out.audit_l2_lbs.insert(out.audit_l2_lbs.end(),
                                            t_lbs[t].begin(), t_lbs[t].end());
            }
        }
        out.pruned_by_bound = pruned;
    }
    // prefilter = what the fixed cutoff discarded WITHOUT a certificate.
    // Clamp: n_exact and pruned_by_bound are not disjoint.
    out.pruned_by_prefilter = std::max<long long>(0,
        (long long)N - (long long)out.k3 - out.pruned_by_bound);

    out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

SearchResult MadhavaL2::search(const float* query) const {
    std::vector<float> qn;
    return search(query, qn);
}

// M1 (v1.8.0): batch search — nq queries, k resultados cada.
// Paralelizado cross-query com OpenMP: cada thread processa um sub-range de
// queries (o engine é imutável após build → thread-safe). O resultado é
// concatenado na ordem das queries (determinístico).
std::vector<int> MadhavaL2::search_batch(const float* queries, int nq, int k) const {
    std::vector<int> all;
    if (!built_ || nq <= 0 || k <= 0) return all;
    all.resize((size_t)nq * k);
#pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < nq; qi++) {
        SearchResult r = search(queries + (size_t)qi * cfg_.dim);
        int base = qi * k;
        for (int j = 0; j < (int)r.indices.size() && j < k; j++)
            all[base + j] = r.indices[j];
        // se o motor retornou menos que k (não deve), preenche com -1
        for (int j = (int)r.indices.size(); j < k; j++)
            all[base + j] = -1;
    }
    return all;
}

// M2 (v1.8.0): persistência do índice (formato binário, mmap-friendly).
// Layout do arquivo:
//   [magic:8B "WMADHAV1"][cfg][n:int32][s1:int32][s2:int32]
//   [P1: s1*D float32][P2: s2*D float32]
//   [pr1: n*s1 int8][pr2: n*s2 int8]
//   [pr1_scale: s1 float32][pr2_scale: s2 float32]
//   [e1: n float32][e2: n float32]
//   [vn: n float32][vn_eff: n float32]
// O corpus bruto NÃO é salvo (fica no disco); o chamador re-anexa o base.
bool MadhavaL2::save_index(const std::string& path) const {
    if (!built_) return false;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const char* magic = "WMADHAV1";
    fwrite(magic, 1, 8, f);
    int n = n_, s1 = cfg_.stage1_dim, s2 = cfg_.stage2_dim;
    int D = cfg_.dim;
    fwrite(&cfg_.dim, sizeof(int), 1, f);
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&s1, sizeof(int), 1, f);
    fwrite(&s2, sizeof(int), 1, f);
    fwrite(P1_, sizeof(float), (size_t)s1 * D, f);
    if (s2 > 0) fwrite(P2_, sizeof(float), (size_t)s2 * D, f);
    if (cfg_.quant == QuantMode::Int8) {
        fwrite(pr1_, 1, (size_t)n * s1, f);
        if (s2 > 0) fwrite(pr2_, 1, (size_t)n * s2, f);
    } else {
        fwrite(pr1_f_, sizeof(float), (size_t)n * s1, f);
        if (s2 > 0) fwrite(pr2_f_, sizeof(float), (size_t)n * s2, f);
    }
    fwrite(pr1_scale_, sizeof(float), (size_t)s1, f);
    if (s2 > 0) fwrite(pr2_scale_, sizeof(float), (size_t)s2, f);
    fwrite(e1_, sizeof(float), (size_t)n, f);
    if (s2 > 0) fwrite(e2_, sizeof(float), (size_t)n, f);
    fwrite(vn_, sizeof(float), (size_t)n, f);
    fwrite(vn_eff_, sizeof(float), (size_t)n, f);
    fclose(f);
    return true;
}

bool MadhavaL2::load_index(const std::string& path) {
    // Limpa o estado atual.
    delete[] P1_; delete[] P2_; delete[] pr1_; delete[] pr2_;
    delete[] pr1_f_; delete[] pr2_f_; delete[] pr1_scale_; delete[] pr2_scale_;
    delete[] e1_; delete[] e2_; delete[] vn_; delete[] vn_eff_;
    P1_ = P2_ = nullptr; pr1_ = pr2_ = nullptr;
    pr1_f_ = pr2_f_ = nullptr; pr1_scale_ = pr2_scale_ = nullptr;
    e1_ = e2_ = vn_ = vn_eff_ = nullptr;
    built_ = false;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8] = {0};
    if (fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "WMADHAV1", 8) != 0) {
        fclose(f); return false;
    }
    int n = 0, s1 = 0, s2 = 0, D = 0;
    if (fread(&D, sizeof(int), 1, f) != 1 || fread(&n, sizeof(int), 1, f) != 1 ||
        fread(&s1, sizeof(int), 1, f) != 1 || fread(&s2, sizeof(int), 1, f) != 1) {
        fclose(f); return false;
    }
    if (n <= 0 || s1 <= 0 || D <= 0 || s1 > 4096 || s2 > 4096) {
        fclose(f); return false;
    }
    cfg_.dim = D;
    cfg_.stage1_dim = s1;
    cfg_.stage2_dim = s2;
    n_ = n;

    P1_ = new float[(size_t)s1 * D];
    if (s2 > 0) P2_ = new float[(size_t)s2 * D];
    pr1_scale_ = new float[s1];
    if (s2 > 0) pr2_scale_ = new float[s2];
    e1_ = new float[n];
    if (s2 > 0) e2_ = new float[n];
    vn_ = new float[n];
    vn_eff_ = new float[n];

    bool ok = true;
    ok &= fread(P1_, sizeof(float), (size_t)s1 * D, f) == (size_t)s1 * D;
    if (s2 > 0) ok &= fread(P2_, sizeof(float), (size_t)s2 * D, f) == (size_t)s2 * D;
    if (cfg_.quant == QuantMode::Int8) {
        pr1_ = new int8_t[(size_t)n * s1];
        ok &= fread(pr1_, 1, (size_t)n * s1, f) == (size_t)n * s1;
        if (s2 > 0) {
            pr2_ = new int8_t[(size_t)n * s2];
            ok &= fread(pr2_, 1, (size_t)n * s2, f) == (size_t)n * s2;
        }
    } else {
        pr1_f_ = new float[(size_t)n * s1];
        ok &= fread(pr1_f_, sizeof(float), (size_t)n * s1, f) == (size_t)n * s1;
        if (s2 > 0) {
            pr2_f_ = new float[(size_t)n * s2];
            ok &= fread(pr2_f_, sizeof(float), (size_t)n * s2, f) == (size_t)n * s2;
        }
    }
    ok &= fread(pr1_scale_, sizeof(float), (size_t)s1, f) == (size_t)s1;
    if (s2 > 0) ok &= fread(pr2_scale_, sizeof(float), (size_t)s2, f) == (size_t)s2;
    // SCAN INT8: the index file stores pr1_f_ (float32); if scan_int8 is on,
    // rebuild the int8 pr1_ from pr1_f_ (the file format is unchanged). Only
    // after pr1_scale_ is read (needed for the quantization).
    if (cfg_.scan_int8 && cfg_.quant == QuantMode::None && ok) {
        pr1_ = new int8_t[(size_t)n * s1];
        for (int i = 0; i < n; i++)
            quantize(pr1_f_ + (size_t)i * s1, pr1_ + (size_t)i * s1, pr1_scale_, s1);
        if (s2 > 0 && pr2_f_ != nullptr) {
            pr2_ = new int8_t[(size_t)n * s2];
            for (int i = 0; i < n; i++)
                quantize(pr2_f_ + (size_t)i * s2, pr2_ + (size_t)i * s2, pr2_scale_, s2);
        }
    }
    ok &= fread(e1_, sizeof(float), (size_t)n, f) == (size_t)n;
    if (s2 > 0) ok &= fread(e2_, sizeof(float), (size_t)n, f) == (size_t)n;
    ok &= fread(vn_, sizeof(float), (size_t)n, f) == (size_t)n;
    ok &= fread(vn_eff_, sizeof(float), (size_t)n, f) == (size_t)n;
    fclose(f);

    built_ = ok;
    return ok;
}

SearchResult MadhavaL2::search_exact(const float* query) const {
    std::vector<float> qn;
    return search_exact(query, qn);
}

SearchResult MadhavaL2::search_exact(const float* query, const std::vector<float>& query_norm) const {
    SearchResult out;
    if (!built_) return out;
    int N = n_, K = cfg_.k;
    bool is_l2 = (cfg_.metric == Metric::L2);
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<float, int>> scores((size_t)N);
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        scores[i] = {exact_score(i, query), i};
    }
    out.bound_pairs = N;
    out.k3 = N;
    // search_exact varre o corpus inteiro (k3 == N): o resultado É o top-K
    // global exato. recall_guarantee é factual, derivado de k3 == N.
    out.recall_guarantee = "exact_global";
    if (is_l2)
        std::partial_sort(scores.begin(), scores.begin() + std::min(K, N), scores.end());
    else
        std::partial_sort(scores.begin(), scores.begin() + std::min(K, N), scores.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
    for (int i = 0; i < std::min(K, N); i++) out.indices.push_back(scores[i].second);
    out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

// ---------------------------------------------------------------------------
// Audited search — the per-document mathematical certificate
// ---------------------------------------------------------------------------
// Consumed by tracer-gov (GovAuditRecord), tracer-med (certificate/QR) and the
// winnex-audit-cpp spec (AuditRecord). The math is 100% the motor's own:
//   - search()        → top-K + threshold (post-filter exact)
//   - ub_raw(layer=1) → Stage-1 upper bound per doc
//   - e1_/e2_         → residuals computed at build
//   - exact_score     → true_cosine on the audited docs
// No bound is reimplemented; this is a thin certifier over the C++ motor.
AuditResult MadhavaL2::search_audited(const float* query, int64_t k,
                                      int64_t max_audit_records) const {
    // WITNESS, NOT JUDGE (2.0.0 architecture):
    // This method does NOT recompute the Cauchy-Schwarz bounds after the fact.
    // It turns on the motor's audit hook (Config::audit_record), runs the real
    // search(), and READS the per-document pruning decision the motor captured
    // AT THE MOMENT it made it (audit_ids / audit_ubs / audit_threshold).
    // The certificate is therefore byte-for-byte identical to the motor's own
    // pruning — eliminating the 1.9.0 false positives (462/973 violations on
    // arXiv d=1536) that came from recomputing bounds with inconsistent state.
    AuditResult out;
    if (!built_) return out;
    int N = n_;
    int K = (int)std::min<int64_t>(k, cfg_.k);
    auto t0 = std::chrono::high_resolution_clock::now();

    // The search() call with the hook enabled captures the exact pruning
    // decision (same loop, same threshold, same ub_raw as the motor).
    SearchResult base = search(query);
    out.base = base;

    const int64_t n_pruned = (int64_t)base.audit_ids.size();
    out.audit_candidates = N;                 // the motor examined all N docs
    out.audit_excluded = n_pruned;            // = pruned_by_bound, by construction

    // WITNESS PATH ONLY (2026-09-03, 1.9.11): the motor's search() ALWAYS
    // captures the per-document pruning decision (audit_ids/audit_ubs/
    // audit_threshold/audit_residuals) — the audit hook at the end of search()
    // is UNCONDITIONAL, so audit_ids is populated whenever the search returned
    // K results. There is NO fallback here by design: this method must NEVER
    // recompute the Cauchy-Schwarz bounds after the fact — the 1.9.0 bug
    // (462/973 false "excluded" on arXiv d=1536) came from exactly that
    // "judge" pattern. The removed `else` branch (rebuild the witness from
    // audit_threshold) was dead code (cfg_.audit_record is never read by
    // search()) AND architecturally wrong.
    //
    // FIX (2026-09-03, 1.9.11): HONOR max_audit_records. The old code
    // materialized an AuditRecord for EVERY excluded doc regardless of the
    // parameter — at d=1536 pca_corpus that was ~20k records per call (~37ms,
    // ~2.7MB audit_json) even with max_audit_records=50. The count
    // (audit_excluded) stays EXACT (it is base.audit_ids.size()); only the
    // per-doc record expansion is capped at max_audit_records (the records most
    // useful for spot-check audit). The top-K in_topk records are few and
    // always included.
    const int64_t mar = std::max<int64_t>(0, max_audit_records);
    const double thr = base.audit_threshold;
    size_t n_excl_records = 0;
    out.audit.reserve((size_t)std::min<int64_t>(base.audit_ids.size(), mar)
                      + base.indices.size());
    // Records for the excluded docs (the witness list), capped at `mar`.
    for (size_t i = 0; i < base.audit_ids.size(); ++i) {
        if ((int64_t)n_excl_records >= mar) break;   // max_audit_records honored
        int64_t id = base.audit_ids[i];
        double ub = base.audit_ubs[i];
        double resid = base.audit_residuals.size() > i ? (double)base.audit_residuals[i]
                                                      : 0.0;   // captured at decision time
        AuditRecord rec;
        rec.doc_id = id;
        // projected_cosine = ub - residual - (quant margin / safety margin),
        // i.e. the components the motor used — recovered, never re-derived.
        double margin = (cfg_.quant == QuantMode::Int8) ? 0.0 : 1e-4;
        rec.projected_cosine = ub - resid - margin;
        rec.residual_norm = resid;
        rec.true_cosine = (double)exact_score((int)id, query);
        rec.upper_bound = ub;             // the motor's exact UB at decision time
        rec.threshold = thr;              // the motor's exact heap[0] threshold
        rec.excluded = true;
        rec.stage = "stage1";
        out.audit.push_back(std::move(rec));
        n_excl_records++;
    }
    // Records for the kept top-K (survived / in_topk).
    for (int idx : base.indices) {
        AuditRecord rec;
        rec.doc_id = idx;
        rec.true_cosine = (double)exact_score(idx, query);
        rec.projected_cosine = 0.0;
        rec.residual_norm = 0.0;
        rec.upper_bound = 0.0;
        rec.threshold = thr;
        rec.excluded = false;
        rec.stage = "in_topk";
        out.audit.push_back(std::move(rec));
    }

    out.base.latency_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

std::string MadhavaL2::audit_json(const float* query, int64_t k,
                                  int64_t max_audit_records) const {
    AuditResult r = search_audited(query, k, max_audit_records);
    std::string json = "{\n  \"query_id\": \"madhava_audit\",\n";
    json += "  \"latency_ms\": " + std::to_string(r.base.latency_ms) + ",\n";
    json += "  \"bound_violations\": " + std::to_string(r.base.bound_violations) + ",\n";
    json += "  \"results\": [";
    for (size_t i = 0; i < r.base.indices.size(); ++i) {
        if (i > 0) json += ",";
        json += "\n    {\"rank\": " + std::to_string(i + 1) +
                ", \"doc_id\": " + std::to_string(r.base.indices[i]) + "}";
    }
    json += "\n  ],\n  \"audit_trail\": [\n";
    int count = 0;
    for (const auto& rec : r.audit) {
        if (count++ > 0) json += ",\n";
        json += "    {\"doc_id\": " + std::to_string(rec.doc_id) +
                ", \"true_cosine\": " + std::to_string(rec.true_cosine) +
                ", \"upper_bound\": " + std::to_string(rec.upper_bound) +
                ", \"threshold\": " + std::to_string(rec.threshold) +
                ", \"excluded\": " + (rec.excluded ? "true" : "false") +
                ", \"stage\": \"" + rec.stage + "\"}";
    }
    json += "\n  ]\n}\n";
    return json;
}

// ---------------------------------------------------------------------------
// Lightweight audited search — the production commitment path
// ---------------------------------------------------------------------------
// Returns a compact AuditCommitment instead of the full O(N) certificate:
//   - total_excluded_count: docs the bound PROVED outside the exact top-K,
//   - global_threshold:     the exact score of the K-th result (the true
//                           global threshold the motor decided with),
//   - sampled_records:      a DETERMINISTIC reservoir sample (up to
//                           max_sample) of the excluded docs nearest the
//                           boundary — for quick spot-check audit.
//
// Memory: O(max_sample), NOT O(N) — the motor never materializes the excluded
// list. The Python compliance service (tracer-gov/tracer-med) signs this
// commitment with Ed25519 (key held outside the C++ binary) for
// non-repudiation, then stores the ~500-byte signed record in the WORM.
//
// The sample is deterministic (fixed PRNG seed derived from the query hash +
// the engine seed): the same query always yields the same sampled doc_ids, so
// an auditor can reproduce it. The sample is biased to the boundary (docs
// whose bound is nearest the threshold — the ones regulators care about),
// then reservoir-filled with a deterministic stride for the rest.
AuditCommitment MadhavaL2::search_with_commitment(const float* query, int64_t k,
                                                  int64_t max_sample) const {
    AuditCommitment out;
    if (!built_) return out;
    int N = n_, D = cfg_.dim, s1 = cfg_.stage1_dim, s2 = cfg_.stage2_dim, K = cfg_.k;
    if (k > 0) K = (int)std::min<int64_t>(k, K);
    bool is_l2 = (cfg_.metric == Metric::L2);
    bool normalize = (cfg_.metric == Metric::Cosine) && cfg_.normalize_input;
    auto t0 = std::chrono::high_resolution_clock::now();

    // --- Reuse the exact same search path as search() to get the real
    // top-K + global threshold. We run search() (which is additive and
    // already returns the post-filter exact top-K + audit_threshold). ---
    SearchResult base = search(query);
    out.indices = base.indices;
    out.bound_pairs = base.bound_pairs;
    out.bound_violations = base.bound_violations;
    out.latency_ms = base.latency_ms;
    // Recall scope of THIS commitment: derived from the underlying search().
    // When the motor only re-scored the pool (k3 < N), the threshold and the
    // sampled exclusions prove exclusion from the POOL top-K, not the global
    // top-K — the WORM record must carry this so it is not over-read.
    out.recall_guarantee = base.recall_guarantee;

    // The global K-th threshold: exact score of the K-th returned result.
    // (Mirrors the 1.9.1 witness fix — the certificate must use the TRUE
    // global threshold, not a pool threshold.)
    if ((int)out.indices.size() < K || K <= 0) {
        out.global_threshold = 0.0f;
        out.total_excluded_count = 0;
        return out;
    }
    float worst = exact_score(out.indices[K - 1], query);
    out.global_threshold = worst;

    // Query projections (same as search() Stage-1) for the bound scan.
    float qn = std::sqrt(dot_f32(query, query, D));
    float qn_eff = normalize ? 1.0f : qn;
    std::vector<float> qbuf;
    const float* qproj = query;
    if (normalize && qn > 1e-10f) {
        qbuf.resize((size_t)D);
        float inv = 1.0f / qn;
        for (int j = 0; j < D; j++) qbuf[j] = query[j] * inv;
        qproj = qbuf.data();
    }
    float pq1[256];
    float q1s = 0;
    for (int j = 0; j < s1; j++) { pq1[j] = dot_f32(qproj, P1_ + (size_t)j * D, D); q1s += pq1[j] * pq1[j]; }
    float qr1 = std::sqrt(std::max(0.0f, qn_eff * qn_eff - q1s));
    // SCAN INT8: the quant margin qm is needed when ub_raw uses the int8
    // projections — native Int8 quant OR the opt-in scan_int8 on a float32
    // corpus (which keeps pr1_ int8 alongside pr1_f_).
    const bool scan_uses_int8 = (cfg_.quant == QuantMode::Int8)
        || (cfg_.scan_int8 && pr1_ != nullptr && pr1_scale_ != nullptr);
    float qm1 = 0;
    if (scan_uses_int8)
        for (int j = 0; j < s1; j++) qm1 += 0.5f * pr1_scale_[j] * std::fabs(pq1[j]);
    float qn2 = qn_eff * qn_eff;

    // --- Count excluded + deterministic boundary-biased sample ---
    // (2026-09-04) The search() call above uses a LOCAL UB buffer now (no
    // shared mutable state), so this commitment cannot read the search's UB.
    // The UB of each doc depends only on (pq1, qr1, qm1) — recomputed here —
    // so we materialize the same per-query UB vector locally. Cost: one O(N)
    // ub_raw pass (the same as search()'s Stage-1 scan); the commitment is an
    // audit path, not the hot query path.
    // Deterministic PRNG: seed from the query hash + engine seed, so the
    // same query always yields the same sample (reproducible by an auditor).
    // We do NOT use a cryptographic RNG here — determinism is the goal.
    uint32_t seed = (uint32_t)cfg_.seed;
    for (int j = 0; j < std::min(D, 16); j++) {
        // fold the first bytes of the (normalized) query into the seed
        uint32_t bits; std::memcpy(&bits, qproj + j, sizeof(uint32_t));
        seed ^= bits + 0x9e3779b9u + (seed << 6) + (seed >> 2);
    }
    std::mt19937 rng(seed);

    long long excluded = 0;
    const int64_t ms = std::max<int64_t>(0, max_sample);
    out.sampled_records.clear();
    out.sampled_records.reserve((size_t)std::min<int64_t>(ms, 4096));
    std::vector<float> ub_local((size_t)N);
    for (int i = 0; i < N; i++) ub_local[(size_t)i] = ub_raw(i, 1, pq1, qr1, qm1);

    // Two-tier deterministic sampling:
    //  (1) Boundary tier: docs whose bound is within `boundary_band` of the
    //      threshold are the highest-value audit records. We collect them
    //      directly (up to max_sample) — deterministic, no RNG.
    //  (2) Reservoir tier: if the boundary band is empty (or we want a spread
    //      sample), fill the rest with a deterministic reservoir over the
    //      excluded set (seeded RNG → reproducible).
    // This biases the sample to the boundary while staying deterministic and
    // bounded. A fixed band of ~1% of the threshold range covers the docs
    // nearest the cut — the ones regulators spot-check.
    float band = 0.01f * (1.0f + std::fabs(worst));
    for (int i = 0; i < N; i++) {
        float ub = ub_local[(size_t)i];   // UB materialized locally above
        bool excl;
        if (is_l2) {
            float lb = vn_eff_[i] * vn_eff_[i] + qn2 - 2.0f * ub;
            excl = lb > worst;
        } else {
            excl = ub < worst;
        }
        if (!excl) continue;
        excluded++;

        if ((int64_t)out.sampled_records.size() < ms) {
            // Boundary tier first (nearest the threshold).
            float gap = is_l2 ? (vn_eff_[i] * vn_eff_[i] + qn2 - 2.0f * ub) - worst
                              : worst - ub;
            if (gap < band || (int64_t)out.sampled_records.size() < 4) {
                AuditSample s; s.doc_id = i; s.upper_bound = ub; s.excluded = true;
                out.sampled_records.push_back(s);
            }
        }
    }
    // Reservoir fill: if we have fewer than max_sample, top up with a
    // deterministic spread over the excluded set (seeded RNG → reproducible).
    // We re-walk in a second pass ONLY if the sample is not yet full, keeping
    // the common case (boundary-rich) O(N) total.
    if ((int64_t)out.sampled_records.size() < ms && excluded > (int64_t)out.sampled_records.size()) {
        // Deterministic reservoir: for the first `seen` excluded we take the
        // slot; afterwards replace with probability ms/seen (seeded).
        long long seen = 0;
        std::vector<AuditSample> reservoir;
        reservoir.reserve((size_t)ms);
        for (int i = 0; i < N; i++) {
            float ub = ub_local[(size_t)i];   // local UB (materialized above)
            bool excl;
            if (is_l2) {
                float lb = vn_eff_[i] * vn_eff_[i] + qn2 - 2.0f * ub;
                excl = lb > worst;
            } else {
                excl = ub < worst;
            }
            if (!excl) continue;
            seen++;
            if ((int64_t)reservoir.size() < ms) {
                AuditSample s; s.doc_id = i; s.upper_bound = ub; s.excluded = true;
                reservoir.push_back(s);
            } else {
                uint64_t r = rng();
                if ((r % (uint64_t)seen) < (uint64_t)ms) {
                    size_t slot = (size_t)(r % (uint64_t)ms);
                    reservoir[slot].doc_id = i;
                    reservoir[slot].upper_bound = ub;
                }
            }
        }
        // Merge: boundary samples first, then reservoir fill.
        for (auto& s : reservoir) {
            if ((int64_t)out.sampled_records.size() >= ms) break;
            out.sampled_records.push_back(s);
        }
    }

    out.total_excluded_count = excluded;
    out.latency_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

} // namespace winnex_madhava
