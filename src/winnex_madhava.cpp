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
                     int sample_cap, bool normalize, float* P /* s×D row-major */) {
    if (s <= 0 || s > D) return false;
    int sample = std::min(n, sample_cap);
    if (sample < 1) return false;

    // 1. Load a subsample of the corpus, optionally L2-normalized.
    std::mt19937 rng(seed);
    std::vector<float> A((size_t)sample * D);
    for (int i = 0; i < sample; i++) {
        int src = (int)(rng() % (unsigned)n);
        const float* v = base_f32 + (size_t)src * D;
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
    //    A cyclic-sweep Jacobi under-converges at d=1536 in float32 (the
    //    off-diagonal tolerance is unreachable), so we use the robust
    //    bound-guided power iteration here.
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
        for (int it = 0; it < 200; ++it) {
            // w = C·v
            std::vector<float> work(D, 0.0f);
            for (int i = 0; i < D; ++i) {
                float s_ = 0.0f;
                const float* Ci = W.data() + (size_t)i * D;
                for (int j = 0; j < D; ++j) s_ += Ci[j] * v[j];
                work[i] = s_;
            }
            // MGS re-orthogonalization against previously found directions.
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
            for (int j = 0; j < D; ++j) Wi[j] -= (float)(lambda * v[i] * v[j]);
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
    for (int i = 0; i < n_; i++) {
        const float* base_v;
        if (corpus_f32_ != nullptr) base_v = corpus_f32_ + (size_t)i * D;
        else { for (int j = 0; j < D; j++) buf[j] = (float)base_[(size_t)i * D + j]; base_v = buf.data(); }
        float nn = 0.0f;
        for (int j = 0; j < D; j++) nn += base_v[j] * base_v[j];
        float vn2 = normalize ? 1.0f : nn;
        // recompute Stage-1 projection + residual
        float pn1 = 0.0f;
        for (int j = 0; j < s1; j++) {
            float d = dot_f32(base_v, P1_ + (size_t)j * D, D);
            pr1_f_[(size_t)i * s1 + j] = d;
            pn1 += d * d;
        }
        e1_[i] = std::sqrt(std::max(0.0f, vn2 - pn1));
        if (s2 > 0) {
            float pn2 = 0.0f;
            for (int j = 0; j < s2; j++) {
                float d = dot_f32(base_v, P2_ + (size_t)j * D, D);
                pr2_f_[(size_t)i * s2 + j] = d;
                pn2 += d * d;
            }
            e2_[i] = std::sqrt(std::max(0.0f, vn2 - pn2));
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
                                  cfg_.pca_sample, cfg_.normalize_input, P1_);
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
                                      cfg_.pca_sample, cfg_.normalize_input, P2_);
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
    }
    if (s2 > 0) {
        e2_ = new float[n];
        pr2_scale_ = new float[s2];
        if (cfg_.quant == QuantMode::Int8) {
            pr2_ = new int8_t[(size_t)n * s2];
        } else {
            pr2_f_ = new float[(size_t)n * s2];
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
    float qm1 = 0;
    if (cfg_.quant == QuantMode::Int8)
        for (int j = 0; j < s1; j++) qm1 += 0.5f * pr1_scale_[j] * std::fabs(pq1[j]);
    float qn2 = qn_eff * qn_eff;

    // Stage-1: bound pruning over all N.
    out.bound_pairs = N;
    std::vector<std::pair<float, int>> b1((size_t)N);
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float ub = ub_raw(i, 1, pq1, qr1, qm1);
        // Score for sorting:
        //   L2:    lb = ‖v‖² + ‖q‖² − 2·ub   (ascending = best)
        //   Cosine: ub (descending = best) — sort ascending on −ub
        float score = is_l2 ? (vn_eff_[i] * vn_eff_[i] + qn2 - 2.0f * ub) : -ub;
        b1[i] = {score, i};
    }
    int k1 = std::max(cfg_.k1_min, (int)(N * cfg_.k1_fraction));
    if (k1 > N) k1 = N;
    if (k1 < N) std::nth_element(b1.begin(), b1.begin() + k1, b1.end());
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
        if (cfg_.quant == QuantMode::Int8)
            for (int j = 0; j < s2; j++) qm2 += 0.5f * pr2_scale_[j] * std::fabs(pq2[j]);

        b2.resize((size_t)k1);
#pragma omp parallel for
        for (int i = 0; i < k1; i++) {
            int vi = b1[i].second;
            float ub2 = ub_raw(vi, 2, pq2, qr2, qm2);
            float score2 = is_l2 ? (vn_eff_[vi] * vn_eff_[vi] + qn2 - 2.0f * ub2) : -ub2;
            b2[i] = {score2, vi};
        }
        int k2t = std::max(cfg_.k2_min, (int)(N * cfg_.k2_fraction));
        if (k2t > k1) k2t = k1;
        // Cap Stage-2 survivors (streaming/100M: limit post-filter cost).
        if (k2t > cfg_.k2_max) k2t = cfg_.k2_max;
        std::nth_element(b2.begin(), b2.begin() + k2t, b2.end());
        k2 = k2t;
        survivors = &b2;
    }
    out.k2 = k2;

    // Rank the survivors:
    //   - If modulation is on, rank by modulated = B1 + α(B2−B1)
    //   - Else rank by the (already computed) bound score.
    std::vector<std::pair<float, int>> ranked(k2);
    double mod_gain = 0.0;
    if (cfg_.modulation && s2 > 0) {
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
        if (cfg_.quant == QuantMode::Int8) {
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
    if (cfg_.postfilter) {
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
        if (cfg_.early_exit) {
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
        long long n_bound_pruned = 0;
        if ((int)heap.size() == K) {
            float worst = heap[0].first;   // K-th best exact score
            for (int i = 0; i < N; i++) {
                float ub = ub_raw(i, 1, pq1, qr1, qm1);
                if (is_l2) {
                    // L2: exact L2² ≥ lb. If lb > worst, v is proven outside top-K.
                    float lb = vn_eff_[i] * vn_eff_[i] + qn2 - 2.0f * ub;
                    if (lb > worst) n_bound_pruned++;
                } else {
                    // cosine: exact cos ≤ UB. If UB < worst, v is proven outside.
                    if (ub < worst) n_bound_pruned++;
                }
            }
        }
        out.pruned_by_bound = n_bound_pruned;
        out.pruned_by_prefilter = (long long)N - n_exact - n_bound_pruned;

        if (cfg_.early_exit && (int)heap.size() == K) {
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
    fread(&D, sizeof(int), 1, f);
    fread(&n, sizeof(int), 1, f);
    fread(&s1, sizeof(int), 1, f);
    fread(&s2, sizeof(int), 1, f);
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
    if (is_l2)
        std::partial_sort(scores.begin(), scores.begin() + std::min(K, N), scores.end());
    else
        std::partial_sort(scores.begin(), scores.begin() + std::min(K, N), scores.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
    for (int i = 0; i < std::min(K, N); i++) out.indices.push_back(scores[i].second);
    out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

} // namespace winnex_madhava
