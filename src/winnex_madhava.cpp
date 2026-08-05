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
}

void MadhavaL2::build(const uint8_t* raw_base, int n) {
    base_ = raw_base;
    n_ = n;
    built_ = true;
    int D = cfg_.dim, s1 = cfg_.stage1_dim, s2 = cfg_.stage2_dim;
    bool normalize = (cfg_.metric == Metric::Cosine) && cfg_.normalize_input;
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Build the QR-orthogonalized projection(s) via MGS.
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
    if (s2 > 0) {
        P2_ = new float[(size_t)s2 * D];
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
    const int CHUNK = 500000;
    std::vector<float> ch((size_t)CHUNK * D);
    int p = 0;
    while (p < n) {
        int nt = std::min(CHUNK, n - p);
        for (int i = 0; i < nt; i++) {
            float raw_norm;
            load_raw(base_ + (size_t)(p + i) * D, ch.data() + (size_t)i * D, D, normalize);
            vn_[p + i] = raw_norm;
            vn_eff_[p + i] = normalize ? 1.0f : raw_norm;
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
        float inner = 0;
        for (int j = 0; j < s; j++) inner += (float)pr[j] * scale[j] * pq[j];
        float e = (layer == 1) ? e1_[idx] : e2_[idx];
        return inner + e * qr + qm + 1e-5f;
    } else {
        const float* prf = (layer == 1) ? pr1_f_ + (size_t)idx * s : pr2_f_ + (size_t)idx * s;
        float inner = 0;
        for (int j = 0; j < s; j++) inner += prf[j] * pq[j];
        float e = (layer == 1) ? e1_[idx] : e2_[idx];
        return inner + e * qr;
    }
}

// Exact metric score between a raw vector and a float query.
float MadhavaL2::exact_score(int idx, const float* q) const {
    if (cfg_.metric == Metric::L2) {
        return l2_sq(base_ + (size_t)idx * cfg_.dim, q, cfg_.dim);
    } else {
        // Cosine on raw uint8 vs a (possibly un-normalized) float query.
        float dot = 0, vn2 = 0, qn2 = 0;
        for (int j = 0; j < cfg_.dim; j++) {
            float v = (float)base_[(size_t)idx * cfg_.dim + j];
            dot += v * q[j];
            vn2 += v * v;
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
    if (cfg_.postfilter) {
        std::vector<std::pair<float, int>> exact(k2);
#pragma omp parallel for
        for (int i = 0; i < k2; i++) {
            int vi = ranked[i].second;
            exact[i] = {exact_score(vi, query), vi};
        }
        out.k3 = k2;
        // Sort: L2 ascending, cosine descending.
        if (is_l2)
            std::partial_sort(exact.begin(), exact.begin() + std::min(K, k2), exact.end());
        else {
            std::partial_sort(exact.begin(), exact.begin() + std::min(K, k2), exact.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
        }
        for (int i = 0; i < std::min(K, k2); i++) out.indices.push_back(exact[i].second);
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
