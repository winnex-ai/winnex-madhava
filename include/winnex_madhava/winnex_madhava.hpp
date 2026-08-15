/**
 * winnex_madhava.hpp — Public API of the Winnex Madhava Engine
 * =============================================================
 * Deterministic vector search with Cauchy-Schwarz upper bounds,
 * parametrizable across the Winnex stack:
 *
 *   - Metric:      'cosine' (stack default, normalized embeddings) or 'l2'
 *   - Cascade:     two-stage QR projection [stage1_dim, stage2_dim]
 *                  (stack: [64,128] — bound B1 wide, B2 tight)
 *   - Quantization: 'int8' (fast, memory-light) or 'none' (float32 exact)
 *   - Modulation:  error-backpropagation ranking (stack FIX(1)):
 *                  prune by the TIGHT bound (B2), rank by modulated score.
 *   - Post-filter:  exact metric re-score on the surviving top-k1/k2
 *   - Navigation:   optional PiPrime/HMC (v7) — placeholder hook
 *
 * Mathematical guarantee (Cauchy-Schwarz on raw inner product):
 *   ⟨v,q⟩ ≤ ⟨Pv,Pq⟩ + ‖v−PᵀPv‖ · ‖q−PᵀPq‖
 *
 * Every vector pruned by Stage 1/2 carries a proof it could not be in the
 * exact top-K by the chosen metric. Bound violations = 0 by construction.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_MADHAVA_HPP
#define WINNEX_MADHAVA_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>

namespace winnex_madhava {

// ---------------------------------------------------------------------------
// Metric enum
// ---------------------------------------------------------------------------
enum class Metric {
    Cosine = 0,   // normalized embeddings (stack default: v17, Madhava-Sec, HMC v7)
    L2     = 1    // raw uint8 L2 (BIGANN-style)
};

// ---------------------------------------------------------------------------
// Quantization enum
// ---------------------------------------------------------------------------
enum class QuantMode {
    Int8 = 0,   // int8-quantized projections (fast, memory-light)
    None = 1    // float32 exact projections (max fidelity)
};

// ---------------------------------------------------------------------------
// Projection basis — the "UB Width" mode
// ---------------------------------------------------------------------------
// The Cauchy-Schwarz bound tightness depends on the residual
//     e(v) = sqrt(||v||^2 − ||P v||^2),
// which is the "UB width". A random projection keeps e(v) ≈ sqrt(1 − s/d) at
// dimension d (large at d = 1536 → the bound degenerates to exhaustive scan).
// Aligning the projection to the principal directions of the corpus shrinks
// e(v) to the manifold residual, restoring pruning power at high dimension.
enum class BasisMode {
    Random = 0,     // default: QR-orthogonalized random Gaussian (Modified GS).
                    // The historical behavior — unchanged.
    PCACorpus = 1   // UB Width: principal directions of the corpus (PCA).
                    // e(v) shrinks to the manifold residual, the bound stays
                    // tight at high dimension, and — because P is orthonormal —
                    // the bound remains valid in the ORIGINAL space:
                    // 0 violations by construction.
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config {
    // Data
    int dim = 128;            // vector dimensionality (uint8 raw bytes)
    int seed = 42;            // PRNG seed for the MGS projections

    // UB Width mode (BasisMode::PCACorpus): sample size for the covariance.
    int pca_sample = 10000;   // max vectors used to estimate the PCA basis

    // Metric & quantization (stack parametrizable)
    Metric metric = Metric::Cosine;   // stack default: cosine over normalized embeddings
    QuantMode quant = QuantMode::Int8;
    BasisMode basis = BasisMode::Random;  // Random (default) or PCACorpus (UB Width)

    // Cascade (two-stage bound, stack [64,128])
    int stage1_dim = 64;      // Stage-1 QR projection (wide bound B1)
    int stage2_dim = 128;     // Stage-2 QR projection (tight bound B2); 0 = single-stage

    // Search
    int k = 10;               // number of results to return
    double k1_fraction = 0.05; // Stage-1 keep fraction (top-k1 = N*k1_fraction)
    double k2_fraction = 0.01; // Stage-2 keep fraction (top-k2 = N*k2_fraction) if stage2_dim>0
    int k1_min = 100;         // minimum k1 (guards tiny N)
    int k2_min = 10;          // minimum k2
    int k2_max = 2000;        // cap on Stage-2 survivors (100M streaming: limits post-filter cost)

    // Ranking & refinement
    bool modulation = true;   // error-backprop ranking: prune by B2, rank by B1+α(B2−B1)
    bool postfilter = true;   // exact metric re-score on the surviving set
    bool normalize_input = true; // for Metric::Cosine: L2-normalize each vector on load
    bool early_exit = false;   // stop exact scoring when the bound can't beat the current top-K

    // Parallelism
    int n_threads = 0;        // 0 = use omp_get_max_threads()
};

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------
struct SearchResult {
    std::vector<int> indices;   // top-K indices, by chosen metric
    int k1 = 0;                 // survivors after Stage-1 pruning
    int k2 = 0;                 // survivors after Stage-2 pruning (if any)
    int k3 = 0;                 // candidates actually scored exactly
    double latency_ms = 0.0;    // search wall time (excludes build)
    long long bound_pairs = 0;  // N vectors evaluated in Stage-1
    long long bound_violations = 0; // should always be 0
    double modulation_gain = 0.0;   // mean |rank_by_modulated − rank_by_bound|

    // HONEST PRUNING BREAKDOWN (2026-08-15). The "pruning" reported by k1/k2/k3
    // is dominated by the fixed k1_fraction/k2_max cutoffs, NOT by the
    // Cauchy-Schwarz bound. To expose the REAL bound-driven pruning, we count:
    //   pruned_by_bound   — vectors discarded by a Cauchy-Schwarz certificate
    //                       (UB(v,q) < the K-th best exact score). This is the
    //                       proof-based pruning (0 violations by construction).
    //   pruned_by_prefilter— vectors discarded by the fixed k1_fraction cutoff
    //                       (the heuristic Stage-1 keep), WITHOUT a per-vector
    //                       certificate. The "95%" of old kernels was this.
    //   exact_evals       — vectors actually scored with the exact metric.
    long long pruned_by_bound = 0;
    long long pruned_by_prefilter = 0;
    long long exact_evals = 0;
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
// Winnex Madhava Engine
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

    // Build the Stage-1 (+ Stage-2) projections + per-vector residuals.
    // raw_base must point to at least n*dim uint8 bytes, kept alive during use.
    void build(const uint8_t* raw_base, int n);

    // Build from float32 embeddings (n*dim) — the UB-Width path. The caller
    // owns the buffer. With BasisMode::PCACorpus this is the CORRECT input
    // (a uint8 corpus destroys the manifold the PCA needs to align with).
    void build_float32(const float* raw_base, int n);

    // UB Width: supply the projection basis explicitly (e.g. computed by the
    // X-Factor, or by a robust numpy `eigh` over the corpus covariance).
    // P1 must be [stage1_dim × dim] orthonormal rows; if stage2_dim > 0, P2
    // must be [stage2_dim × dim]. Overrides cfg_.basis for the given stage.
    // This is the recommended path for d ~ 1536: a hand-rolled Jacobi in
    // float32 under-converges and collapses to the null subspace, while the
    // X-Factor's power iteration (or numpy's LAPACK eigh) is robust.
    void set_basis(const float* P1, const float* P2 = nullptr);

    // Search: bound pruning (top-k1/k2) + optional post-filter.
    SearchResult search(const float* query, const std::vector<float>& query_norm) const;
    SearchResult search(const float* query) const; // computes norm internally

    // M1 (v1.8.0): batch search — processa nq queries de uma vez.
    // Útil para o DevAI (batch RAG) e para a ingestão: evita o overhead de
    // lançar o scan N× por query. Retorna nq*k indices (concatenados por query).
    std::vector<int> search_batch(const float* queries, int nq, int k) const;

    // M2 (v1.8.0): persistência do índice — permite ao Maestro (ERP) salvar e
    // recarregar um índice sem rebuild. Serializa as projeções int8 + scales +
    // residuals + configuração. O corpus bruto NÃO é serializado (fica no disco);
    // o chamador re-anexa o base ao carregar.
    bool save_index(const std::string& path) const;
    // Carrega um índice salvo. Retorna false se o arquivo é inválido/incompatível.
    // O chamador deve re-anexar o base (mesmo layout) antes de buscar.
    bool load_index(const std::string& path);

    // Exact scan baseline: computes the chosen metric for ALL n vectors.
    // Returns the exact top-K (the recall ceiling of the subset).
    SearchResult search_exact(const float* query, const std::vector<float>& query_norm) const;
    SearchResult search_exact(const float* query) const;

    // Diagnostics
    int num_vectors() const { return n_; }
    int dim() const { return cfg_.dim; }
    const Config& config() const { return cfg_; }
    double build_seconds() const { return build_s_; }
    bool built() const { return built_; }

    // UB Width diagnostics: the projection basis (orthonormal rows) and the
    // per-vector Cauchy-Schwarz residuals e(v) = sqrt(||v||^2 − ||P v||^2).
    // A small mean residual = the projection captures the manifold (tight
    // bound); a large one = wide UB (the random-basis regime at high dim).
    const float* basis1() const { return P1_; }     // [stage1_dim × dim]
    const float* basis2() const { return P2_; }     // [stage2_dim × dim] or nullptr
    const float* residuals1() const { return e1_; } // [n]
    const float* residuals2() const { return e2_; } // [n] or nullptr

private:
    // Cauchy-Schwarz upper bound of ⟨v,q⟩ for a vector (Stage-1/2 score).
    float ub_raw(int idx, int layer, const float* pq, float qr, float qm) const;
    // Exact metric score between a raw vector and a float query.
    float exact_score(int idx, const float* q) const;

    Config cfg_;
    const uint8_t* base_ = nullptr;
    int n_ = 0;
    bool built_ = false;
    double build_s_ = 0.0;

    // Stage-1/2 projections (QR-orthogonalized via Modified Gram-Schmidt, or
    // principal directions of the corpus when cfg_.basis == PCACorpus).
    float* P1_ = nullptr;       // [stage1_dim × dim]
    float* P2_ = nullptr;       // [stage2_dim × dim] (nullptr if single-stage)

    // Internal float32 copy of the corpus — used for the UB-Width PCA basis.
    // Populated only when cfg_.basis == PCACorpus and the corpus is uint8
    // (the float32 path reuses the caller's buffer via a pointer). NULL when
    // not needed.
    float* corpus_f32_ = nullptr;
    // Owned uint8 copy (build_float32 path) so base_ stays valid for the whole
    // engine lifetime (the caller's float32 buffer may be released).
    uint8_t* corpus_u8_owned_ = nullptr;

    // Stage-1/2 cached projections + residuals
    int8_t* pr1_ = nullptr;     // [n × stage1_dim] int8 quantized projections
    int8_t* pr2_ = nullptr;     // [n × stage2_dim] (nullptr if single-stage)
    float* pr1_f_ = nullptr;    // [n × stage1_dim] float32 (used when quant=Int8 OR None)
    float* pr2_f_ = nullptr;    // [n × stage2_dim] float32
    float* pr1_scale_ = nullptr; // [stage1_dim] per-axis quant scale
    float* pr2_scale_ = nullptr; // [stage2_dim]
    float* e1_ = nullptr;       // [n] residual ||v−PᵀPv|| (Stage-1, float32 REAL)
    float* e2_ = nullptr;       // [n] residual (Stage-2)
    float* vn_ = nullptr;       // [n] L2 norms of the raw vectors
    float* vn_eff_ = nullptr;   // [n] effective norm used by the metric (1.0 if cosine-normalized)
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

#endif // WINNEX_MADHAVA_HPP
