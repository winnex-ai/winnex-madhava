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
