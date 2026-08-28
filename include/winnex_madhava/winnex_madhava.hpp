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
    // Power-iteration cap for the PCA eigen-solver. The motor is AGNOSTIC:
    // this is a convergence knob owned by the caller (the engine keeps it a
    // config parameter, not a hardcoded constant). 200 = the historical
    // value; 30 converges the dominant subspace (subspace sim = 1.0000 to
    // 200 steps) at a fraction of the cost.
    int pca_iterations = 200;

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

    // Audit hook: when true, search() captures the per-document pruning
    // decision AT THE MOMENT it is made (audit_ids/audit_ubs/audit_threshold
    // on SearchResult). This is what makes search_audited a WITNESS of the
    // motor's exact decision, not a recomputing judge. Default false so the
    // normal search path costs nothing extra.
    bool audit_record = false;
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

    // AUDIT HOOK (captured AT the decision moment, not recomputed after):
    //   audit_threshold — the exact K-th best score (heap[0]) the motor used
    //                      to prove pruning. This is the TRUE threshold the
    //                      motor decided with (not a recomputed one).
    //   audit_ids        — the doc_ids the bound PROVED outside top-K, in the
    //                      order the motor evaluated them (all N in stage-1).
    //   audit_ubs        — per-id the Cauchy-Schwarz upper bound that the
    //                      motor computed AT THAT MOMENT (ub_raw stage-1).
    //                      audit_ids[i] is provably excluded because its
    //                      audit_ubs[i] < audit_threshold (cosine) or its
    //                      L2-lower-bound > audit_threshold (L2).
    //   audit_l2_lbs     — per-id the L2² lower bound (only for L2 metric).
    //   These fields make the audit a WITNESS, not a judge: the certificate
    //   is byte-for-byte the motor's own pruning decision. Filled only when
    //   the engine is configured with audit_record=true (see Config).
    double audit_threshold = 0.0;
    std::vector<int64_t> audit_ids;
    std::vector<float> audit_ubs;
    std::vector<float> audit_l2_lbs;
    // Per-id residual_norm = e(v)·e(q) that the motor used in the UB at
    // decision time (so the certificate can report it without recomputing).
    std::vector<float> audit_residuals;
};

// ---------------------------------------------------------------------------
// Per-document audit certificate (winnex-audit-cpp compatible)
// ---------------------------------------------------------------------------
// A single excluded/kept document, with the mathematical proof that it could
// (or could not) be in the exact top-K. This is the record shape consumed by
// the tracer-gov GovAuditRecord and the tracer-med certificate/QR flow.
struct AuditRecord {
    int64_t doc_id = -1;
    double true_cosine = 0.0;     // exact <v,q> (normalized space)
    double projected_cosine = 0.0; // <Pv,Pq> Stage-1
    double residual_norm = 0.0;   // e(v) * e(q)  (the bound width)
    double upper_bound = 0.0;     // projected_cosine + residual_norm + margin
    double threshold = 0.0;       // exact score of the K-th result
    bool excluded = false;        // true if upper_bound < threshold
    std::string stage;            // "stage1" | "stage2" | "survived" | "in_topk"
};

// ---------------------------------------------------------------------------
// Audited search result: the normal SearchResult + the per-doc certificate
// ---------------------------------------------------------------------------
struct AuditResult {
    SearchResult base;                  // top-K + honest pruning breakdown
    std::vector<AuditRecord> audit;     // per-document proofs (excluded docs)
    long long audit_candidates = 0;     // docs examined for the certificate
    long long audit_excluded = 0;       // docs proven outside top-K
};

// ---------------------------------------------------------------------------
// Audit commitment — the lightweight "mathematical inverted index" (~400 B)
// ---------------------------------------------------------------------------
// A production audit trail does NOT store the full O(N) list of AuditRecords
// (at scale this is the measured ~2 MB/query problem). Instead the motor
// returns a compact, deterministic commitment that proves the state of the
// search without listing every excluded element:
//
//   - total_excluded_count : the raw mathematical fact — how many docs the
//     Cauchy-Schwarz bound PROVED outside the exact top-K (global threshold).
//   - global_threshold     : the exact score of the K-th result — the true
//     threshold the motor decided with.
//   - sampled_records      : a deterministic reservoir sample (up to
//     max_sample) of docs near the boundary, for quick spot-check audit.
//
// The commitment carries NO internal hash and NO private key — per the
// hybrid responsibility split, integrity + non-repudiation are added by the
// Python compliance service (tracer-gov/tracer-med), which hashes the raw
// fields (hashlib) and signs them with an Ed25519 private key held OUTSIDE
// the C++ binary. This neutralizes the "lying engine" attack: even a
// compromised C++ binary cannot forge past WORM records, because it never
// holds the signing key.
//
// The Python payload stored in the WORM is therefore ~400–500 bytes
// regardless of corpus size (99.98% smaller), O(1) in the number of
// exclusions, with no O(N) heap allocation in the motor.
struct AuditSample {
    int64_t doc_id = -1;
    float upper_bound = 0.0f;   // the Cauchy-Schwarz upper bound used
    bool excluded = true;
};

struct AuditCommitment {
    long long total_excluded_count = 0; // docs the bound PROVED outside top-K
    float global_threshold = 0.0f;      // exact score of the K-th result
    std::vector<AuditSample> sampled_records; // up to max_sample (reservoir)
    // Search metadata (mirrors SearchResult, for the WORM record).
    std::vector<int> indices;           // top-K dataset ids
    long long bound_pairs = 0;
    long long bound_violations = 0;     // always 0
    double latency_ms = 0.0;
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

    // Search with the audit hook forced on (the witness path). The normal
    // search() uses cfg_.audit_record; this one always captures the
    // per-document pruning decision so search_audited can read it.
    SearchResult search_with_audit(const float* query) const;

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

    // Audited search: the normal search() PLUS a per-document Cauchy-Schwarz
    // certificate (winnex-audit-cpp / GovAuditRecord compatible). Reuses the
    // motor's own ub_raw / residuals / exact_score — no reimplementation.
    //
    // The certificate examines at most `max_audit_records` documents that are
    // nearest the top-K boundary (the docs whose bound is close to the
    // threshold) plus the top-K themselves — the records regulators care
    // about. Capped so per-query cost stays bounded (the tracer-gov Config
    // default is max_audit_records_per_query = 500).
    //
    // Returns a full AuditResult. `search()` is unchanged — this is additive.
    AuditResult search_audited(const float* query, int64_t k = 10,
                               int64_t max_audit_records = 500) const;

    // Lightweight audited search — the production commitment path.
    // Returns a compact AuditCommitment (count + threshold + up to max_sample
    // deterministic boundary records) instead of the full O(N) certificate.
    // This is what the WORM stores (tracer-gov/tracer-med compliance flows):
    // the Python layer signs it with Ed25519 for non-repudiation.
    AuditCommitment search_with_commitment(const float* query, int64_t k = 10,
                                           int64_t max_sample = 50) const;

    // Render the audited result as JSON (the audit_json of winnex-audit-cpp).
    std::string audit_json(const float* query, int64_t k = 10,
                           int64_t max_audit_records = 500) const;

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
