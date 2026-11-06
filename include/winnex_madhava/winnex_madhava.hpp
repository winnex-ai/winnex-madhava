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
#include <mutex>

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
    int n_threads = 0;        // 0 = use omp_get_max_threads(); >0 caps the
                              // OpenMP thread count (implemented 2026-09-06)

    // Audit hook: search() ALWAYS captures the per-document pruning decision
    // AT THE MOMENT it is made (audit_ids/audit_ubs/audit_threshold/
    // audit_residuals on SearchResult) — the hook at the end of search() is
    // UNCONDITIONAL since 1.9.1. This is what makes search_audited a WITNESS
    // of the motor's exact decision, not a recomputing judge (there is NO
    // fallback that recomputes bounds — removed in 1.9.11).

    // EXHAUSTIVE AUDIT MODE (2026-09-03): when true, search() FORCES the
    // post-filter pool to cover the ENTIRE corpus (k1 = k2 = N) and disables
    // early_exit, so the exact re-score is GLOBAL — the returned top-K IS the
    // exact top-K of the whole corpus, and SearchResult.recall_guarantee is
    // always "exact_global".
    //
    // WHY: the default `k1_fraction` prefilter cuts candidates WITHOUT a proof
    // (pruned_by_prefilter). When the embedding manifold is weak (e.g. a
    // degraded corpus), the pool may not contain the true global top-K, and
    // search() can return a pool-top-K that differs from the exact top-K while
    // still reporting bound_violations == 0 (the guarantee is per-document
    // bound-correctness WITHIN the pool, not global recall).
    //
    // audit_exhaustive=true is the price of a GLOBAL guarantee: O(N·d) per
    // query (no recall pruning). Compliance/WORM consumers (tracer-gov /
    // tracer-med) that sign a certificate should use this mode so the
    // certificate covers the whole corpus. Default false (historical behavior,
    // no perf change).
    bool audit_exhaustive = false;

    // SCAN INT8 (2026-09-03): for a FLOAT32 corpus (quant=QuantMode::None), the
    // motor keeps ONLY the float32 projections pr1_f_ (4B/doc/dim) and the
    // Stage-1 scan reads them from DRAM every query — memory-bound at large N
    // (measured: pr1_f_ = N·s1·4B exceeds L3 beyond ~40k docs and the scan
    // degrades). When scan_int8=true, the motor ALSO quantizes the Stage-1
    // projections to int8 (pr1_ + pr1_scale_, 1B/doc/dim) and ub_raw uses
    // dot_int8_scaled for the scan — ~2-2.4× faster (validated: prototype
    // measured 1.6-2.4× with 0 dangerous flips: the int8 bound never excludes a
    // doc the float32 bound keeps, so recall is preserved; the quant margin qm
    // covers the error). pr1_f_ is RETAINED for e(v)/exact (the residual needs
    // the real float32 projection; only the scan uses int8).
    //
    // Memory cost: +N·s1 bytes (int8) on top of N·s1·4B (float32) = +25%.
    // Default false: identical behavior to before (float32 scan). Callers at
    // large N who want the faster scan opt in.
    bool scan_int8 = false;
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
    //   is byte-for-byte the motor's own pruning decision.
    //
    //   GATED since 2026-10-30 (was unconditional, which was a regression):
    //   the per-document materialization AND the global-threshold scan
    //   (O(N·d) + N·4 bytes of scratch) run only when the caller asks for a
    //   certificate (search_audited / search_with_commitment /
    //   collect_audit=true). The plain search() returns the top-K and reports
    //   `collect_audit_used = false`; in that mode audit_threshold is the
    //   K-th score WITHIN THE POOL and must not be read as the corpus
    //   threshold. Release 1.9.15 dropped this gate while its comment still
    //   promised it, making every plain search pay a full exact rescan.
    double audit_threshold = 0.0;
    // True when the caller requested the audit (so the fields below and the
    // global threshold are meaningful). False on the plain search() path.
    bool collect_audit_used = false;
    std::vector<int64_t> audit_ids;
    std::vector<float> audit_ubs;
    std::vector<float> audit_l2_lbs;
    // Per-id residual_norm = e(v)·e(q) that the motor used in the UB at
    // decision time (so the certificate can report it without recomputing).
    std::vector<float> audit_residuals;

    // RECALL GUARANTEE (2026-09-03): a FACTUAL statement of whether the
    // returned top-K is the exact top-K of the ENTIRE corpus or only the best
    // within the post-filter pool.
    //
    //   "exact_global" — the post-filter scored all N vectors (k3 == N, which
    //                    is guaranteed by Config::audit_exhaustive or a
    //                    k1_fraction high enough that k1 == N). The returned
    //                    top-K IS the global exact top-K.
    //   "pool_only"    — the post-filter scored only the k1/k2 survivors
    //                    (k3 < N). The returned top-K is the best WITHIN the
    //                    pool; docs cut by the prefilter (pruned_by_prefilter)
    //                    were discarded WITHOUT a per-vector proof, so the
    //                    global top-K is NOT guaranteed.
    //
    // This is derived from observed state, NOT a promise: when the bound is
    // loose the motor cannot prove whether a doc outside the pool belongs in
    // the global top-K, so "pool_only" is the honest answer.
    // bound_violations == 0 does NOT imply "exact_global".
    std::string recall_guarantee = "pool_only";  // "pool_only" | "exact_global"
    // WHICH route established the label above (2026-10-30):
    //   "scored_all"         — the exact post-filter scored all N (k3 == N);
    //   "proven_exclusions"  — the bound proved pruned_by_bound >= N-K docs
    //                          outside the top-K, so the returned K ARE the
    //                          global top-K by proof;
    //   "none"               — neither; the label is "pool_only".
    // Exposed so a reader never has to infer the reason from k3.
    std::string recall_guarantee_basis = "none";
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
    float global_threshold = 0.0f;      // exact score of the K-th result OF THE CORPUS
    std::vector<AuditSample> sampled_records; // up to max_sample (reservoir)
    // Search metadata (mirrors SearchResult, for the WORM record).
    std::vector<int> indices;           // top-K dataset ids
    long long bound_pairs = 0;
    long long bound_violations = 0;     // always 0
    double latency_ms = 0.0;
    // PRUNING BREAKDOWN (2026-10-30) — WHY total_excluded_count is what it is.
    //
    // Without these two the auditor sees `total_excluded_count = 0` and cannot
    // tell the two very different causes apart:
    //   - the bound found nothing to prove (loose bound), or
    //   - the bound never got the chance, because a fixed k1_fraction cutoff
    //     discarded the corpus first.
    // Measured on a 20k x 384d corpus with stage1=64 (the historical default):
    // pruned_by_bound = 0, pruned_by_prefilter = 19,800 — every document was
    // dropped by the cutoff, none by proof. Those two zeros mean opposite
    // things and only the pair says which.
    long long pruned_by_bound = 0;      // score < threshold: PROVEN outside top-K
    long long pruned_by_prefilter = 0;  // discarded WITHOUT a proof (fixed cutoff)
    // The denominator that makes the two above readable: without N and the
    // number of docs actually exact-scored, "19,800 discarded" is a number
    // without a fraction. An auditor reading `total_excluded_count = 0` needs
    // n_corpus=20,000 / scored=200 / proved=0 / cutoff=19,800 to see the shape.
    long long n_corpus = 0;             // documents in the index
    long long exact_scored = 0;         // docs the exact post-filter scored (k3)
    //
    // THESE THREE ARE NOT A PARTITION (measured 2026-10-30). `pruned_by_bound`
    // counts every doc whose bound fell below the threshold; `exact_scored`
    // counts every doc the post-filter scored. A pool survivor that then lost
    // is in BOTH. Measured at s1=s2=D=256 (where the residual is 0 and the
    // bound is exact): scored=60, bound=5990, prefilter=0 — the two overlap by
    // 50, and they sum to N+50. So:
    //   - pruned_by_bound <= N - K   is the invariant worth checking;
    //   - pruned_by_prefilter is a CLAMPED remainder (max(0, N - scored -
    //     bound)), so it can under-report when the sets overlap — read it as
    //     "at least this many were cut without a proof", never as a partition.
    // Stated here because summing them looks reasonable and is wrong.
    // RECALL SCOPE (2026-09-03, widened 2026-10-30): "exact_global" when the
    // returned top-K is provably the global top-K. Two independent routes reach
    // that:
    //   1. the exact post-filter scored all N (k3 == N) — the historical test;
    //   2. the bound proved every doc outside the top-K (pruned_by_bound >= N-K)
    //      — which implies the top-K we returned IS the global one, by proof.
    // Route 2 was already computed and thrown away: a config with s1 == d
    // returned recall 1.0000 and pruned_by_bound = N-K while being labelled
    // "pool_only". A compliance layer MUST NOT sign a "pool_only" commitment as
    // if it proved the global top-K — but it equally must not understate a
    // proof it holds. `recall_guarantee_basis` says which route was taken.
    std::string recall_guarantee = "pool_only";  // "pool_only" | "exact_global"
    // WHY A COMMITMENT WAS DEGRADED (2026-11-06). Empty = the count means what
    // it says. Non-empty = the proof was NOT computed and the record must not
    // be read as "the bound excluded nothing":
    //   "pool_exhausted" — search returned fewer than K results, so no global
    //                      K-th score exists; total_excluded_count is 0 by
    //                      construction, not by measurement.
    // Before this field, a k larger than the pool produced a signed record
    // with total_excluded_count = 0 and no marker distinguishing it from a
    // genuine zero (a bound too loose to prove anything).
    std::string degraded_reason = "";
    long long requested_k = 0;   // K the caller asked for (when degraded)
    long long returned = 0;      // results actually produced (when degraded)
    // "scored_all" | "proven_exclusions" | "none" — the OBSERVED basis of the
    // label above, so a reader never has to infer the reason from k3.
    std::string recall_guarantee_basis = "none";
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

    // ANEXA o corpus float32 SEM construir as projeções.
    //
    // Existe para o caminho load_index(): o índice em disco já traz P1_/P2_,
    // pr1_f_/pr2_f_, e1_/e2_, vn_/vn_eff_ prontos. O que falta para a busca
    // funcionar é só `base_` — o corpus uint8 que o postfilter lê em
    // exact_score(). `build_float32` faz isso E recalcula todas as projeções,
    // que é exatamente o trabalho que o load_index traz do arquivo.
    //
    // Medido (V=151936, d=2560): build_float32 = 7.77 s, load_index = 0.04 s.
    // Chamar build_float32 antes de load_index anula o ganho do artefato: o
    // EngineTokenizer rendia 1.24x em vez de ~174x.
    //
    // Retorna false (sem anexar nada) se o corpus for inválido.
    bool attach_float32(const float* raw_base, int n);

    // UB Width: supply the projection basis explicitly (e.g. computed by the
    // X-Factor, or by a robust numpy `eigh` over the corpus covariance).
    // P1 must be [stage1_dim × dim] orthonormal rows; if stage2_dim > 0, P2
    // must be [stage2_dim × dim]. Overrides cfg_.basis for the given stage.
    // This is the recommended path for d ~ 1536: a hand-rolled Jacobi in
    // float32 under-converges and collapses to the null subspace, while the
    // X-Factor's power iteration (or numpy's LAPACK eigh) is robust.
    void set_basis(const float* P1, const float* P2 = nullptr);

    // Search: bound pruning (top-k1/k2) + optional post-filter.
    // collect_audit (internal, 2026-09-06): when false, the search still
    // GATED on collect_audit (2026-10-30). With collect_audit=false the path
    // skips BOTH the per-document materialization (audit_ids/audit_ubs) AND the
    // O(N·d) global-threshold scan, so it returns the top-K fast — but its
    // `pruned_by_bound` / `audit_threshold` are then scoped to the POST-FILTER
    // POOL, not the corpus, and `recall_guarantee_basis` can never be
    // "proven_exclusions" (the proof would be about the wrong threshold).
    // Callers that need the corpus-wide proof (search_audited,
    // search_with_commitment) MUST pass true — it is not a default, so that the
    // choice is visible at the call site rather than inherited by accident.
    // k_override: per-call K (2026-11-06). 0 = use cfg_.k (historical).
    // Prior to this, K was always the build-time k and the `k` argument of
    // search_with_commitment could only shrink it — so a signed WORM record
    // could carry k_requested=N while the search executed k=cfg_.k.
    SearchResult search(const float* query, const std::vector<float>& query_norm,
                        bool collect_audit, int k_override = 0) const;
    // Public 2-arg search (collects the audit — backward compatible).
    SearchResult search(const float* query, const std::vector<float>& query_norm) const;
    SearchResult search(const float* query) const; // computes norm internally

    // Search with the audit hook (the witness path). The normal search()
    // ALREADY captures the per-document pruning decision unconditionally (the
    // hook is unconditional since 1.9.1).

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

    // Phase-3 GPU Stage-1 (2026-09-04): enable the upload-once GPU scan of the
    // Stage-1 bound (the O(N) hot loop) after the engine is built. Uploads
    // pr1_f_/e1_/vn_eff_ to the device ONCE; subsequent search() calls run the
    // Stage-1 scan on the GPU when this is active. opencl_lib selects the
    // loader ("" = standard / WINNEX_OPENCL_LIB). Falls back silently to the
    // CPU Stage-1 scan when the GPU is unavailable (returns false). The engine
    // must be built (quant="none" — the float32 projection path) and must NOT
    // be used concurrently with enable/disable. search() remains thread-safe:
    // the GPU scan is serialized by an internal mutex and each call uses its
    // own scores buffer.
    bool enable_gpu_stage1(const std::string& opencl_lib = "");


    // UB Width diagnostics: the projection basis (orthonormal rows) and the
    // per-vector Cauchy-Schwarz residuals e(v) = sqrt(||v||^2 − ||P v||^2).
    // A small mean residual = the projection captures the manifold (tight
    // bound); a large one = wide UB (the random-basis regime at high dim).
    const float* basis1() const { return P1_; }     // [stage1_dim × dim]
    const float* basis2() const { return P2_; }     // [stage2_dim × dim] or nullptr
    const float* residuals1() const { return e1_; } // [n]
    const float* residuals2() const { return e2_; } // [n] or nullptr

private:
    // Quantiza float32 -> uint8 no buffer owned. Extraído de build_float32
    // para que attach_float32() possa reaproveitar sem reprojetar nada.
    void quantize_float32_to_u8(const float* raw_base, int n);
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
    // NOTE: no per-query mutable scratch (2026-09-04). search() is const and
    // thread-safe for CONCURRENT calls on the same engine (search_batch /
    // OpenMP / GPU). All per-query scratch (the Stage-1 UB vector for the audit
    // hook) is allocated LOCALLY inside search() — see src/winnex_madhava.cpp.

    // Phase-3 GPU Stage-1 state (2026-09-04). `gpu_keepalive_` holds a
    // SpeedEngine GPU instance alive so the process-global OpenCL context and
    // the upload-once projection buffers (madhava_gpu_stage1_*) stay valid for
    // the engine's lifetime. `gpu_mu_` serializes the per-query GPU scan (the
    // OpenCL command queue is not safe for concurrent enqueue from multiple
    // threads). `gpu_stage1_ready_` gates the GPU path in search(). These are
    // mutable because search() is const; the mutex only guards the GPU enqueue,
    // NOT any engine data (the engine is immutable after build), so concurrent
    // search() calls remain correct (they serialize only on the short GPU op).
    mutable std::mutex gpu_mu_;
    mutable bool gpu_stage1_ready_ = false;
    void* gpu_keepalive_ = nullptr;   // owns a SpeedEngine (keeps OpenCL alive)
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
// Read the official BIGANN ground-truth file (header: nq:int32, dim:int32,
// then per query: dim ids int32 + dim dists float32).
// Returns gt[query_index][neighbor_slot] = dataset id.
std::vector<std::vector<int>> read_bigann_groundtruth(const std::string& path, int n_queries);

// ---------------------------------------------------------------------------
// Phase-3 GPU Stage-1 scan — upload-once helpers (2026-09-04)
// ---------------------------------------------------------------------------
// The bound engine's Stage-1 scan (UB over all N docs) is the O(N) hot loop.
// These helpers run it on the GPU with the projection buffers uploaded ONCE
// (init) and reused across queries (scan) — the upload-once pattern that makes
// the GPU win real (~25x measured). Implemented in speed_opencl.cpp when
// MADHAVA_HAS_OPENCL; CPU-only stubs in speed_cpu.cpp otherwise.
//
// init: uploads pr1 ([N x s1] float32, row-major) + e1 ([N] residuals) +
//   vn_eff ([N] effective norms) to the device. Returns true on success.
//   The buffers persist (process-global) until madhava_gpu_stage1_free().
// scan: computes the Stage-1 sortable score for one projected query into
//   scores_out ([N]): L2 = ||v_eff||^2 + ||q_eff||^2 − 2·UB; cosine = −UB,
//   with UB = <pr1[v],pq> + e1[v]·qr + qm + eps. pq is the ALREADY-PROJECTED
//   query ([s1]); qr/qm/qn2 are the query residual / int8 margin / ||q_eff||^2
//   computed exactly as search() does. Returns true on success.
bool madhava_gpu_stage1_init(const float* pr1, const float* e1, const float* vn_eff,
                             int N, int s1, const char* opencl_lib);
bool madhava_gpu_stage1_scan(const float* pq, float qr, float qm, float qn2,
                             int N, int s1, int is_l2, float* scores_out);
void madhava_gpu_stage1_free();

} // namespace winnex_madhava

#endif // WINNEX_MADHAVA_HPP
