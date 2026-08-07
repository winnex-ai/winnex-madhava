/**
 * speed_engine.hpp — native speed mode: QKᵀ matmul + topk in C++
 * =================================================================
 * The "speed" mode uses exactly the attention operation from
 * "Attention is all you need":
 *
 *     Attention(Q, K, V) = softmax( (Q Kᵀ) / √dₖ ) V
 *
 * For vector search the "Q Kᵀ" is a batched matmul between queries and the
 * corpus, followed by a per-row argmax (topk). This is implemented NATIVELY
 * in C++ — no torch, no triton, no torchvision:
 *
 *   - GPU path:  cuBLAS SGEMM (the QKᵀ matmul) + a custom topk kernel,
 *                compiled when a CUDA toolkit is present.
 *   - CPU path:  OpenMP parallel-for + AVX2 dot products (the same kernels
 *                used by the bound engine), + std::nth_element for the topk.
 *
 * The backend is chosen at runtime. Without a CUDA toolkit the CPU path is
 * always available and correct; the GPU path is enabled only when the wheel
 * was built with CUDA AND a compatible device is present.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_MADHAVA_SPEED_ENGINE_HPP
#define WINNEX_MADHAVA_SPEED_ENGINE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace winnex_madhava {

// Forward decl of the metric enum (avoid pulling the full header).
enum class Metric;

struct SpeedResult {
    std::vector<int> indices;    // top-k dataset ids (best first)
    double latency_ms = 0.0;
    long long bound_pairs = 0;   // N vectors scored (the audit count)
    long long bound_violations = 0; // always 0 (exact scan)
};

class SpeedEngine {
public:
    // corpus: (n, dim) float32 (normalized for cosine) or uint8 (for L2).
    // The engine copies the corpus to its own storage.
    // n_anchors: K PiPrime anchors (SVD+Gram-Schmidt) for O(K) navigation.
    //   When n_anchors >= 2, the search is sublinear: the query is routed to
    //   the nprobe most-similar anchor cells, and QKᵀ runs only over the
    //   members of those cells. When n_anchors < 2, it falls back to the full
    //   exact scan (brute force).
    SpeedEngine(const float* corpus, int n, int dim, Metric metric,
                int n_anchors = 0, int nprobe = 4, bool require_gpu = false);
    SpeedEngine(const uint8_t* corpus, int n, int dim, Metric metric,
                int n_anchors = 0, int nprobe = 4, bool require_gpu = false);
    ~SpeedEngine();

    SpeedEngine(const SpeedEngine&) = delete;
    SpeedEngine& operator=(const SpeedEngine&) = delete;

    // True if this build has a usable CUDA backend AND a device was found.
    bool has_gpu() const { return use_gpu_; }

    // Human-readable backend in use: "gpu" or "cpu" (plus the reason the GPU
    // was not enabled, via gpu_reason()). Useful for logs and diagnostics.
    const char* backend_name() const { return use_gpu_ ? "gpu" : "cpu"; }
    // Why the GPU was not enabled (e.g. "build without CUDA", "no CUDA device
    // found"). Empty string when use_gpu_ is true.
    const std::string& gpu_reason() const { return gpu_reason_; }

    // Single query: scores = q @ corpus.T → topk.
    // query: (dim,) float32. Returns top-k indices (best first).
    SpeedResult search(const float* query, int k) const;

    // Batch: scores = Q @ corpus.T → topk per row (the throughput mode).
    // queries: (nq, dim) float32 contiguous. Returns nq*k indices.
    SpeedResult search_batch(const float* queries, int nq, int k) const;

    int num_vectors() const { return n_; }
    int dim() const { return dim_; }

private:
    // CPU implementation (always compiled).
    SpeedResult search_cpu(const float* query, int k) const;
    SpeedResult search_batch_cpu(const float* queries, int nq, int k) const;

    // Enable the GPU backend (CUDA hooks defined in speed_gpu.cu when a CUDA
    // toolkit is present; CPU stub in speed_cpu.cpp otherwise). Copies the
    // corpus to device memory and sets use_gpu_ on success. When require_gpu_
    // is true, throws std::runtime_error instead of silently falling back to
    // the CPU backend, and records the reason in gpu_reason_.
    void enable_gpu(const std::vector<float>& corpus, int n, int dim);

    // GPU hooks. Implemented by the compiled backend:
    //   - speed_gpu.cu      when a CUDA toolkit is present (MADHAVA_HAS_CUDA)
    //   - speed_opencl.cpp  when OpenCL is present (MADHAVA_HAS_OPENCL)
    //   - speed_cpu.cpp     CPU-only stubs (otherwise)
    static void init_gpu_impl();
    static void free_gpu_impl();
    static bool gpu_available();
    // Returns scores[nq][N] computed via cuBLAS SGEMM (the QKᵀ matmul).
    // Normalizes cosine queries; caller applies the L2 correction + topk.
    void scores_gpu(const float* queries, int nq, float* scores_host) const;

    // Optimized GPU path: QKᵀ matmul + device topk, returning only the nq·k
    // indices (no full-score D2H copy). Buffers reused across calls.
    SpeedResult scores_gpu_topk(const float* queries, int nq, int k,
                                std::vector<int>& out_indices) const;

    int n_;
    int dim_;
    bool is_cosine_;
    bool use_gpu_ = false;
    bool require_gpu_ = false;
    std::string gpu_reason_;   // why the GPU was not enabled (log/diagnostics)

    // O(K) anchor navigation: K PiPrime anchors + per-vector cell assignment.
    int n_anchors_ = 0;
    int nprobe_ = 4;
    bool use_anchors_ = false;
    std::vector<float> anchors_;        // [K * dim] orthonormal anchors
    std::vector<int> cell_of_;          // [N] anchor index per vector
    std::vector<std::vector<int>> cells_; // cells_[k] = vector ids in anchor cell k
    std::vector<float> cell_radius_;    // [K] max ||v - a_k|| in cell k

    // CPU storage: normalized float32 corpus + per-vector norms (for L2).
    std::vector<float> corpus_f32_;
    std::vector<float> norms_;   // ||v|| (cosine normalized to 1.0; else raw)

    // GPU state (only allocated when has_gpu()).
    float* corpus_gpu_ = nullptr;
    float* norms_gpu_ = nullptr;

    // Build the PiPrime anchors (SVD + Gram-Schmidt) + Voronoi assignment.
    void build_anchors();
};

} // namespace winnex_madhava

#endif // WINNEX_MADHAVA_SPEED_ENGINE_HPP
