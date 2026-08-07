/**
 * speed_gpu.cu — native speed mode, CUDA backend (cuBLAS QKᵀ + device topk)
 * =================================================================
 * The QKᵀ matmul — the attention operation from "Attention is all you need"
 * — is a cuBLAS SGEMM: scores[nq][N] = Q[nq][d] @ corpus[N][d]ᵀ.
 *
 * Optimizations (vs the naive host-topk version):
 *   1. The per-row topk runs ON THE DEVICE (a CUDA kernel) — only nq·k
 *      indices are transferred back, not the full nq×N score matrix
 *      (which can be gigabytes for large N).
 *   2. Device buffers are pre-allocated once and reused (no cudaMalloc/
 *      cudaFree per query) — the SGEMM workspace is reused too.
 *   3. Cosine query normalization runs on the device (a small kernel).
 *   4. The L2 correction (2·ip − ||v||²) runs on the device before topk.
 *
 * No torch, no triton, no torchvision — just cuBLAS + CUDA kernels.
 *
 * Compiled ONLY when a CUDA toolkit is present (CMake find_package(CUDA)).
 * Without CUDA, the CPU backend (speed_cpu.cpp) is used.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#include "winnex_madhava/speed_engine.hpp"
#include "winnex_madhava/winnex_madhava.hpp"  // for Metric

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace winnex_madhava {

namespace {
cublasHandle_t g_cublas = nullptr;
int g_device = -1;
}

// --- CUDA kernels ---------------------------------------------------------

// Normalize a batch of queries (cosine): q_i = q_i / ||q_i||.
// One thread per query, d threads per query.
__global__ void normalize_queries_kernel(const float* q, float* qout, int nq, int d) {
    int qi = blockIdx.x;
    if (qi >= nq) return;
    const float* src = q + (size_t)qi * d;
    float* dst = qout + (size_t)qi * d;
    float norm = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) norm += src[j] * src[j];
    // block reduction
    __shared__ float red[256];
    int tid = threadIdx.x;
    red[tid] = norm;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float inv = (red[0] > 1e-12f) ? rsqrtf(red[0]) : 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) dst[j] = src[j] * inv;
}

// Apply the L2 correction: score = 2·ip − ||v||², in place on the score row.
// One thread per element.
__global__ void l2_correct_kernel(float* scores, const float* norms, int nq, int N) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nq * N;
    if (idx >= total) return;
    int qi = (int)(idx / N);
    int vi = (int)(idx % N);
    scores[idx] = 2.0f * scores[idx] - norms[vi];
}

// Per-row topk: one thread block per query. Each block keeps a small sorted
// top-k (k <= 64) and scans the row. Returns the top-k indices.
__global__ void topk_kernel(const float* scores, int* out_idx, int nq, int N, int k) {
    int qi = blockIdx.x;
    if (qi >= nq) return;
    const float* row = scores + (size_t)qi * N;
    int* out = out_idx + (size_t)qi * k;

    // Shared top-k storage: k values + k indices, per block (small k).
    extern __shared__ float sh_v[];
    int* sh_i = (int*)(sh_v + k);
    for (int j = 0; j < k; j++) { sh_v[j] = -1e30f; sh_i[j] = -1; }

    // Scan the row, maintaining the top-k in shared memory (serial per block,
    // but each element is a cheap compare+shift for k<=10).
    // For large N this is memory-bound on the row read, which is what we want.
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float s = row[i];
        // insert into sorted top-k (small k, so a simple loop is fine)
        for (int j = k - 1; j >= 0; j--) {
            if (j == 0 || sh_v[j - 1] >= s) {
                if (sh_v[j] < s) { sh_v[j] = s; sh_i[j] = i; }
                break;
            }
            sh_v[j] = sh_v[j - 1];
            sh_i[j] = sh_i[j - 1];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 0; j < k; j++) out[j] = sh_i[j];
    }
}

// --- Static CUDA lifecycle ------------------------------------------------
void SpeedEngine::init_gpu_impl() {
    if (g_device >= 0) return;
    if (cudaGetDevice(&g_device) != cudaSuccess) { g_device = -2; return; }
    if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) { g_device = -2; }
}

void SpeedEngine::free_gpu_impl() {
    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = nullptr; }
}

bool SpeedEngine::gpu_available() {
    if (g_device >= 0) return true;
    if (g_device == -2) return false;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) { g_device = -2; return false; }
    g_device = 0;
    return true;
}

// --- CUDA enable (called from the constructors in speed_cpu.cpp) ----------
// Real GPU path: allocates the corpus (and L2 norms) on the device, sets
// use_gpu_. When require_gpu_ is set and the GPU cannot be enabled, throws
// instead of silently degrading to the CPU backend.
void SpeedEngine::enable_gpu(const std::vector<float>& corpus, int n, int dim) {
    if (!SpeedEngine::gpu_available()) {
        gpu_reason_ = "no CUDA device found (cudaGetDeviceCount == 0 or error)";
        if (require_gpu_) {
            throw std::runtime_error(
                "SpeedEngine: GPU required but no CUDA device is available. "
                "Reason: " + gpu_reason_);
        }
        std::fprintf(stderr,
                     "[Winnex Madhava] SpeedEngine: GPU unavailable — falling "
                     "back to CPU backend. Reason: %s\n",
                     gpu_reason_.c_str());
        return;
    }
    SpeedEngine::init_gpu_impl();
    cudaMalloc(&corpus_gpu_, (size_t)n * dim * sizeof(float));
    cudaMemcpy(corpus_gpu_, corpus.data(), (size_t)n * dim * sizeof(float),
               cudaMemcpyHostToDevice);
    // Norms for the L2 correction: ||v||² per vector (precomputed on device).
    if (!is_cosine_) {
        cudaMalloc(&norms_gpu_, (size_t)n * sizeof(float));
        cudaMemcpy(norms_gpu_, norms_.data(), (size_t)n * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    use_gpu_ = true;
    std::fprintf(stderr,
                 "[Winnex Madhava] SpeedEngine: GPU backend enabled "
                 "(cuBLAS QK^T + device topk).\n");
}

// --- scores via cuBLAS SGEMM (the QKᵀ matmul), topk ON DEVICE -------------
// Computes scores[nq][N] = Q[nq][d] @ corpus[N][d]ᵀ, applies L2 correction
// on device, and runs the per-row topk on device. Only the nq·k indices are
// copied back. Buffers are reused across calls.
SpeedResult SpeedEngine::scores_gpu_topk(const float* queries, int nq, int k,
                                         std::vector<int>& out_indices) const {
    SpeedResult res;
    int N = n_, d = dim_;
    if (!g_cublas) init_gpu_impl();
    static float* qdev = nullptr;
    static float* sdev = nullptr;
    static int qdev_cap = 0, sdev_cap = 0;
    static int* idxdev = nullptr;

    // Reuse device buffers (grow if needed).
    if (qdev == nullptr || (size_t)nq * d > (size_t)qdev_cap) {
        if (qdev) cudaFree(qdev);
        qdev_cap = (int)((size_t)nq * d * 1.2f);
        cudaMalloc(&qdev, (size_t)qdev_cap * sizeof(float));
    }
    if (sdev == nullptr || (size_t)nq * N > (size_t)sdev_cap) {
        if (sdev) cudaFree(sdev);
        sdev_cap = (int)((size_t)nq * N * 1.2f);
        cudaMalloc(&sdev, (size_t)sdev_cap * sizeof(float));
    }
    if (idxdev == nullptr) cudaMalloc(&idxdev, (size_t)64 * 64 * sizeof(int));

    // Copy queries to device (raw; normalization happens on device if cosine).
    cudaMemcpy(qdev, queries, (size_t)nq * d * sizeof(float), cudaMemcpyHostToDevice);

    // Normalize on device for cosine.
    if (is_cosine_) {
        normalize_queries_kernel<<<nq, 256>>>(qdev, qdev, nq, d);
    }

    // SGEMM: C[nq][N] = Q[nq][d] @ corpus[N][d]ᵀ (cuBLAS col-major).
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, nq, d,
                &alpha,
                corpus_gpu_, N,   // B: corpus [N][d]
                qdev, d,          // A: q [nq][d]
                &beta,
                sdev, N);         // C: [N][nq] → [nq][N] row-major

    // L2 correction on device (only when metric is L2).
    if (!is_cosine_) {
        int threads = 256;
        long long total = (long long)nq * N;
        int blocks = (int)((total + threads - 1) / threads);
        l2_correct_kernel<<<blocks, threads>>>(sdev, norms_gpu_, nq, N);
    }

    // Per-row topk on device (only nq·k indices are written).
    k = std::min(k, N);
    int shared = 2 * k * sizeof(float);  // values + indices
    topk_kernel<<<nq, 256, shared>>>(sdev, idxdev, nq, N, k);

    // Copy back only the nq·k indices.
    out_indices.resize((size_t)nq * k);
    cudaMemcpy(out_indices.data(), idxdev, (size_t)nq * k * sizeof(int),
               cudaMemcpyDeviceToHost);

    res.bound_pairs = (long long)nq * N;
    res.bound_violations = 0;
    return res;
}

// Backward-compatible hook: the CPU dispatch calls scores_gpu to fetch scores.
// For the optimized path we override it to call scores_gpu_topk when possible.
void SpeedEngine::scores_gpu(const float* queries, int nq, float* scores_host) const {
    // Fallback: naive full-score path (kept for compatibility with the
    // CPU dispatch when the topk-on-device path is not wired up).
    int N = n_, d = dim_;
    if (!g_cublas) init_gpu_impl();

    std::vector<float> qhost((size_t)nq * d);
    for (int qi = 0; qi < nq; qi++) {
        const float* q = queries + (size_t)qi * d;
        float qn = 0;
        for (int j = 0; j < d; j++) qn += q[j] * q[j];
        float inv = is_cosine_ && qn > 1e-12f ? 1.0f / std::sqrt(qn) : (is_cosine_ ? 0.f : 1.f);
        for (int j = 0; j < d; j++) qhost[(size_t)qi * d + j] = q[j] * inv;
    }
    float *qdev = nullptr, *sdev = nullptr;
    cudaMalloc(&qdev, (size_t)nq * d * sizeof(float));
    cudaMalloc(&sdev, (size_t)nq * N * sizeof(float));
    cudaMemcpy(qdev, qhost.data(), (size_t)nq * d * sizeof(float), cudaMemcpyHostToDevice);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, nq, d, &alpha,
                corpus_gpu_, N, qdev, d, &beta, sdev, N);
    cudaMemcpy(scores_host, sdev, (size_t)nq * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(qdev);
    cudaFree(sdev);
}

} // namespace winnex_madhava
