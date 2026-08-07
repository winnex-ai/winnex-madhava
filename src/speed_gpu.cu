/**
 * speed_gpu.cu — native speed mode, CUDA backend (cuBLAS SGEMM for QKᵀ)
 * =================================================================
 * The QKᵀ matmul — the attention operation from "Attention is all you need"
 * — is a cuBLAS SGEMM:
 *
 *     scores[nq][N] = Q[nq][d] @ corpus[N][d]ᵀ
 *
 * The per-row topk is done on the host (exact, fast for k<=10). This keeps
 * the CUDA code minimal and correct: cuBLAS does the heavy matmul, the host
 * does the small topk. No torch, no triton, no torchvision.
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
#include <vector>

namespace winnex_madhava {

namespace {
cublasHandle_t g_cublas = nullptr;
int g_device = -1;
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

bool SpeedEngine::cuda_available() {
    if (g_device >= 0) return true;
    if (g_device == -2) return false;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) { g_device = -2; return false; }
    g_device = 0;
    return true;
}

// --- CUDA enable (called from the constructors in speed_cpu.cpp) ----------
void _enable_gpu(SpeedEngine& self, const std::vector<float>& corpus, int n, int dim) {
    if (!SpeedEngine::cuda_available()) return;
    SpeedEngine::init_gpu_impl();
    cudaMalloc(&self.corpus_gpu_, (size_t)n * dim * sizeof(float));
    cudaMemcpy(self.corpus_gpu_, corpus.data(), (size_t)n * dim * sizeof(float),
               cudaMemcpyHostToDevice);
    self.use_gpu_ = true;
}

// --- scores via cuBLAS SGEMM (the QKᵀ matmul) ------------------------------
void SpeedEngine::scores_gpu(const float* queries, int nq, float* scores_host) const {
    int N = n_, d = dim_;
    if (!g_cublas) init_gpu_impl();

    // Normalize cosine queries on the host.
    std::vector<float> qhost((size_t)nq * d);
    for (int qi = 0; qi < nq; qi++) {
        const float* q = queries + (size_t)qi * d;
        float qn = 0;
        for (int j = 0; j < d; j++) qn += q[j] * q[j];
        float inv = 1.0f;
        if (is_cosine_) inv = qn > 1e-12f ? 1.0f / std::sqrt(qn) : 0.0f;
        for (int j = 0; j < d; j++) qhost[(size_t)qi * d + j] = q[j] * inv;
    }

    float *qdev = nullptr, *sdev = nullptr;
    cudaMalloc(&qdev, (size_t)nq * d * sizeof(float));
    cudaMalloc(&sdev, (size_t)nq * N * sizeof(float));
    cudaMemcpy(qdev, qhost.data(), (size_t)nq * d * sizeof(float), cudaMemcpyHostToDevice);

    // C[nq][N] = A[nq][d] @ B[d][N], where B = corpusᵀ ([N][d] row-major).
    // cuBLAS is column-major: C(N, nq) = B^T * A^T, with B [N][d] → [d][N]ᵀ.
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, nq, d,
                &alpha,
                corpus_gpu_, N,   // B: corpus [N][d], col-major stride N
                qdev, d,          // A: q [nq][d], col-major stride d
                &beta,
                sdev, N);         // C: [N][nq] → stored as [nq][N] row-major
    cudaMemcpy(scores_host, sdev, (size_t)nq * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(qdev);
    cudaFree(sdev);
}

} // namespace winnex_madhava
