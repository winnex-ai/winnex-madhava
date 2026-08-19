/**
 * speed_opencl.cpp — native speed mode, OpenCL backend (generic GPU)
 * =================================================================
 * The QKᵀ matmul (scores = Q @ corpus.T) and the per-row topk, implemented
 * on a GENERIC GPU via OpenCL — the portable, vendor-neutral compute API
 * (the "OpenGL of compute"). Unlike the CUDA backend, OpenCL kernels are
 * compiled at RUNTIME by the driver (JIT), so no offline compiler (nvcc) and
 * no vendor-specific SDK are needed. It runs on NVIDIA, AMD, and Intel GPUs.
 *
 * Optimizations (same contract as speed_gpu.cu):
 *   1. The per-row topk runs ON THE DEVICE — only nq·k indices come back,
 *      not the full nq×N score matrix (which can be gigabytes for large N).
 *   2. Device buffers are pre-allocated once and reused across calls.
 *   3. Cosine query normalization runs on the device (a small kernel).
 *   4. The L2 correction (2·ip − ||v||²) runs on the device before topk.
 *
 * Kernels are embedded as source strings and compiled by clBuildProgram at
 * engine construction (JIT), exactly like a game engine compiles GLSL shaders
 * at runtime.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#include "winnex_madhava/speed_engine.hpp"
#include "winnex_madhava/winnex_madhava.hpp"  // for Metric

// Vendored OpenCL headers (types + constants only; no link-time loader).
#include <CL/cl.h>

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace winnex_madhava {

namespace {

// ---- Runtime OpenCL loading (dlopen, no link-time dependency) ----------
// The wheel must run on any system: with a GPU (NVIDIA/AMD/Intel OpenCL ICD
// loader present) it uses OpenCL; otherwise it falls back to CPU. So we do
// NOT link against libOpenCL at build time — we dlopen() it at first use and
// resolve the functions we need (like a game engine loads GL at runtime).
// This keeps auditwheel/manylinux happy (no missing shared-lib deps) and the
// same wheel works with or without an OpenCL loader.
struct OCLApi;
static OCLApi* g_ocl = nullptr;  // nullptr until dlopen succeeds
static void* g_ocl_handle = nullptr;  // dlopen handle of the loaded loader
static std::string g_ocl_lib;    // the loader name actually loaded (for the
                                 // per-engine resolution: a DIFFERENT loader
                                 // requested by a later engine must not be
                                 // satisfied by a previously-loaded one)

#define OCL_FUNC(ret, name, args) ret (*name) args;
struct OCLApi {
    OCL_FUNC(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*))
    OCL_FUNC(cl_int, clGetDeviceIDs, (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*))
    OCL_FUNC(cl_context, clCreateContext, (const cl_context_properties*, cl_uint, const cl_device_id*, void (CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*))
    OCL_FUNC(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_command_queue_properties, cl_int*))
    OCL_FUNC(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))
    OCL_FUNC(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void (CL_CALLBACK*)(cl_program, void*), void*))
    OCL_FUNC(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*))
    OCL_FUNC(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*))
    OCL_FUNC(cl_mem, clCreateBuffer, (cl_context, cl_mem_flags, size_t, void*, cl_int*))
    OCL_FUNC(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*))
    OCL_FUNC(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, cl_event*, cl_event*))
    OCL_FUNC(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*))
    OCL_FUNC(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))
    OCL_FUNC(cl_int, clReleaseMemObject, (cl_mem))
    OCL_FUNC(cl_int, clReleaseKernel, (cl_kernel))
    OCL_FUNC(cl_int, clReleaseProgram, (cl_program))
    OCL_FUNC(cl_int, clReleaseCommandQueue, (cl_command_queue))
    OCL_FUNC(cl_int, clReleaseContext, (cl_context))
};
#undef OCL_FUNC

// Resolve all OpenCL entry points from an explicit loader. Returns true on success.
//
// TRANSPARENT LOADER (no hardcoded vendor .so):
//   The loader name is passed by the caller (SpeedEngine::opencl_lib_ or the
//   WINNEX_OPENCL_LIB env var). When empty, the standard platform ICD loader
//   is used. There is NO silent vendor cascade: the caller decides which .so
//   to load, and the failure is reported via gpu_reason_ / stderr. This keeps
//   the behavior explainable and configurable without a hardcoded driver list.
bool load_opencl(const char* explicit_lib = nullptr) {
    // Resolve the loader name for THIS request (per-engine):
    //   explicit param > $WINNEX_OPENCL_LIB > default loader.
    std::string want;
    if (explicit_lib && *explicit_lib) want = explicit_lib;
    else if (const char* env = std::getenv("WINNEX_OPENCL_LIB")) { if (*env) want = env; }
    else want = "libOpenCL.so.1";

    // Reuse the already-loaded loader ONLY when it is the SAME one requested.
    // A different loader (or a fresh default) must load its own .so — the
    // config of THIS engine is authoritative, never masked by an earlier one.
    if (g_ocl && g_ocl_lib == want) return true;
    if (g_ocl && explicit_lib && *explicit_lib && g_ocl_lib != want) {
        // The previously loaded loader is not the requested one — release it.
        dlclose(g_ocl_handle);
        delete g_ocl;
        g_ocl = nullptr;
        g_ocl_lib.clear();
    }

    void* h = dlopen(want.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr,
                     "[Winnex Madhava] OpenCL loader '%s' failed to load: %s\n",
                     want.c_str(), dlerror() ? dlerror() : "dlopen failed");
        return false;
    }
    std::fprintf(stderr,
                 "[Winnex Madhava] OpenCL loader resolved: %s\n", want.c_str());
    auto api = new OCLApi;
    bool ok = true;
#define LOAD(name) \
    *(void**)(&api->name) = dlsym(h, #name); \
    if (!api->name) ok = false
    LOAD(clGetPlatformIDs);
    LOAD(clGetDeviceIDs);
    LOAD(clCreateContext);
    LOAD(clCreateCommandQueue);
    LOAD(clCreateProgramWithSource);
    LOAD(clBuildProgram);
    LOAD(clGetProgramBuildInfo);
    LOAD(clCreateKernel);
    LOAD(clCreateBuffer);
    LOAD(clSetKernelArg);
    LOAD(clEnqueueNDRangeKernel);
    LOAD(clEnqueueWriteBuffer);
    LOAD(clEnqueueReadBuffer);
    LOAD(clReleaseMemObject);
    LOAD(clReleaseKernel);
    LOAD(clReleaseProgram);
    LOAD(clReleaseCommandQueue);
    LOAD(clReleaseContext);
#undef LOAD
    if (!ok) {
        dlclose(h);
        delete api;
        return false;
    }
    g_ocl = api;
    g_ocl_lib = want;
    g_ocl_handle = h;
    return true;
}

// Shortcut macros so the rest of this file reads like normal OpenCL calls.
#define clGetPlatformIDs            g_ocl->clGetPlatformIDs
#define clGetDeviceIDs              g_ocl->clGetDeviceIDs
#define clCreateContext             g_ocl->clCreateContext
#define clCreateCommandQueue        g_ocl->clCreateCommandQueue
#define clCreateProgramWithSource   g_ocl->clCreateProgramWithSource
#define clBuildProgram              g_ocl->clBuildProgram
#define clGetProgramBuildInfo       g_ocl->clGetProgramBuildInfo
#define clCreateKernel              g_ocl->clCreateKernel
#define clCreateBuffer              g_ocl->clCreateBuffer
#define clSetKernelArg              g_ocl->clSetKernelArg
#define clEnqueueNDRangeKernel      g_ocl->clEnqueueNDRangeKernel
#define clEnqueueWriteBuffer        g_ocl->clEnqueueWriteBuffer
#define clEnqueueReadBuffer         g_ocl->clEnqueueReadBuffer
#define clReleaseMemObject          g_ocl->clReleaseMemObject
#define clReleaseKernel             g_ocl->clReleaseKernel
#define clReleaseProgram            g_ocl->clReleaseProgram
#define clReleaseCommandQueue       g_ocl->clReleaseCommandQueue
#define clReleaseContext            g_ocl->clReleaseContext

// Runtime-compiled OpenCL kernels (embedded source, JIT like GLSL).
const char* g_kernels = R"(
// Normalize a batch of queries (cosine): q_i = q_i / ||q_i||.
// One work-group per query, d work-items per query.
__kernel void normalize_queries(__global const float* q, __global float* qout,
                                int nq, int d) {
    int qi = get_group_id(0);
    if (qi >= nq) return;
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    float norm = 0.0f;
    for (int j = lid; j < d; j += lsz) norm += q[qi*d + j] * q[qi*d + j];
    red[lid] = norm;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = (red[0] > 1e-12f) ? rsqrt(red[0]) : 0.0f;
    for (int j = lid; j < d; j += lsz) qout[qi*d + j] = q[qi*d + j] * inv;
}

// QK^T: scores[q][v] = sum_d Q[q][d] * C[v][d]  (the attention matmul).
// OTIMIZADO: 1 work-group por query. A query (d floats) é carregada em local
// memory UMA vez e reutilizada por todos os work-items do grupo — cada item
// varre um sub-range de corpus com a query em local (evita reler a query da
// memória global para cada v). Grid: nq work-groups × lsz work-items.
// (Cada query é pequena, d=128 floats = 512B, cabe em local mem.)
__kernel void qkt(__global const float* q, __global const float* corpus,
                  __global float* scores, int nq, int N, int d) {
    int qi = get_group_id(0);
    if (qi >= nq) return;
    int lid = get_local_id(0), lsz = get_local_size(0);

    // Carrega a query em local memory (uma vez por grupo).
    __local float lq[512];
    for (int j = lid; j < d; j += lsz) lq[j] = q[qi*d + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Cada work-item varre um sub-range do corpus, com a query em local.
    for (int vi = lid; vi < N; vi += lsz) {
        const __global float* v = corpus + (size_t)vi * d;
        float s = 0.0f;
        for (int j = 0; j < d; j++) s += lq[j] * v[j];
        scores[qi*N + vi] = s;
    }
}

// L2 correction: score = 2·ip − ||v||², in place.
__kernel void l2_correct(__global float* scores, __global const float* norms,
                         int nq, int N) {
    long idx = (long)get_global_id(0);
    long total = (long)nq * N;
    if (idx >= total) return;
    int qi = (int)(idx / N);
    int vi = (int)(idx % N);
    scores[idx] = 2.0f * scores[idx] - norms[vi];
}

// Per-row topk PARALELO (divide-and-conquer): o scan de N é o gargalo.
// topk_local: M work-groups por query, cada um varre um CHUNK contíguo de N
//   e mantém um top-k local (insertion sort em local mem). O scan de N vira
//   N/M por work-group — M× mais paralelismo.
// topk_merge: funde os top-k locais de M work-groups → top-k global exato.
//
// Grid topk_local: nq*M work-groups × 256 itens.
// Grid topk_merge: nq work-groups × 256 itens.
__kernel void topk_local(__global const float* scores, __global int* local_idx,
                         int nq, int N, int k, int M) {
    int gid = get_group_id(0);
    int qi = gid / M;          // query
    int chunk = gid % M;       // chunk 0..M-1
    if (qi >= nq) return;
    int lid = get_local_id(0), lsz = get_local_size(0);

    // Chunk contíguo [start, end)
    int per = (N + M - 1) / M;
    int start = chunk * per;
    int end = min(start + per, N);

    __local float lv[256][16];
    __local int li[256][16];
    for (int j = 0; j < k; j++) { lv[lid][j] = -1e30f; li[lid][j] = -1; }
    for (int vi = start + lid; vi < end; vi += lsz) {
        float s = scores[(size_t)qi*N + vi];
        for (int j = k-1; j >= 0; j--) {
            if (j == 0 || lv[lid][j-1] >= s) {
                if (lv[lid][j] < s) { lv[lid][j] = s; li[lid][j] = vi; }
                break;
            }
            lv[lid][j] = lv[lid][j-1]; li[lid][j] = li[lid][j-1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        // merge dos top-k de cada work-item deste grupo → top-k do chunk
        float tv[16]; int ti[16];
        for (int j = 0; j < k; j++) { tv[j] = -1e30f; ti[j] = -1; }
        for (int w = 0; w < lsz; w++) {
            for (int j = 0; j < k; j++) {
                float s = lv[w][j]; if (s < -1e29f) continue;
                for (int p = k-1; p >= 0; p--) {
                    if (p == 0 || tv[p-1] >= s) {
                        if (tv[p] < s) { tv[p] = s; ti[p] = li[w][j]; }
                        break;
                    }
                    tv[p] = tv[p-1]; ti[p] = ti[p-1];
                }
            }
        }
        for (int j = 0; j < k; j++)
            local_idx[((size_t)qi*M + chunk)*k + j] = ti[j];
    }
}

// Funde os top-k locais de M chunks por query → top-k global exato.
__kernel void topk_merge(__global const float* scores, __global const int* local_idx,
                         __global int* idx, int nq, int N, int k, int M) {
    int qi = get_group_id(0);
    if (qi >= nq) return;
    int lid = get_local_id(0), lsz = get_local_size(0);

    __local float tv[64];
    __local int ti[64];
    if (lid == 0) {
        for (int j = 0; j < k; j++) { tv[j] = -1e30f; ti[j] = -1; }
        // merge das M listas de k
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < k; j++) {
                int cand = local_idx[((size_t)qi*M + m)*k + j];
                if (cand < 0) continue;
                float s = scores[(size_t)qi*N + cand];
                for (int p = k-1; p >= 0; p--) {
                    if (p == 0 || tv[p-1] >= s) {
                        if (tv[p] < s) { tv[p] = s; ti[p] = cand; }
                        break;
                    }
                    tv[p] = tv[p-1]; ti[p] = ti[p-1];
                }
            }
        }
        for (int j = 0; j < k; j++) idx[qi*k + j] = ti[j];
    }
}

// ===========================================================================
// FUSED QK^T + topk (single pass, memory-bound otimizado) — v1.7.2
// ===========================================================================
// A correcao do gargalo do single-query: o kernel `qkt` usava 1 work-group por
// query (1 SM ativo de 32 → ~3% da GPU) e materializava scores[N] na memoria
// global (2x trafego). Este kernel:
//
//   1. PARALELIZA por chunk: M work-groups por query, cada um varre um chunk
//      contiguo de N. Grid = nq*M work-groups → todos os SMs ativos mesmo com
//      nq=1. (Mesmo padrao do topk_local — o scan de N vira N/M por grupo.)
//
//   2. ACESSO COALESCIDO: dentro do chunk, work-items adjacentes processam
//      vetores CONSECUTIVOS (vi = start + lid, lid+lsz, ...). Como cada vetor
//      e lido via corpus[vi*d + j], os work-items de uma warp leem enderecos
//      adjacentes por j → cada carregamento de 32B da cache lida e usada por
//      todos os itens da warp (coalescido).
//
//   3. FUSION: mantem o top-k local em local mem (k<=16) e escreve APENAS os
//      k indices por chunk em `local_idx`. Nenhum scores[N] na memoria global.
//      O topk_merge depois funde os M top-k locais → top-k global exato.
//
//   L2 correction e aplicada aqui (2*ip - ||v||^2) usando o buffer norms.
//   Grid: nq*M work-groups x lsz itens. M>=1, k<=16.
//
//   Saida em local_idx (intercalada): [score0, idx0, score1, idx1, ...] por
//   (query, chunk). O topk_merge usa os scores reais dos candidatos.
__kernel void qkt_fused_topk(__global const float* q, __global const float* corpus,
                             __global const float* norms, __global float* local_idx,
                             int nq, int N, int d, int k, int M, int is_l2) {
    int gid = get_group_id(0);
    int qi = gid / M;           // query
    int chunk = gid % M;        // chunk 0..M-1
    if (qi >= nq) return;
    int lid = get_local_id(0), lsz = get_local_size(0);

    // Carrega a query em local memory (uma vez por grupo).
    __local float lq[512];
    for (int j = lid; j < d; j += lsz) lq[j] = q[qi*d + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Chunk contiguo [start, end) — varredura coalescida.
    int per = (N + M - 1) / M;
    int start = chunk * per;
    int end = min(start + per, N);

    // Top-k local deste work-item (valores + indices).
    float tv[16];
    int ti[16];
    for (int j = 0; j < k; j++) { tv[j] = -1e30f; ti[j] = -1; }

    // Scan coalescido: itens adjacentes leem vetores adjacentes.
    // (vi = start+lid, start+lid+lsz, ...) → warp acessa corpus contiguo.
    for (int vi = start + lid; vi < end; vi += lsz) {
        const __global float* v = corpus + (size_t)vi * d;
        float ip = 0.0f;
        for (int j = 0; j < d; j++) ip += lq[j] * v[j];
        float s = is_l2 ? (2.0f * ip - norms[vi]) : ip;
        // insertion sort no top-k local (k<=16)
        for (int p = k - 1; p >= 0; p--) {
            if (p == 0 || tv[p-1] >= s) {
                if (tv[p] < s) { tv[p] = s; ti[p] = vi; }
                break;
            }
            tv[p] = tv[p-1]; ti[p] = ti[p-1];
        }
    }

    // Merge dos top-k dos work-items deste work-group → top-k do chunk.
    __local float lv[256][16];
    __local int li[256][16];
    for (int j = 0; j < k; j++) { lv[lid][j] = tv[j]; li[lid][j] = ti[j]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        float out_v[16]; int out_i[16];
        for (int j = 0; j < k; j++) { out_v[j] = -1e30f; out_i[j] = -1; }
        for (int w = 0; w < lsz; w++) {
            for (int j = 0; j < k; j++) {
                float s = lv[w][j]; if (s < -1e29f) continue;
                for (int p = k - 1; p >= 0; p--) {
                    if (p == 0 || out_v[p-1] >= s) {
                        if (out_v[p] < s) { out_v[p] = s; out_i[p] = li[w][j]; }
                        break;
                    }
                    out_v[p] = out_v[p-1]; out_i[p] = out_i[p-1];
                }
            }
        }
        // Intercala scores + indices: local_idx[(q,m,k')*2 + 0]=score, +1=idx
        for (int j = 0; j < k; j++) {
            size_t off = ((size_t)qi*M + chunk) * k * 2 + j * 2;
            local_idx[off] = out_v[j];
            local_idx[off + 1] = (float)out_i[j];
        }
    }
}

// Funde os top-k locais com SCORES (escritos pelo qkt_fused_topk).
// local_idx e float intercalado [score0, idx0, ...]; nao relê scores[N].
__kernel void topk_merge_scores(__global const float* local_idx, __global int* idx,
                                int nq, int k, int M) {
    int qi = get_group_id(0);
    if (qi >= nq) return;
    int lid = get_local_id(0), lsz = get_local_size(0);

    __local float tv[16];
    __local int ti[16];
    if (lid == 0) {
        for (int j = 0; j < k; j++) { tv[j] = -1e30f; ti[j] = -1; }
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < k; j++) {
                size_t off = ((size_t)qi*M + m) * k * 2 + j * 2;
                float s = local_idx[off];
                int cand = (int)local_idx[off + 1];
                if (cand < 0) continue;
                for (int p = k - 1; p >= 0; p--) {
                    if (p == 0 || tv[p-1] >= s) {
                        if (tv[p] < s) { tv[p] = s; ti[p] = cand; }
                        break;
                    }
                    tv[p] = tv[p-1]; ti[p] = ti[p-1];
                }
            }
        }
        for (int j = 0; j < k; j++) idx[qi*k + j] = ti[j];
    }
}
)";

// Static OpenCL state (shared across all engine instances in this process).
cl_context g_ctx = nullptr;
cl_command_queue g_queue = nullptr;
cl_program g_prog = nullptr;
cl_kernel g_kqkt = nullptr;
cl_kernel g_kqkt_fused = nullptr;
cl_kernel g_ktopk = nullptr;
cl_kernel g_ktopk_local = nullptr;
cl_kernel g_ktopk_merge = nullptr;
cl_kernel g_ktopk_merge_scores = nullptr;
cl_kernel g_kl2 = nullptr;
cl_kernel g_knorm = nullptr;
cl_mem g_corpus = nullptr;   // device copy of the corpus
cl_mem g_norms = nullptr;    // device copy of ||v||² (L2 only)
int g_state = -1;            // -1 unknown, 0 unavailable, 1 ready

// Reusable per-call buffers (grown as needed).
cl_mem g_qbuf = nullptr;   size_t g_qcap = 0;
cl_mem g_sbuf = nullptr;   size_t g_scap = 0;
cl_mem g_idxbuf = nullptr; size_t g_idxcap = 0;
cl_mem g_local_idx = nullptr; size_t g_localcap = 0;  // top-k locais (topk_local)

bool cl_err(cl_int e, const char* what) {
    if (e != CL_SUCCESS) {
        std::fprintf(stderr, "[Winnex Madhava] OpenCL error in %s: code %d\n",
                     what, (int)e);
        return false;
    }
    return true;
}

} // namespace

// --- Static GPU lifecycle (OpenCL implementation) -------------------------
bool SpeedEngine::gpu_available() {
    if (g_state >= 0) return g_state == 1;
    if (!load_opencl()) { g_state = 0; return false; }
    cl_int err;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) {
        g_state = 0; return false;
    }
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    cl_uint ngpu = 0;
    for (auto p : plats) {
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &ngpu);
        if (ngpu > 0) { g_state = 1; return true; }
    }
    g_state = 0;
    return false;
}

void SpeedEngine::init_gpu_impl() {
    if (g_ctx) return;
    if (!load_opencl()) { g_state = 0; return; }
    cl_int err;
    cl_uint nplat = 0;
    clGetPlatformIDs(0, nullptr, &nplat);
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    cl_device_id dev = nullptr;
    for (auto p : plats) {
        cl_uint ngpu = 0;
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &ngpu);
        if (ngpu > 0) {
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
            break;
        }
    }
    if (!dev) { g_state = 0; return; }
    g_ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (!cl_err(err, "clCreateContext")) { g_state = 0; return; }
    g_queue = clCreateCommandQueue(g_ctx, dev, 0, &err);
    if (!cl_err(err, "clCreateCommandQueue")) { g_state = 0; return; }

    // JIT-compile the kernels (like a game engine compiling GLSL at startup).
    g_prog = clCreateProgramWithSource(g_ctx, 1, &g_kernels, nullptr, &err);
    if (!cl_err(err, "clCreateProgramWithSource")) { g_state = 0; return; }
    err = clBuildProgram(g_prog, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(g_prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::fprintf(stderr, "[Winnex Madhava] OpenCL kernel build failed:\n%s\n", log);
        g_state = 0; return;
    }
    g_kqkt  = clCreateKernel(g_prog, "qkt", &err);
    g_kqkt_fused = clCreateKernel(g_prog, "qkt_fused_topk", &err);
    g_ktopk_local = clCreateKernel(g_prog, "topk_local", &err);
    g_ktopk_merge = clCreateKernel(g_prog, "topk_merge", &err);
    g_ktopk_merge_scores = clCreateKernel(g_prog, "topk_merge_scores", &err);
    g_ktopk = g_ktopk_local;  // alias para compatibilidade
    g_kl2   = clCreateKernel(g_prog, "l2_correct", &err);
    g_knorm = clCreateKernel(g_prog, "normalize_queries", &err);
    if (!g_kqkt || !g_kqkt_fused || !g_ktopk_local || !g_ktopk_merge || !g_ktopk_merge_scores || !g_kl2 || !g_knorm) { g_state = 0; return; }
    g_state = 1;
}

void SpeedEngine::free_gpu_impl() {
    if (g_corpus) clReleaseMemObject(g_corpus);
    if (g_norms) clReleaseMemObject(g_norms);
    if (g_qbuf) clReleaseMemObject(g_qbuf);
    if (g_sbuf) clReleaseMemObject(g_sbuf);
    if (g_idxbuf) clReleaseMemObject(g_idxbuf);
    if (g_kqkt) clReleaseKernel(g_kqkt);
    if (g_kqkt_fused) clReleaseKernel(g_kqkt_fused);
    if (g_ktopk_local) clReleaseKernel(g_ktopk_local);
    if (g_ktopk_merge) clReleaseKernel(g_ktopk_merge);
    if (g_ktopk_merge_scores) clReleaseKernel(g_ktopk_merge_scores);
    if (g_kl2) clReleaseKernel(g_kl2);
    if (g_knorm) clReleaseKernel(g_knorm);
    if (g_local_idx) clReleaseMemObject(g_local_idx);
    if (g_prog) clReleaseProgram(g_prog);
    if (g_queue) clReleaseCommandQueue(g_queue);
    if (g_ctx) clReleaseContext(g_ctx);
    g_corpus = g_norms = g_qbuf = g_sbuf = g_idxbuf = g_local_idx = nullptr;
    g_kqkt = g_kqkt_fused = g_ktopk_local = g_ktopk_merge = g_ktopk = g_ktopk_merge_scores = nullptr;
    g_kl2 = g_knorm = nullptr;
    g_prog = nullptr; g_queue = nullptr; g_ctx = nullptr;
    g_state = -1;
}

// --- GPU enable (called from the constructors in speed_cpu.cpp) -----------
void SpeedEngine::enable_gpu(const std::vector<float>& corpus, int n, int dim) {
    // Transparent loader: use the caller-specified .so (opencl_lib_) if set,
    // else the WINNEX_OPENCL_LIB env var, else the standard ICD loader. No
    // hardcoded vendor cascade — the caller decides the driver.
    const char* loader = opencl_lib_.empty() ? nullptr : opencl_lib_.c_str();
    if (!load_opencl(loader)) {
        const char* shown = loader;
        if (!shown) shown = std::getenv("WINNEX_OPENCL_LIB");
        if (!shown || !*shown) shown = "libOpenCL.so.1";
        gpu_reason_ = std::string("OpenCL loader not found: ") + shown;
        if (require_gpu_) {
            throw std::runtime_error(
                "SpeedEngine: GPU required but the OpenCL loader is missing. "
                "Reason: " + gpu_reason_ + ". Set SpeedEngine(opencl_lib=...) "
                "or WINNEX_OPENCL_LIB to an installed loader/driver .so.");
        }
        std::fprintf(stderr,
                     "[Winnex Madhava] SpeedEngine: GPU unavailable — falling "
                     "back to CPU backend. Reason: %s\n",
                     gpu_reason_.c_str());
        return;
    }
    if (!SpeedEngine::gpu_available()) {
        gpu_reason_ = "no OpenCL GPU device found";
        if (require_gpu_) {
            throw std::runtime_error(
                "SpeedEngine: GPU required but no OpenCL device is available. "
                "Reason: " + gpu_reason_);
        }
        std::fprintf(stderr,
                     "[Winnex Madhava] SpeedEngine: GPU unavailable — falling "
                     "back to CPU backend. Reason: %s\n",
                     gpu_reason_.c_str());
        return;
    }
    init_gpu_impl();
    if (g_state != 1) {
        gpu_reason_ = "OpenCL kernels failed to build";
        if (require_gpu_) {
            throw std::runtime_error(
                "SpeedEngine: GPU required but the OpenCL backend failed to "
                "initialize. Reason: " + gpu_reason_);
        }
        std::fprintf(stderr,
                     "[Winnex Madhava] SpeedEngine: GPU unavailable — falling "
                     "back to CPU backend. Reason: %s\n",
                     gpu_reason_.c_str());
        return;
    }
    cl_int err;
    g_corpus = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY,
                              (size_t)n * dim * sizeof(float), nullptr, &err);
    if (!cl_err(err, "create corpus buffer")) { return; }
    clEnqueueWriteBuffer(g_queue, g_corpus, CL_TRUE, 0,
                         (size_t)n * dim * sizeof(float), corpus.data(),
                         0, nullptr, nullptr);
    // g_norms SEMPRE criado (o kernel fusionado o referencia como arg mesmo em
    // cosine; o valor e ignorado quando is_l2=0). Preenche com 1.0 em cosine
    // (neutral — 2*ip - 1.0 nao e usado).
    g_norms = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY,
                             (size_t)n * sizeof(float), nullptr, &err);
    if (!cl_err(err, "create norms buffer")) { return; }
    if (is_cosine_) {
        std::vector<float> ones((size_t)n, 1.0f);
        clEnqueueWriteBuffer(g_queue, g_norms, CL_TRUE, 0,
                             (size_t)n * sizeof(float), ones.data(),
                             0, nullptr, nullptr);
    } else {
        clEnqueueWriteBuffer(g_queue, g_norms, CL_TRUE, 0,
                             (size_t)n * sizeof(float), norms_.data(),
                             0, nullptr, nullptr);
    }
    use_gpu_ = true;
    std::fprintf(stderr,
                 "[Winnex Madhava] SpeedEngine: GPU backend enabled "
                 "(OpenCL QK^T + device topk, JIT-compiled kernels).\n");
}

// --- scores via OpenCL QKᵀ matmul, topk ON DEVICE -------------------------
// Computes scores[nq][N] = Q[nq][d] @ corpus[N][d]ᵀ, applies L2 correction
// on device, and runs the per-row topk on device. Only the nq·k indices are
// copied back. Buffers are reused across calls.
SpeedResult SpeedEngine::scores_gpu_topk(const float* queries, int nq, int k,
                                         std::vector<int>& out_indices) const {
    SpeedResult res;
    int N = n_, d = dim_;
    k = std::min(k, N);
    if (k > 16) k = 16;  // the topk kernel holds k<=16 partial lists

    cl_int err;
    // Reuse the query buffer (grow if needed).
    size_t qneed = (size_t)nq * d * sizeof(float);
    if (g_qbuf == nullptr || qneed > g_qcap) {
        if (g_qbuf) clReleaseMemObject(g_qbuf);
        g_qcap = qneed * 2;
        g_qbuf = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, g_qcap, nullptr, &err);
    }
    clEnqueueWriteBuffer(g_queue, g_qbuf, CL_TRUE, 0, qneed, queries,
                         0, nullptr, nullptr);

    // Normalize queries on device for cosine.
    if (is_cosine_) {
        size_t gs[] = {(size_t)nq * 256}, ls[] = {256};
        clSetKernelArg(g_knorm, 0, sizeof(cl_mem), &g_qbuf);
        clSetKernelArg(g_knorm, 1, sizeof(cl_mem), &g_qbuf);
        int a2 = nq, a3 = d;
        clSetKernelArg(g_knorm, 2, sizeof(int), &a2);
        clSetKernelArg(g_knorm, 3, sizeof(int), &a3);
        clEnqueueNDRangeKernel(g_queue, g_knorm, 1, nullptr, gs, ls,
                               0, nullptr, nullptr);
    }

    // --- FUSED QK^T + topk (v1.7.2): 1 kernel, sem scores[N] materializado. ---
    // M work-groups por query paralelizam o scan (todos os SMs ativos mesmo com
    // nq=1). Cada grupo varre um chunk contiguo de N e mantém top-k local.
    // O topk_merge depois funde os M top-k por query → top-k global exato.
    size_t ineed = (size_t)nq * k * sizeof(int);
    if (g_idxbuf == nullptr || ineed > g_idxcap) {
        if (g_idxbuf) clReleaseMemObject(g_idxbuf);
        g_idxcap = ineed * 2;
        g_idxbuf = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, g_idxcap, nullptr, &err);
    }

    // M: work-groups por query. Escolhe um M que cubra os SMs da GPU (32+),
    // mas limitado para chunks nao vazios. Para nq pequeno, M alto ativa
    // todos os SMs; para batch grande, o paralelismo entre queries domina.
    int M = 64;
    if ((size_t)M * nq > 4096) M = 4096 / nq;   // cap no grid total
    if (M > N) M = N;
    if (M < 1) M = 1;
    // Ajusta M para que cada chunk tenha >= lsz itens (evita idle em chunks
    // pequenos no single-query com N pequeno).
    int lsz = 256;
    if (N < lsz) lsz = N;
    if (N < (int)((size_t)lsz * M)) M = (N + lsz - 1) / lsz;
    if (M < 1) M = 1;

    size_t lneed = (size_t)nq * M * k * 2 * sizeof(float);  // intercala score+idx
    if (g_local_idx == nullptr || lneed > g_localcap) {
        if (g_local_idx) clReleaseMemObject(g_local_idx);
        g_localcap = lneed * 2;
        g_local_idx = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, g_localcap, nullptr, &err);
    }

    // qkt_fused_topk: grid nq*M work-groups x lsz itens.
    size_t gF[] = {(size_t)nq * M * lsz}, lF[] = {(size_t)lsz};
    clSetKernelArg(g_kqkt_fused, 0, sizeof(cl_mem), &g_qbuf);
    clSetKernelArg(g_kqkt_fused, 1, sizeof(cl_mem), &g_corpus);
    clSetKernelArg(g_kqkt_fused, 2, sizeof(cl_mem), &g_norms);   // usado so p/ L2
    clSetKernelArg(g_kqkt_fused, 3, sizeof(cl_mem), &g_local_idx);
    int f2 = nq, f3 = N, f4 = d, f5 = k, f6 = M;
    int f7 = is_cosine_ ? 0 : 1;
    clSetKernelArg(g_kqkt_fused, 4, sizeof(int), &f2);
    clSetKernelArg(g_kqkt_fused, 5, sizeof(int), &f3);
    clSetKernelArg(g_kqkt_fused, 6, sizeof(int), &f4);
    clSetKernelArg(g_kqkt_fused, 7, sizeof(int), &f5);
    clSetKernelArg(g_kqkt_fused, 8, sizeof(int), &f6);
    clSetKernelArg(g_kqkt_fused, 9, sizeof(int), &f7);
    clEnqueueNDRangeKernel(g_queue, g_kqkt_fused, 1, nullptr, gF, lF, 0, nullptr, nullptr);

    // topk_merge_scores: funde os M top-k por query usando os scores
    // intercalados no local_idx (sem reler scores[N]).
    size_t gM2[] = {(size_t)nq * lsz};
    clSetKernelArg(g_ktopk_merge_scores, 0, sizeof(cl_mem), &g_local_idx);
    clSetKernelArg(g_ktopk_merge_scores, 1, sizeof(cl_mem), &g_idxbuf);
    int b2 = nq, b3 = k, b4 = M;
    clSetKernelArg(g_ktopk_merge_scores, 2, sizeof(int), &b2);
    clSetKernelArg(g_ktopk_merge_scores, 3, sizeof(int), &b3);
    clSetKernelArg(g_ktopk_merge_scores, 4, sizeof(int), &b4);
    clEnqueueNDRangeKernel(g_queue, g_ktopk_merge_scores, 1, nullptr, gM2, lF, 0, nullptr, nullptr);

    // Copy back only the nq·k indices.
    out_indices.resize((size_t)nq * k);
    clEnqueueReadBuffer(g_queue, g_idxbuf, CL_TRUE, 0, ineed,
                        out_indices.data(), 0, nullptr, nullptr);

    res.bound_pairs = (long long)nq * N;
    res.bound_violations = 0;
    return res;
}

// Backward-compatible hook: fetch the full score matrix (legacy path).
void SpeedEngine::scores_gpu(const float* queries, int nq, float* scores_host) const {
    int N = n_, d = dim_;
    init_gpu_impl();
    cl_int err;
    std::vector<float> qnorm((size_t)nq * d);
    for (int qi = 0; qi < nq; qi++) {
        const float* q = queries + (size_t)qi * d;
        float qn = 0;
        for (int j = 0; j < d; j++) qn += q[j] * q[j];
        float inv = is_cosine_ && qn > 1e-12f ? 1.0f / std::sqrt(qn)
                                             : (is_cosine_ ? 0.f : 1.f);
        for (int j = 0; j < d; j++) qnorm[(size_t)qi * d + j] = q[j] * inv;
    }
    cl_mem qb = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, (size_t)nq * d * 4, nullptr, &err);
    cl_mem sb = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, (size_t)nq * N * 4, nullptr, &err);
    clEnqueueWriteBuffer(g_queue, qb, CL_TRUE, 0, (size_t)nq * d * 4, qnorm.data(), 0, nullptr, nullptr);
    size_t g2[] = {(size_t)nq * 256}, l2[] = {256};
    clSetKernelArg(g_kqkt, 0, sizeof(cl_mem), &qb);
    clSetKernelArg(g_kqkt, 1, sizeof(cl_mem), &g_corpus);
    clSetKernelArg(g_kqkt, 2, sizeof(cl_mem), &sb);
    int q3 = nq, q4 = N, q5 = d;
    clSetKernelArg(g_kqkt, 3, sizeof(int), &q3);
    clSetKernelArg(g_kqkt, 4, sizeof(int), &q4);
    clSetKernelArg(g_kqkt, 5, sizeof(int), &q5);
    clEnqueueNDRangeKernel(g_queue, g_kqkt, 1, nullptr, g2, l2, 0, nullptr, nullptr);
    clEnqueueReadBuffer(g_queue, sb, CL_TRUE, 0, (size_t)nq * N * 4, scores_host, 0, nullptr, nullptr);
    clReleaseMemObject(qb);
    clReleaseMemObject(sb);
}

} // namespace winnex_madhava
