/**
 * test_topk_parallel.cpp — compara o argsort top-k OpenCL atual (insertion sort
 * 1 work-group por query) vs o novo divide-and-conquer (M work-groups + merge).
 *
 * Este é um teste STANDALONE que valida corretude e mede o ganho ANTES de
 * integrar ao engine. Usa o OpenCL diretamente (dlopen) com os mesmos kernels.
 *
 * Build (com o prefixo ocl-local):
 *   g++ tests/test_topk_parallel.cpp -Ivendor -Iinclude \
 *       -I/home/wnnx_user/ocl-local/include -L/home/wnnx_user/ocl-local/lib \
 *       -lOpenCL -ldl -O2 -o /tmp/test_topk_parallel
 * Run:
 *   LD_LIBRARY_PATH=/home/wnnx_user/ocl-local/lib /tmp/test_topk_parallel
 */
#include <CL/cl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

static const char* KERNELS = R"(
// ---- Kernel ATUAL (1 work-group por query, insertion sort) ----
__kernel void topk_old(__global const float* scores, __global int* idx,
                       int nq, int N, int k) {
    int qi = get_group_id(0);
    if (qi >= nq) return;
    __local float lv[256][16];
    __local int li[256][16];
    int lid = get_local_id(0), lsz = get_local_size(0);
    for (int j = 0; j < k; j++) { lv[lid][j] = -1e30f; li[lid][j] = -1; }
    for (int vi = lid; vi < N; vi += lsz) {
        float s = scores[qi*N + vi];
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
        for (int j = 0; j < k; j++) idx[qi*k + j] = ti[j];
    }
}

// ---- Kernel NOVO, fase 1: M work-groups por query, top-k local ----
// grid = (nq * M * lsz) threads, 1 work-group = 1 (query, chunk).
// Cada grupo varre um chunk contíguo de N e mantém top-k local em local mem.
__kernel void topk_local(__global const float* scores, __global int* local_idx,
                         int nq, int N, int k, int M) {
    int gid = get_group_id(0);
    int qi = gid / M;              // query
    int chunk = gid % M;           // qual chunk (0..M-1)
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
        // merge dos top-k de cada work-item deste grupo -> top-k do chunk
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
        for (int j = 0; j < k; j++) {
            local_idx[((size_t)qi*M + chunk)*k + j] = ti[j];
        }
    }
}

// ---- Kernel NOVO, fase 2: merge dos top-k de M chunks por query ----
// grid = nq work-groups. Cada um merge M listas de k.
__kernel void topk_merge(__global const float* scores, __global const int* local_idx,
                         __global int* idx, int nq, int N, int k, int M) {
    int qi = get_group_id(0);
    if (qi >= nq) return;
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float tv[16]; __local int ti[16];
    if (lid == 0) {
        for (int j = 0; j < k; j++) { tv[j] = -1e30f; ti[j] = -1; }
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < k; j++) {
                int cand_id = local_idx[((size_t)qi*M + m)*k + j];
                if (cand_id < 0) continue;
                float s = scores[(size_t)qi*N + cand_id];
                for (int p = k-1; p >= 0; p--) {
                    if (p == 0 || tv[p-1] >= s) {
                        if (tv[p] < s) { tv[p] = s; ti[p] = cand_id; }
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

static const char* err_str(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "OK";
        case CL_INVALID_WORK_GROUP_SIZE: return "INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_PROGRAM: return "INVALID_PROGRAM";
        case CL_INVALID_KERNEL_NAME: return "INVALID_KERNEL_NAME";
        case CL_BUILD_PROGRAM_FAILURE: return "BUILD_PROGRAM_FAILURE";
        default: return "?";
    }
}

int main() {
    // --- setup OpenCL (dlopen) ---
    void* h = dlopen("libOpenCL.so.1", RTLD_NOW);
    if (!h) { std::printf("FALHA: sem libOpenCL.so.1 (%s)\n", dlerror()); return 1; }
    using F = cl_int(*)(cl_uint, cl_platform_id*, cl_uint*);
    auto clGetPlatformIDs = (F)dlsym(h, "clGetPlatformIDs");
    using FD = cl_int(*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    auto clGetDeviceIDs = (FD)dlsym(h, "clGetDeviceIDs");
    using FC = cl_context(*)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
    auto clCreateContext = (FC)dlsym(h, "clCreateContext");
    using FQ = cl_command_queue(*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
    auto clCreateCommandQueue = (FQ)dlsym(h, "clCreateCommandQueue");
    using FP = cl_program(*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    auto clCreateProgramWithSource = (FP)dlsym(h, "clCreateProgramWithSource");
    using FB = cl_int(*)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
    auto clBuildProgram = (FB)dlsym(h, "clBuildProgram");
    using FG = cl_int(*)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
    auto clGetProgramBuildInfo = (FG)dlsym(h, "clGetProgramBuildInfo");
    using FK = cl_kernel(*)(cl_program, const char*, cl_int*);
    auto clCreateKernel = (FK)dlsym(h, "clCreateKernel");
    using FMB = cl_mem(*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    auto clCreateBuffer = (FMB)dlsym(h, "clCreateBuffer");
    using FSA = cl_int(*)(cl_kernel, cl_uint, size_t, const void*);
    auto clSetKernelArg = (FSA)dlsym(h, "clSetKernelArg");
    using FEN = cl_int(*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, cl_event*, cl_event*);
    auto clEnqueueNDRangeKernel = (FEN)dlsym(h, "clEnqueueNDRangeKernel");
    using FWB = cl_int(*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
    auto clEnqueueWriteBuffer = (FWB)dlsym(h, "clEnqueueWriteBuffer");
    using FRB = cl_int(*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
    auto clEnqueueReadBuffer = (FRB)dlsym(h, "clEnqueueReadBuffer");

    cl_uint np = 0; clGetPlatformIDs(0, nullptr, &np);
    cl_platform_id plat; clGetPlatformIDs(1, &plat, nullptr);
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNELS, nullptr, &err);
    err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[16384]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::printf("BUILD FALHOU:\n%s\n", log); return 1;
    }
    cl_kernel k_old = clCreateKernel(prog, "topk_old", &err);
    cl_kernel k_local = clCreateKernel(prog, "topk_local", &err);
    cl_kernel k_merge = clCreateKernel(prog, "topk_merge", &err);

    // --- dados de teste ---
    int nq = 32, N = 200000, k = 10, M = 32;
    std::vector<float> scores((size_t)nq * N);
    std::srand(42);
    for (auto& x : scores) x = (float)std::rand() / RAND_MAX;

    cl_mem ds = clCreateBuffer(ctx, CL_MEM_READ_ONLY, scores.size()*4, nullptr, &err);
    cl_mem didx_old = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nq*k*4, nullptr, &err);
    cl_mem dlocal = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)nq*M*k*4, nullptr, &err);
    cl_mem didx_new = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nq*k*4, nullptr, &err);
    clEnqueueWriteBuffer(queue, ds, CL_TRUE, 0, scores.size()*4, scores.data(), 0, nullptr, nullptr);

    // --- validar corretude vs CPU (reference) ---
    std::vector<int> ref(nq*k);
    for (int qi = 0; qi < nq; qi++) {
        std::vector<std::pair<float,int>> s(N);
        for (int i = 0; i < N; i++) s[i] = {scores[(size_t)qi*N+i], i};
        std::partial_sort(s.begin(), s.begin()+k, s.end(),
                          [](auto&a, auto&b){return a.first>b.first;});
        for (int j = 0; j < k; j++) ref[qi*k+j] = s[j].second;
    }

    // old
    size_t g_old[] = {(size_t)nq*256}, l_old[] = {256};
    clSetKernelArg(k_old,0,sizeof(cl_mem),&ds); clSetKernelArg(k_old,1,sizeof(cl_mem),&didx_old);
    int a2=nq,a3=N,a4=k; clSetKernelArg(k_old,2,sizeof(int),&a2); clSetKernelArg(k_old,3,sizeof(int),&a3); clSetKernelArg(k_old,4,sizeof(int),&a4);
    clEnqueueNDRangeKernel(queue,k_old,1,nullptr,g_old,l_old,0,nullptr,nullptr);
    std::vector<int> idx_old(nq*k); clEnqueueReadBuffer(queue,didx_old,CL_TRUE,0,nq*k*4,idx_old.data(),0,nullptr,nullptr);

    // new
    size_t g_new[] = {(size_t)nq*M*256}, l_new[] = {256};
    clSetKernelArg(k_local,0,sizeof(cl_mem),&ds); clSetKernelArg(k_local,1,sizeof(cl_mem),&dlocal);
    int b2=nq,b3=N,b4=k,b5=M; clSetKernelArg(k_local,2,sizeof(int),&b2); clSetKernelArg(k_local,3,sizeof(int),&b3); clSetKernelArg(k_local,4,sizeof(int),&b4); clSetKernelArg(k_local,5,sizeof(int),&b5);
    clEnqueueNDRangeKernel(queue,k_local,1,nullptr,g_new,l_new,0,nullptr,nullptr);
    clSetKernelArg(k_merge,0,sizeof(cl_mem),&ds); clSetKernelArg(k_merge,1,sizeof(cl_mem),&dlocal); clSetKernelArg(k_merge,2,sizeof(cl_mem),&didx_new);
    int c2=nq,c3=N,c4=k,c5=M; clSetKernelArg(k_merge,3,sizeof(int),&c2); clSetKernelArg(k_merge,4,sizeof(int),&c3); clSetKernelArg(k_merge,5,sizeof(int),&c4); clSetKernelArg(k_merge,6,sizeof(int),&c5);
    clEnqueueNDRangeKernel(queue,k_merge,1,nullptr,g_old,l_old,0,nullptr,nullptr);
    std::vector<int> idx_new(nq*k); clEnqueueReadBuffer(queue,didx_new,CL_TRUE,0,nq*k*4,idx_new.data(),0,nullptr,nullptr);

    // corretude
    int ok_old = 0, ok_new = 0;
    for (int qi = 0; qi < nq; qi++) {
        std::vector<int> rold(idx_old.begin()+qi*k, idx_old.begin()+(qi+1)*k);
        std::vector<int> rnew(idx_new.begin()+qi*k, idx_new.begin()+(qi+1)*k);
        std::vector<int> rref(ref.begin()+qi*k, ref.begin()+(qi+1)*k);
        std::sort(rold.begin(), rold.end()); std::sort(rnew.begin(), rnew.end()); std::sort(rref.begin(), rref.end());
        if (rold == rref) ok_old++;
        if (rnew == rref) ok_new++;
    }
    std::printf("nq=%d N=%d k=%d M=%d\n", nq, N, k, M);
    std::printf("corretude old : %d/%d\n", ok_old, nq);
    std::printf("corretude new : %d/%d\n", ok_new, nq);

    // --- benchmark ---
    auto bench = [&](auto fn) {
        // warmup
        fn();
        int iters = 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1-t0).count() / iters;
    };
    auto run_old = [&]() {
        clEnqueueNDRangeKernel(queue,k_old,1,nullptr,g_old,l_old,0,nullptr,nullptr);
        clEnqueueReadBuffer(queue,didx_old,CL_TRUE,0,nq*k*4,idx_old.data(),0,nullptr,nullptr);
    };
    auto run_new = [&]() {
        clEnqueueNDRangeKernel(queue,k_local,1,nullptr,g_new,l_new,0,nullptr,nullptr);
        clEnqueueNDRangeKernel(queue,k_merge,1,nullptr,g_old,l_old,0,nullptr,nullptr);
        clEnqueueReadBuffer(queue,didx_new,CL_TRUE,0,nq*k*4,idx_new.data(),0,nullptr,nullptr);
    };
    double t_old = bench(run_old);
    double t_new = bench(run_new);
    std::printf("tempo old: %.3f ms (batch %d queries)\n", t_old, nq);
    std::printf("tempo new: %.3f ms (batch %d queries)\n", t_new, nq);
    std::printf("GANHO: %.2fx\n", t_old/t_new);

    return 0;
}
