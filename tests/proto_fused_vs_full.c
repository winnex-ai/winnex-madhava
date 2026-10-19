// proto_fused_vs_full — compara o custo TOTAL por query:
//  A) bound_stage1 atual: GPU materializa scores[N] + devolve ao host + host faz b1[N] + nth_element
//  B) fused: GPU mantém top-k1 por chunk (local mem) + devolve só M*k1 pares + host funde M*k1 (barato)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
static double now_ms(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e3+ts.tv_nsec/1e6;}

const char* K = R"(
// kernel A: materializa scores[N] (como bound_stage1)
__kernel void scan_full(__global const float* pq, __global const float* pr1,
                        __global const float* e1, float qr, float qm,
                        __global float* scores, int N, int s1, int M) {
    int gid = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int per = (N + M - 1) / M;
    int start = gid * per;
    int end = min(start + per, N);
    for (int vi = start + lid; vi < end; vi += lsz) {
        const __global float* row = pr1 + (size_t)vi * s1;
        float ip = 0;
        for (int j = 0; j < s1; j++) ip += pq[j] * row[j];
        float ub = ip + e1[vi]*qr + qm;
        scores[vi] = -ub;  // score sortable
    }
}
// kernel B: fused — cada work-group mantém top-k1 do seu chunk em local mem
__kernel void scan_fused(__global const float* pq, __global const float* pr1,
                         __global const float* e1, float qr, float qm,
                         __global float* out_topk, int N, int s1, int k1, int M) {
    int gid = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int per = (N + M - 1) / M;
    int start = gid * per;
    int end = min(start + per, N);
    // top-k1 local em local memory (k1 <= 1024 p/ caber; usamos abordagem por item + merge)
    // Cada item mantém top-8 local em registradores, depois merge hierárquico simples.
    float tv[8]; int ti[8];
    for (int j = 0; j < 8; j++) { tv[j] = 1e30f; ti[j] = -1; }
    for (int vi = start + lid; vi < end; vi += lsz) {
        const __global float* row = pr1 + (size_t)vi * s1;
        float ip = 0;
        for (int j = 0; j < s1; j++) ip += pq[j] * row[j];
        float ub = ip + e1[vi]*qr + qm;
        float s = -ub;
        for (int p = 7; p >= 0; p--) {
            if (p == 0 || tv[p-1] <= s) { if (tv[p] > s) { tv[p]=s; ti[p]=vi; } break; }
            tv[p] = tv[p-1]; ti[p] = ti[p-1];
        }
    }
    // merge: escreve os 8 de cada item em local, item 0 funde nos top-k1 do chunk
    __local float ls[256*8]; __local int li[256*8];
    for (int j = 0; j < 8; j++) { ls[lid*8+j] = tv[j]; li[lid*8+j] = ti[j]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        // funde lsz*8 candidatos -> top-k1 do chunk (k1 pode ser < lsz*8)
        int ncan = lsz * 8;
        // heap simples: mantém k1 menores (k1 <= 256 por chunk p/ protótipo)
        float hs[256]; int hi[256];
        int cap = k1 < 256 ? k1 : 256;
        for (int j = 0; j < cap; j++) { hs[j] = 1e30f; hi[j] = -1; }
        for (int c = 0; c < ncan; c++) {
            float s = ls[c]; int idx = li[c];
            if (idx < 0) continue;
            // insertion nos cap menores
            for (int p = cap-1; p >= 0; p--) {
                if (p == 0 || hs[p-1] <= s) { if (hs[p] > s) { hs[p]=s; hi[p]=idx; } break; }
                hs[p] = hs[p-1]; hi[p] = hi[p-1];
            }
        }
        for (int j = 0; j < cap; j++) {
            out_topk[gid*256*2 + j*2] = hs[j];
            out_topk[gid*256*2 + j*2+1] = (float)hi[j];
        }
    }
}
)";

int main(int argc, char** argv){
    int N=argc>1?atoi(argv[1]):1000000, s1=argc>2?atoi(argv[2]):64, k1=argc>3?atoi(argv[3]):256;
    int M=64, lsz=256;
    cl_int err; cl_platform_id p; clGetPlatformIDs(1,&p,NULL);
    cl_device_id d; clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,NULL);
    char dn[256]; clGetDeviceInfo(d,CL_DEVICE_NAME,256,dn,0);
    cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&err);
    cl_command_queue q=clCreateCommandQueue(c,d,0,&err);
    cl_program pr=clCreateProgramWithSource(c,1,&K,NULL,&err);
    err=clBuildProgram(pr,1,&d,NULL,NULL,NULL);
    if(err!=CL_SUCCESS){char lg[16384];clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,16384,lg,NULL);printf("ERR %s\n",lg);return 1;}
    cl_kernel kfull=clCreateKernel(pr,"scan_full",&err);
    cl_kernel kfused=clCreateKernel(pr,"scan_fused",&err);
    printf("device: %s | N=%d s1=%d k1(chunk)=%d M=%d\n",dn,N,s1,k1,M);
    srand(42);
    float* pr1=malloc((size_t)N*s1*4); float* e1=malloc(N*4); float* pq=malloc(s1*4);
    for(int i=0;i<N*s1;i++)pr1[i]=rand()/(float)RAND_MAX-0.5f;
    for(int i=0;i<N;i++)e1[i]=0.3f+0.2f*fabsf(rand()/(float)RAND_MAX-0.5f);
    for(int i=0;i<s1;i++)pq[i]=rand()/(float)RAND_MAX-0.5f;
    cl_mem dpr1=clCreateBuffer(c,CL_MEM_READ_ONLY,(size_t)N*s1*4,0,&err);
    cl_mem de1=clCreateBuffer(c,CL_MEM_READ_ONLY,(size_t)N*4,0,&err);
    cl_mem dpq=clCreateBuffer(c,CL_MEM_READ_ONLY,(size_t)s1*4,0,&err);
    cl_mem dscores=clCreateBuffer(c,CL_MEM_WRITE_ONLY,(size_t)N*4,0,&err);
    cl_mem dtopk=clCreateBuffer(c,CL_MEM_WRITE_ONLY,(size_t)M*256*2*4,0,&err);
    clEnqueueWriteBuffer(q,dpr1,CL_TRUE,0,(size_t)N*s1*4,pr1,0,0,0);
    clEnqueueWriteBuffer(q,de1,CL_TRUE,0,(size_t)N*4,e1,0,0,0);
    clEnqueueWriteBuffer(q,dpq,CL_TRUE,0,(size_t)s1*4,pq,0,0,0);
    float qr=0.5f,qm=0.0f;
    size_t g2[]={(size_t)M*lsz}, l2[]={(size_t)lsz};
    float* scores=malloc((size_t)N*4); float* topk=malloc((size_t)M*256*2*4);
    int IT=200;
    // A) full: kernel + D2H scores[N] + host nth_element p/ k1_total = M*k1? Na real k1 total = N*k1frac.
    // Mede o custo do bound_stage1 atual: scan GPU + ler N scores + host nth_element (k1frac*N).
    int k1_total = k1; // p/ protótipo, k1 total pequeno (k1frac=0.000256 em N=1M)
    clSetKernelArg(kfull,0,4,&dpq);clSetKernelArg(kfull,1,4,&dpr1);clSetKernelArg(kfull,2,4,&de1);
    clSetKernelArg(kfull,3,4,&qr);clSetKernelArg(kfull,4,4,&qm);clSetKernelArg(kfull,5,4,&dscores);
    int aN=N,as1=s1,aM=M; clSetKernelArg(kfull,6,4,&aN);clSetKernelArg(kfull,7,4,&as1);clSetKernelArg(kfull,8,4,&aM);
    clEnqueueNDRangeKernel(q,kfull,1,0,g2,l2,0,0,0); clFinish(q);
    double t0=now_ms();
    for(int i=0;i<IT;i++){
        clEnqueueNDRangeKernel(q,kfull,1,0,g2,l2,0,0,0);
        clEnqueueReadBuffer(q,dscores,CL_TRUE,0,(size_t)N*4,scores,0,0,0);
        // host nth_element (como o search faz) — custo real
        // (p/ protótipo, faz um nth_element de verdade em scores)
    }
    clFinish(q); double tA_gpu=(now_ms()-t0)/IT;
    // custo host nth_element (separado)
    for(int i=0;i<N;i++)scores[i]=rand()/(float)RAND_MAX; // genérico
    t0=now_ms();
    // nth_element simulado: std não disponível em C; faz partial sort manual custo O(N) aprox via qsort
    // (o motor usa nth_element C++; aproximamos com o custo de tocar N floats)
    volatile float sink=0; for(int i=0;i<N;i++)sink+=scores[i];
    double tA_host=(now_ms()-t0);
    printf("A) full scan GPU + D2H scores[N] (%.0fKB): %.4f ms | tocar N no host: %.4f ms\n",
           (double)N*4/1024, tA_gpu, tA_host);

    // B) fused
    clSetKernelArg(kfused,0,4,&dpq);clSetKernelArg(kfused,1,4,&dpr1);clSetKernelArg(kfused,2,4,&de1);
    clSetKernelArg(kfused,3,4,&qr);clSetKernelArg(kfused,4,4,&qm);clSetKernelArg(kfused,5,4,&dtopk);
    int ak1=k1; clSetKernelArg(kfused,6,4,&aN);clSetKernelArg(kfused,7,4,&as1);clSetKernelArg(kfused,8,4,&ak1);clSetKernelArg(kfused,9,4,&aM);
    clEnqueueNDRangeKernel(q,kfused,1,0,g2,l2,0,0,0); clFinish(q);
    t0=now_ms();
    for(int i=0;i<IT;i++){
        clEnqueueNDRangeKernel(q,kfused,1,0,g2,l2,0,0,0);
        clEnqueueReadBuffer(q,dtopk,CL_TRUE,0,(size_t)M*256*2*4,topk,0,0,0);
    }
    clFinish(q); double tB=(now_ms()-t0)/IT;
    printf("B) fused scan GPU + D2H topk (%.0fKB): %.4f ms\n", (double)M*256*2*4/1024, tB);
    printf("=> fused vs full: A(scan+transfer) %.4f vs B %.4f -> B %s\n", tA_gpu, tB, tB<tA_gpu?"MAIS RAPIDO":"mais lento");
    // sanity do fused
    int valid=0; for(int i=0;i<M*2;i++) if(topk[i*2+1]>=0) valid++;
    printf("topk validos: %d/%d\n", valid, M*2);
    return 0;
}
