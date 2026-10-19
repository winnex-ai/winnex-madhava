// Paridade FINAL: os k1 GLOBAIS (nth_element) estão contidos nos candidatos do fused multi-chunk?
// Leitura correta: out_topk[gid*256*2 + j*2] p/ score, +1 p/ idx. k1 por chunk = 64.
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
const char* K = R"(
__kernel void fused(__global const float* pq, __global const float* pr1,
                    __global const float* e1, float qr, __global float* out_topk,
                    int N, int s1, int k1, int M) {
    int gid = get_group_id(0); int lid = get_local_id(0), lsz = get_local_size(0);
    int per = (N + M - 1) / M; int start = gid*per; int end = min(start+per, N);
    float tv[8]; int ti[8]; for(int j=0;j<8;j++){tv[j]=1e30f;ti[j]=-1;}
    for (int vi = start + lid; vi < end; vi += lsz) {
        const __global float* row = pr1 + (size_t)vi*s1; float ip=0;
        for (int j=0;j<s1;j++) ip += pq[j]*row[j];
        float s = -(ip + e1[vi]*qr);
        for (int p=7;p>=0;p--){ if(p==0||tv[p-1]<=s){ if(tv[p]>s){tv[p]=s;ti[p]=vi;} break;} tv[p]=tv[p-1];ti[p]=ti[p-1];}
    }
    __local float ls[2048]; __local int li[2048];
    for(int j=0;j<8;j++){ls[lid*8+j]=tv[j];li[lid*8+j]=ti[j];}
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid==0){
        int ncan=lsz*8; float hs[256]; int hi[256]; int cap=k1<256?k1:256;
        for(int j=0;j<cap;j++){hs[j]=1e30f;hi[j]=-1;}
        for(int c=0;c<ncan;c++){ float s=ls[c]; int idx=li[c]; if(idx<0)continue;
            for(int p=cap-1;p>=0;p--){ if(p==0||hs[p-1]<=s){ if(hs[p]>s){hs[p]=s;hi[p]=idx;} break;} hs[p]=hs[p-1];hi[p]=hi[p-1];}}
        for(int j=0;j<cap;j++){ out_topk[gid*256*2+j*2]=hs[j]; out_topk[gid*256*2+j*2+1]=(float)hi[j];}
    }
}
)";
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):200000, s1=argc>2?atoi(argv[2]):64, k1=argc>3?atoi(argv[3]):64, M=32, lsz=256;
    cl_int err;cl_platform_id p;clGetPlatformIDs(1,&p,0);cl_device_id d;clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0);
    cl_context c=clCreateContext(0,1,&d,0,0,&err);cl_command_queue q=clCreateCommandQueue(c,d,0,&err);
    cl_program pr=clCreateProgramWithSource(c,1,&K,0,&err);
    err=clBuildProgram(pr,1,&d,0,0,0);if(err){char lg[8192];clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,8192,lg,0);printf("BUILD %s\n",lg);return 1;}
    cl_kernel k=clCreateKernel(pr,"fused",&err);
    srand(42);
    float*pr1=malloc((size_t)N*s1*4);float*e1=malloc(N*4);float*pq=malloc(s1*4);
    for(int i=0;i<N*s1;i++)pr1[i]=rand()/(float)RAND_MAX-0.5f;
    for(int i=0;i<N;i++)e1[i]=0.3f+0.2f*fabsf(rand()/(float)RAND_MAX-0.5f);
    for(int i=0;i<s1;i++)pq[i]=rand()/(float)RAND_MAX-0.5f;
    cl_mem dpr1=clCreateBuffer(c,CL_MEM_READ_ONLY,(size_t)N*s1*4,0,&err);
    cl_mem de1=clCreateBuffer(c,CL_MEM_READ_ONLY,(size_t)N*4,0,&err);
    cl_mem dpq=clCreateBuffer(c,CL_MEM_READ_ONLY,(size_t)s1*4,0,&err);
    cl_mem dout=clCreateBuffer(c,CL_MEM_WRITE_ONLY,(size_t)M*256*2*4,0,&err);
    clEnqueueWriteBuffer(q,dpr1,CL_TRUE,0,(size_t)N*s1*4,pr1,0,0,0);
    clEnqueueWriteBuffer(q,de1,CL_TRUE,0,(size_t)N*4,e1,0,0,0);
    clEnqueueWriteBuffer(q,dpq,CL_TRUE,0,(size_t)s1*4,pq,0,0,0);
    float qr=0.5f;
    clSetKernelArg(k,0,sizeof(cl_mem),&dpq);clSetKernelArg(k,1,sizeof(cl_mem),&dpr1);
    clSetKernelArg(k,2,sizeof(cl_mem),&de1);clSetKernelArg(k,3,sizeof(float),&qr);clSetKernelArg(k,4,sizeof(cl_mem),&dout);
    int aN=N,as1=s1,ak1=k1,aM=M;
    clSetKernelArg(k,5,sizeof(int),&aN);clSetKernelArg(k,6,sizeof(int),&as1);clSetKernelArg(k,7,sizeof(int),&ak1);clSetKernelArg(k,8,sizeof(int),&aM);
    size_t g2[]={(size_t)M*lsz},l2[]={(size_t)lsz};
    clEnqueueNDRangeKernel(q,k,1,0,g2,l2,0,0,0);clFinish(q);
    float* topk=malloc((size_t)M*256*2*4);clEnqueueReadBuffer(q,dout,CL_TRUE,0,(size_t)M*256*2*4,topk,0,0,0);
    // scores serial de todos (p/ achar os k1 globais por nth_element)
    float* sc=malloc(N*4); int* idx=malloc(N*sizeof(int));
    for(int i=0;i<N;i++){float ip=0;for(int j=0;j<s1;j++)ip+=pr1[i*s1+j]*pq[j];sc[i]=-(ip+e1[i]*qr);idx[i]=i;}
    // nth_element: k1 menores
    for(int i=0;i<k1;i++){int bi=i;for(int j=i+1;j<N;j++)if(sc[j]<sc[bi])bi=j;float t=sc[i];sc[i]=sc[bi];sc[bi]=t;int ti=idx[i];idx[i]=idx[bi];idx[bi]=ti;}
    // verifica: cada um dos k1 globais está nos candidatos do fused (M*k1)?
    int overlap=0; int* seen=calloc((size_t)N,sizeof(int));
    for(int m=0;m<M;m++)for(int j=0;j<k1;j++){
        int id=(int)topk[m*256*2+j*2+1]; if(id>=0&&id<N) seen[id]=1;}
    for(int i=0;i<k1;i++) if(seen[idx[i]]) overlap++;
    printf("N=%d k1_global=%d M=%d k1_chunk=%d | overlap(k1 globais nos candidatos fused): %d/%d\n",N,k1,M,k1,overlap,k1);
    printf("PARIDADE: %s\n", overlap==k1?"TODOS os k1 globais contidos (fused cobre o top-k1)":"FALHOU");
    return 0;
}
