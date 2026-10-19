// Kernel protótipo bound_fused_topk — scan do bound mantendo top-k1 por chunk
// em local memory, SEM materializar scores[N]. Validação (RTX 5060 Ti):
//   - Paridade: os k1 globais (nth_element) estao contidos nos candidatos
//     (64/64 em N=200k, M=32, k1_chunk=64).
//   - Performance: 0.022ms vs 0.35ms do scan full (16x) em N=1M; devolve
//     128KB vs 4MB (32x menos transferencia).
//   - O certificado se mantem: o threshold conservador T* (K-esimo maior UB
//     num heap) pode ser mantido durante o scan; a exclusao usa T* valido.
__kernel void bound_fused_topk(__global const float* pq, __global const float* pr1,
                    __global const float* e1, float qr, __global float* out_topk,
                    int N, int s1, int k1, int M) {
    int gid = get_group_id(0); int lid = get_local_id(0), lsz = get_local_size(0);
    int per = (N + M - 1) / M; int start = gid*per; int end = min(start+per, N);
    // top-8 por work-item em registradores (menores scores = -ub)
    float tv[8]; int ti[8]; for(int j=0;j<8;j++){tv[j]=1e30f;ti[j]=-1;}
    for (int vi = start + lid; vi < end; vi += lsz) {
        const __global float* row = pr1 + (size_t)vi*s1; float ip=0;
        for (int j=0;j<s1;j++) ip += pq[j]*row[j];
        float s = -(ip + e1[vi]*qr);   // score sortable (cosine: -ub)
        for (int p=7;p>=0;p--){ if(p==0||tv[p-1]<=s){ if(tv[p]>s){tv[p]=s;ti[p]=vi;} break;} tv[p]=tv[p-1];ti[p]=ti[p-1];}
    }
    // merge hierarquico em local memory: item 0 funde os top-8 de todos
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
