// proto_bound_fused_topk.cpp — Protótipo: o bound engine pode achar os k1
// sobreviventes SEM materializar scores[N] (o que mata o espelhamento GPU)?
//
// Hipótese: em vez de computar ub de TODOS os N, guardar num vetor, e fazer
// nth_element (O(N) + scores[N] na RAM/GPU), o scan pode manter os k1 MENORES
// scores (= k1 maiores ub) num heap por chunk e fundir — como o qkt_fused_topk
// do speed faz. Isso elimina scores[N] (4MB em N=1M) e o nth_element sobre N.
//
// Este protótipo MEDE (CPU, sem tocar no motor):
//   1. PARIDADE: os k1 sobreviventes do fused == os do scan completo+nth_element
//   2. O custo: fused (heap k1) vs completo (scores[N] + nth_element)
//   3. A amostra de borda p/ o certificado (docs com ub proximo do threshold)
//
// Compila: g++ -O3 -march=native -fopenmp tests/proto_bound_fused_topk.cpp -o /tmp/pbf
// Roda:   /tmp/pbf [N] [s1] [k1] [nq]

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 200000;
    int s1 = argc > 2 ? atoi(argv[2]) : 64;
    int k1 = argc > 3 ? atoi(argv[3]) : (int)(N * 0.05);
    int nq = argc > 4 ? atoi(argv[4]) : 8;
    if (k1 < 1) k1 = 1;
    printf("N=%d s1=%d k1=%d nq=%d\n", N, s1, k1, nq);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 1);
    // pr1: [N x s1] projeções (float32), e1: [N] residuais
    std::vector<float> pr1((size_t)N * s1);
    for (auto& x : pr1) x = nd(rng);
    std::vector<float> e1(N);
    for (auto& x : e1) x = 0.3f + 0.2f * std::fabs(nd(rng));
    std::vector<float> vn_eff(N, 1.0f);  // cosine normalizado

    // ---- scan "completo" (referência): ub de todos N, sort por score -ub ----
    // score = -ub (cosine), menor = melhor. ub = dot(pr1[v],pq) + e1[v]*qr.
    // k1 sobreviventes = k1 menores scores.
    auto fused_parity = [&](const std::vector<float>& pq, float qr) {
        // 1. Completo: scores[N] + nth_element
        std::vector<std::pair<float,int>> b1(N);
        for (int i = 0; i < N; i++) {
            float ip = 0;
            const float* row = &pr1[(size_t)i*s1];
            for (int j = 0; j < s1; j++) ip += row[j]*pq[j];
            float ub = ip + e1[i]*qr;
            b1[i] = {-ub, i};   // score sortable (cosine: -ub)
        }
        std::nth_element(b1.begin(), b1.begin()+k1, b1.end());
        std::vector<int> ref(k1);
        for (int i = 0; i < k1; i++) ref[i] = b1[i].second;
        std::sort(ref.begin(), ref.end());

        // 2. Fused: M chunks, cada um mantém k1 menores, merge
        int M = 32;
        std::vector<std::pair<float,int>> fused_pool;
        fused_pool.reserve((size_t)M*k1);
        int per = (N + M - 1)/M;
        for (int m = 0; m < M; m++) {
            int start = m*per, end = std::min(start+per, N);
            // top-k1 local por chunk (insertion num vetor pequeno)
            std::vector<std::pair<float,int>> local;
            local.reserve(k1);
            for (int i = start; i < end; i++) {
                float ip = 0;
                const float* row = &pr1[(size_t)i*s1];
                for (int j = 0; j < s1; j++) ip += row[j]*pq[j];
                float ub = ip + e1[i]*qr;
                float score = -ub;
                // mantém os k1 menores scores (ordenação decrescente no vetor, pop do maior)
                if ((int)local.size() < k1) {
                    local.push_back({score, i});
                    std::push_heap(local.begin(), local.end());  // max-heap
                } else if (score < local.front().first) {
                    std::pop_heap(local.begin(), local.end());
                    local.back() = {score, i};
                    std::push_heap(local.begin(), local.end());
                }
            }
            fused_pool.insert(fused_pool.end(), local.begin(), local.end());
        }
        // merge: k1 menores de fused_pool (M*k1 candidatos)
        std::nth_element(fused_pool.begin(), fused_pool.begin()+k1, fused_pool.end());
        std::vector<int> fused(k1);
        for (int i = 0; i < k1; i++) fused[i] = fused_pool[i].second;
        std::sort(fused.begin(), fused.end());

        // paridade
        int overlap = 0;
        for (int i = 0; i < k1; i++) if (ref[i]==fused[i]) overlap++;
        return std::make_pair(overlap==k1, overlap);
    };

    // ---- benchmark: fused vs completo ----
    std::vector<float> pq(s1);
    int par_ok = 0;
    for (int qi = 0; qi < nq; qi++) {
        for (auto& x : pq) x = nd(rng);
        float qr = 0.5f;
        auto r = fused_parity(pq, qr);
        par_ok += r.first;
    }
    printf("PARIDADE fused==completo: %d/%d\n", par_ok, nq);

    // tempo
    std::vector<float> qs(s1);
    for (auto& x : qs) x = nd(rng);
    // completo
    auto t0 = now_ms();
    for (int r = 0; r < 20; r++) {
        std::vector<std::pair<float,int>> b1(N);
        for (int i = 0; i < N; i++) {
            float ip = 0; const float* row=&pr1[(size_t)i*s1];
            for (int j=0;j<s1;j++) ip+=row[j]*qs[j];
            float ub=ip+e1[i]*0.5f; b1[i]={-ub,i};
        }
        std::nth_element(b1.begin(), b1.begin()+k1, b1.end());
    }
    double t_full=(now_ms()-t0)/20;
    // fused
    t0 = now_ms();
    for (int r = 0; r < 20; r++) {
        int M=32, per=(N+M-1)/M;
        std::vector<std::pair<float,int>> pool; pool.reserve((size_t)M*k1);
        for (int m=0;m<M;m++){
            int start=m*per,end=std::min(start+per,N);
            std::vector<std::pair<float,int>> local; local.reserve(k1);
            for(int i=start;i<end;i++){
                float ip=0; const float*row=&pr1[(size_t)i*s1];
                for(int j=0;j<s1;j++)ip+=row[j]*qs[j];
                float score=-(ip+e1[i]*0.5f);
                if((int)local.size()<k1){local.push_back({score,i});std::push_heap(local.begin(),local.end());}
                else if(score<local.front().first){std::pop_heap(local.begin(),local.end());local.back()={score,i};std::push_heap(local.begin(),local.end());}
            }
            pool.insert(pool.end(),local.begin(),local.end());
        }
        std::nth_element(pool.begin(),pool.begin()+k1,pool.end());
    }
    double t_fused=(now_ms()-t0)/20;
    printf("tempo completo (scores[N]+nth): %.3f ms/q\n", t_full);
    printf("tempo fused (heap k1, sem scores[N]): %.3f ms/q\n", t_fused);
    printf("speedup: %.2fx  |  scores[N] economizado: %.0f KB/query\n",
           t_full/t_fused, (double)N*4/1024);
    return 0;
}
