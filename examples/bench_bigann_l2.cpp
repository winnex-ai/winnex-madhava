/**
 * bench_bigann_l2.cpp — Benchmark Madhava L2 against the official BIGANN
 * L2 ground truth, including the exact-scan "ceiling" baseline.
 *
 * Usage:
 *   winnex_madhava_bench <base.u8bin> <queries.u8bin> <gt.bin> [n] [n_queries] [k1_frac]
 *
 * Output (CSV):
 *   Scale,Method,Build_s,Lat_ms,NDCG_L2,R@10_L2,Vio_Pairs,k1,k3
 *
 * This benchmark is the scientific core of the L2 investigation:
 * it reports the exact-scan ceiling alongside the Madhava result, so the
 * reader can see how close the bound+filter engine gets to the physical
 * limit of the evaluated subset.
 */
#include "winnex_madhava/winnex_madhava.hpp"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace winnex_madhava;

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <base.u8bin> <queries.u8bin> <gt.bin> [n] [nq] [k1_frac]\n", argv[0]);
        return 1;
    }
    std::string base_path = argv[1];
    std::string q_path = argv[2];
    std::string gt_path = argv[3];
    int n = argc > 4 ? atoi(argv[4]) : 100000000;
    int nq = argc > 5 ? atoi(argv[5]) : 50;
    double k1_frac = argc > 6 ? atof(argv[6]) : 0.05;
    int dim = 128, k = 10;

    // mmap the base corpus (no RAM copy).
    int fd = open(base_path.c_str(), O_RDONLY);
    if (fd < 0) { perror("open base"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t len = st.st_size;
    const uint8_t* base = (const uint8_t*)mmap(0, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    int total_vecs = (int)(len / dim);
    n = std::min(n, total_vecs);

    // Load queries (the first nq*2 float vectors; we use query 2i for GT[i]).
    FILE* fq = fopen(q_path.c_str(), "rb");
    if (!fq) { perror("open queries"); return 1; }
    std::vector<uint8_t> qbuf((size_t)nq * 2 * dim);
    fread(qbuf.data(), 1, qbuf.size(), fq); fclose(fq);
    std::vector<float> qv((size_t)nq * 2 * dim);
    for (size_t i = 0; i < qbuf.size(); i++) qv[i] = (float)qbuf[i];

    // Load GT (GT[gi] <-> query 2*gi).
    auto gt = read_bigann_groundtruth(gt_path, nq);
    if (gt.empty()) { fprintf(stderr, "GT not readable\n"); return 1; }

    Config cfg;
    cfg.dim = dim;
    cfg.k = k;
    cfg.k1_fraction = k1_frac;
    cfg.postfilter = true;

    MadhavaL2 engine(cfg);
    engine.build(base, n);

    printf("Scale,Method,Build_s,Lat_ms,NDCG_L2,R@10_L2,Vio_Pairs,k1,k3\n");

    auto eval = [&](const char* method, SearchResult (*fn)(const MadhavaL2&, const float*),
                    double* out_r, double* out_n) {
        double tr = 0, tn = 0, tlat = 0;
        long long vio = 0;
        int ak1 = 0, ak3 = 0;
        int n_eval = (int)gt.size();
        for (int gi = 0; gi < n_eval; gi++) {
            int qi = 2 * gi;
            std::vector<int> gset;
            for (int v : gt[gi]) if (v >= 0 && v < n) gset.push_back(v);
            SearchResult r = fn(engine, &qv[(size_t)qi * dim]);
            tn += ndcg_at_k(r.indices, gset, k);
            tr += recall_at_k(r.indices, gset, k);
            tlat += r.latency_ms;
            vio += r.bound_violations;
            ak1 += r.k1; ak3 += r.k3;
        }
        int m = std::max(n_eval, 1);
        printf("%d,%s,%.3f,%.3f,%.4f,%.4f,%lld,%d,%d\n",
               n, method, engine.build_seconds(), tlat / m, tn / m, tr / m, vio, ak1 / m, ak3 / m);
        if (out_r) *out_r = tr / m;
        if (out_n) *out_n = tn / m;
    };

    double r_ceiling = 0, n_ceiling = 0;
    eval("exact_scan", [](const MadhavaL2& e, const float* q) { return e.search_exact(q); }, &r_ceiling, &n_ceiling);
    double r_madhava = 0, n_madhava = 0;
    eval("madhava", [](const MadhavaL2& e, const float* q) { return e.search(q); }, &r_madhava, &n_madhava);

    fprintf(stderr, "\n");
    fprintf(stderr, "Ceiling (exact scan):  R@10=%.4f NDCG=%.4f\n", r_ceiling, n_ceiling);
    fprintf(stderr, "Madhava (bound+filter): R@10=%.4f NDCG=%.4f\n", r_madhava, n_madhava);
    fprintf(stderr, "Efficiency vs ceiling:  %.1f%%\n", r_ceiling > 0 ? 100.0 * r_madhava / r_ceiling : 0.0);
    fprintf(stderr, "GT coverage in subset:  %.1f%%\n", 100.0 * r_ceiling);
    fprintf(stderr, "\nBSL 1.1 | pay@winnex.ai\n");
    return 0;
}
