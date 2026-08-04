/**
 * example_minimal.cpp — minimal usage of the Madhava L2 library.
 *
 * Build: cmake -B build && cmake --build build
 * Run:   ./build/example_minimal
 */
#include "winnex_madhava/winnex_madhava.hpp"

#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using namespace winnex_madhava;

int main() {
    const int n = 50000, dim = 128, k = 10;

    // Tiny synthetic corpus.
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> u(0, 255);
    std::vector<uint8_t> raw((size_t)n * dim);
    for (auto& b : raw) b = (uint8_t)u(rng);

    Config cfg;
    cfg.dim = dim;
    cfg.k = k;
    cfg.k1_fraction = 0.10;
    cfg.postfilter = true;

    MadhavaL2 engine(cfg);
    engine.build(raw.data(), n);
    printf("Built %d vectors (%dD) in %.3fs\n", n, dim, engine.build_seconds());

    // A query = the first raw vector (should find itself as top-1).
    std::vector<float> q(dim);
    for (int j = 0; j < dim; j++) q[j] = (float)raw[j];

    auto res = engine.search(q.data());
    printf("Top-%d by Madhava (bound+filter):\n", k);
    for (int i = 0; i < (int)res.indices.size(); i++) {
        float l2 = l2_sq(raw.data() + (size_t)res.indices[i] * dim, q.data(), dim);
        printf("  %2d. id=%7d  L2²=%.0f\n", i, res.indices[i], l2);
    }
    printf("  (Stage-1 kept k1=%d candidates, latency %.2fms)\n", res.k1, res.latency_ms);
    printf("  Bound violations: %lld\n", res.bound_violations);

    auto exact = engine.search_exact(q.data());
    printf("\nTop-1 exact scan: id=%d (should equal the query itself)\n", exact.indices[0]);
    return 0;
}
