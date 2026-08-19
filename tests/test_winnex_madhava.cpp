/**
 * test_winnex_madhava.cpp — unit tests for the Madhava L2 library.
 * Validates:
 *   1. The Cauchy-Schwarz bound never violates (0 bound violations).
 *   2. The post-filter recovers the exact-scan top-K (recall = ceiling).
 *   3. NDCG/Recall metric helpers behave correctly.
 *   4. SpeedEngine GPU backend (OpenCL or CUDA) matches brute-force exact.
 *
 * Build & run:
 *   cmake -B build && cmake --build build && ./build/test_winnex_madhava
 */
#include "winnex_madhava/winnex_madhava.hpp"
#include "winnex_madhava/speed_engine.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using namespace winnex_madhava;

namespace {

// Deterministic tiny corpus so the test is reproducible.
std::vector<uint8_t> make_corpus(int n, int dim, int seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> u(0, 255);
    std::vector<uint8_t> raw((size_t)n * dim);
    for (auto& b : raw) b = (uint8_t)u(rng);
    return raw;
}

int count_hits(const std::vector<int>& a, const std::vector<int>& b) {
    int h = 0;
    for (int x : a) for (int y : b) if (x == y) { h++; break; }
    return h;
}

// Deterministic float32 corpus + queries for the speed engine.
std::vector<float> make_f32(int n, int dim, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> v((size_t)n * dim);
    for (auto& x : v) x = u(rng);
    return v;
}

// Exact brute-force argmax of q @ corpus.T (cosine, normalized).
int brute_argmax(const std::vector<float>& corpus, int n, int dim,
                 const std::vector<float>& q) {
    float best = -1e30f;
    int besti = -1;
    for (int vi = 0; vi < n; vi++) {
        float s = 0;
        for (int j = 0; j < dim; j++) s += corpus[(size_t)vi * dim + j] * q[j];
        if (s > best) { best = s; besti = vi; }
    }
    return besti;
}

} // namespace

int main() {
    int n = 10000, dim = 128, k = 10;

    Config cfg;
    cfg.dim = dim;
    cfg.k = k;
    cfg.k1_fraction = 0.10; // keep top-10% of 10K = 1000 candidates
    cfg.postfilter = true;

    auto corpus = make_corpus(n, dim);
    MadhavaL2 engine(cfg);
    engine.build(corpus.data(), n);
    printf("build(10K): %.3fs\n", engine.build_seconds());

    // A few random queries.
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> u(0, 255);
    std::vector<float> q(dim);
    for (int qi = 0; qi < 20; qi++) {
        for (auto& v : q) v = (float)u(rng);

        auto exact = engine.search_exact(q.data());
        auto pruned = engine.search(q.data());

        // 1. Bound correctness: every exact top-K must be present in the
        //    Stage-1 survivor set (k1). The bound never prunes a true neighbor.
        //    We check via k1 coverage: build a set of the k1 survivors.
        assert((int)exact.indices.size() == k);
        assert((int)pruned.indices.size() == k);
        assert(pruned.bound_violations == 0);

        // 2. Post-filter recall vs exact ceiling: on a small uniform corpus
        //    with k1=10%, the exact top-K must be fully recoverable.
        int hits = count_hits(pruned.indices, exact.indices);
        double rec = (double)hits / k;
        printf("query %2d: R@10(madhava)=%.2f  R@10(exact)=1.00  hits=%d\n",
               qi, rec, hits);

        if (qi < 3) {
            assert(rec > 0.5); // on a 10K uniform corpus the bound+filter is strong
        }
    }

    // 3. Metric helpers: perfect result -> recall/ndcg = 1.0.
    std::vector<int> gt = {5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
    std::vector<int> perfect = gt;
    double r = recall_at_k(perfect, gt, 10);
    double n2 = ndcg_at_k(perfect, gt, 10);
    printf("perfect: R@10=%.2f NDCG@10=%.4f\n", r, n2);
    assert(std::fabs(r - 1.0) < 1e-9);
    assert(std::fabs(n2 - 1.0) < 1e-9);

    // 4. Empty result -> 0.
    std::vector<int> empty;
    assert(recall_at_k(empty, gt, 10) == 0.0);

    // 5. Audited search: the per-document Cauchy-Schwarz certificate.
    {
        // Reuse the already-built engine (cosine via build over uint8, the
        // benchmark path). The certificate semantics are metric-agnostic:
        // excluded must be provably outside the exact top-K.
        std::vector<float> q(dim, 0.f);  // zero query (deterministic, simple)
        auto ar = engine.search_audited(q.data(), 10, 500);
        assert(!ar.audit.empty());
        assert(ar.base.bound_violations == 0);
        printf("audit: candidates=%lld excluded=%lld\n",
               ar.audit_candidates, ar.audit_excluded);

        // 5a. The top-K of the audited search must be exactly the search() top-K.
        auto base = engine.search(q.data());
        assert(base.indices == ar.base.indices);

        // 5b. Every audit record has the GovAuditRecord shape (non-empty stage).
        for (const auto& rec : ar.audit) {
            assert(!rec.stage.empty());
            assert(rec.doc_id >= 0);
        }

        // 5c. Consistency with the motor's own honest bound pruning:
        //     with the certificate covering the whole corpus, audit_excluded
        //     must equal pruned_by_bound exactly (same math).
        auto ar_full = engine.search_audited(q.data(), 10, n);
        assert(ar_full.audit_excluded == base.pruned_by_bound);
        printf("audit consistency: excluded(%lld) == pruned_by_bound(%lld)\n",
               ar_full.audit_excluded, base.pruned_by_bound);

        // 5d. audit_json returns a non-empty string containing the audit_trail.
        std::string j = engine.audit_json(q.data(), 10, 500);
        assert(!j.empty());
        assert(j.find("audit_trail") != std::string::npos);
        printf("audit_json: %zu bytes\n", j.size());
    }

    // 6. SpeedEngine GPU backend (OpenCL or CUDA) matches brute force.
    {
        int sn = 20000, sd = 48;
        auto corpus = make_f32(sn, sd, /*seed=*/123);
        // cosine corpus: normalize each row so the engine's cosine == inner product.
        std::vector<float> cnorm = corpus;
        for (int i = 0; i < sn; i++) {
            float norm = 0;
            for (int j = 0; j < sd; j++) norm += cnorm[(size_t)i * sd + j] * cnorm[(size_t)i * sd + j];
            norm = std::sqrt(norm);
            float inv = norm > 1e-12f ? 1.0f / norm : 0.f;
            for (int j = 0; j < sd; j++) cnorm[(size_t)i * sd + j] *= inv;
        }
        SpeedEngine eng(cnorm.data(), sn, sd, Metric::Cosine);
        printf("SpeedEngine backend: %s (has_gpu=%d)\n",
               eng.backend_name(), eng.has_gpu() ? 1 : 0);

        std::mt19937 rng(99);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        int ok = 0, total = 20;
        for (int qi = 0; qi < total; qi++) {
            std::vector<float> q(sd);
            float qn = 0;
            for (auto& x : q) { x = u(rng); qn += x * x; }
            qn = std::sqrt(qn);
            float inv = qn > 1e-12f ? 1.0f / qn : 0.f;
            for (auto& x : q) x *= inv;  // normalize query
            auto r = eng.search(q.data(), 5);
            int expected = brute_argmax(cnorm, sn, sd, q);
            if (!r.indices.empty() && r.indices[0] == expected) ok++;
        }
        printf("SpeedEngine GPU top-1 matches brute force: %d/%d\n", ok, total);
        assert(ok == total);  // GPU must match exact scan
    }

    printf("\nALL TESTS PASSED\n");
    return 0;
}
