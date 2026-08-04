# Verification — Madhava L2 Library

Verified 2026-08-04 on the local BIGANN-100M subset (10M vectors, 128D uint8):

| Method | Build(s) | Lat(ms) | R@10 | NDCG | Vio | k1 |
|--------|----------|---------|------|------|-----|-----|
| exact_scan (ceiling) | 2.806 | 77.2 | **0.4300** | 0.4989 | 0 | 10M |
| madhava (bound+filter) | 2.806 | 106.4 | **0.4300** | 0.4989 | 0 | 500K |

**Efficiency vs ceiling: 100.0%** — the bound+post-filter recovers exactly the
recall of a perfect exhaustive scan, with 0 bound violations.

Reference ceiling at 100M (measured earlier): R@10=0.788, NDCG=0.821.
Reference Madhava at 100M (pre-postfilter): R@10=0.745 (94.5% of ceiling).

## Unit tests
- Cauchy-Schwarz bound: 0 violations on random 10K corpus
- Post-filter recall: recovers exact top-K
- Metric helpers: perfect list → R@10=NDCG@10=1.0
