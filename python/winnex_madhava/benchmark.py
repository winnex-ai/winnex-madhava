"""
winnex_madhava.benchmark — run the BIGANN L2 benchmark from the command line.

    python -m winnex_madhava.benchmark --n 10000000 --nq 50

Reports the exact-scan ceiling and the winnex-madhava result side by side.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m winnex_madhava.benchmark",
        description="BIGANN L2 benchmark: exact-scan ceiling vs winnex-madhava.",
    )
    parser.add_argument("--base", default="bigann_data/base.u8bin")
    parser.add_argument("--queries", default="bigann_data/unif_query_10k.u8bin")
    parser.add_argument("--gt", default="bigann_data/unif_groundtruth_10k.bin")
    parser.add_argument("--n", type=int, default=10_000_000, help="corpus size to index")
    parser.add_argument("--nq", type=int, default=50, help="number of GT queries to evaluate")
    parser.add_argument("--k1", type=float, default=0.05, help="Stage-1 keep fraction")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args(argv)

    import numpy as np
    import winnex_madhava

    # mmap the base corpus (no RAM copy).
    with open(args.base, "rb") as f:
        arr = np.memmap(f, dtype=np.uint8, mode="r", shape=(args.n, args.dim))

    engine = winnex_madhava.build_engine(
        arr, dim=args.dim, k=10, k1_fraction=args.k1, postfilter=True
    )
    print(f"indexed {engine.num_vectors()} x {engine.dim()}D in {engine.build_seconds():.2f}s")

    # Queries: GT[gi] <-> query 2*gi (BIGANN sampling).
    qbuf = np.fromfile(args.queries, dtype=np.uint8, count=args.nq * 2 * args.dim)
    q = qbuf.reshape(-1, args.dim).astype(np.float32)

    gt = winnex_madhava.read_bigann_groundtruth(args.gt, args.nq)

    def eval_method(name: str, fn) -> dict:
        tr = tn = tlat = 0.0
        for gi in range(len(gt)):
            res = fn(q[2 * gi])
            gset = [v for v in gt[gi] if 0 <= v < engine.num_vectors()]
            # recall_at_k / ndcg_at_k normalizam por min(k, |gset|) internamente
            # (definição robusta), então passamos o conjunto completo de relevantes
            # no subset — um scan exato perfeito sempre atinge 1.0.
            tr += winnex_madhava.recall_at_k(res.indices, gset, 10)
            tn += winnex_madhava.ndcg_at_k(res.indices, gset, 10)
            tlat += res.latency_ms
        m = max(len(gt), 1)
        return {"R@10": tr / m, "NDCG": tn / m, "lat_ms": tlat / m, "name": name}

    ceiling = eval_method("exact_scan", engine.search_exact)
    madhava = eval_method("winnex-madhava", engine.search)

    print(f"\n{'method':<12} {'R@10':>7} {'NDCG':>7} {'lat_ms':>8}")
    for r in (ceiling, madhava):
        print(f"{r['name']:<12} {r['R@10']:>7.4f} {r['NDCG']:>7.4f} {r['lat_ms']:>8.1f}")

    eff = 100.0 * madhava["R@10"] / ceiling["R@10"] if ceiling["R@10"] else 0.0
    print(f"\nEfficiency vs ceiling: {eff:.1f}%")
    print(f"GT coverage in subset: {100.0 * ceiling['R@10']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
