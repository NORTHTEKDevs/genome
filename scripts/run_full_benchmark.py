"""Run the full professional benchmark: aggregate + diversity + multi-encoder.

Usage:
    python scripts/run_full_benchmark.py [--encoders bge-small,mpnet] [--n-seeds 5]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from genome.benchmark import (
    benchmark_diversity,
    benchmark_encoders,
    benchmark_operators,
    print_benchmark_table,
    save_benchmark,
)
from genome.embeddings import EmbeddingProvider

DEFAULT_ENCODERS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", type=str, default=None,
                        help="Comma-separated encoder names. Defaults to 3 encoders.")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds per stochastic operator for diversity.")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap samples for CIs.")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-multi-encoder", action="store_true",
                        help="Skip the multi-encoder benchmark (slow).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    encoder_list = (
        [e.strip() for e in args.encoders.split(",") if e.strip()]
        if args.encoders else DEFAULT_ENCODERS
    )

    print("=" * 70)
    print("GENOME Professional Benchmark")
    print("=" * 70)

    # 1. Aggregate benchmark with primary encoder
    print("\n[1/3] Aggregate retrieval benchmark (MiniLM, 100 pairs, bootstrap CIs)")
    t0 = time.time()
    provider = EmbeddingProvider(model_name=encoder_list[0])
    agg = benchmark_operators(provider=provider, n_boot=args.n_boot)
    print(f"  runtime: {time.time() - t0:.1f}s")
    save_benchmark(agg, args.out_dir / "benchmark_aggregate.json")
    print_benchmark_table(agg, metric="hit@1")
    print_benchmark_table(agg, metric="hit@3")
    print_benchmark_table(agg, metric="hit@5")

    # 2. Diversity benchmark (Criterion 2)
    print(f"\n[2/3] Diversity benchmark (n_seeds={args.n_seeds}, Criterion 2)")
    t0 = time.time()
    div = benchmark_diversity(provider=provider, n_seeds=args.n_seeds, k=5)
    print(f"  runtime: {time.time() - t0:.1f}s")
    save_benchmark(div, args.out_dir / "benchmark_diversity.json")
    print("\nDiversity per stochastic operator:")
    print(f"  {'operator':36s} {'mean overlap':>14s} {'frac meaningfully diff':>25s}")
    for name, d in sorted(div.items(), key=lambda kv: kv[1]['fraction_meaningfully_different'], reverse=True):
        print(
            f"  {name:36s} {d['mean_overlap']:>14.3f} "
            f"{d['fraction_meaningfully_different']:>25.3f}"
        )

    # 3. Multi-encoder benchmark (design doc Risk 4)
    if args.skip_multi_encoder:
        print("\n[3/3] Multi-encoder: SKIPPED")
    else:
        print(f"\n[3/3] Multi-encoder generalization ({len(encoder_list)} encoders)")
        t0 = time.time()
        multi = benchmark_encoders(model_names=encoder_list)
        print(f"  runtime: {time.time() - t0:.1f}s")
        save_benchmark(multi, args.out_dir / "benchmark_encoders.json")
        print("\nhit@3 across encoders (best operator per encoder):")
        for enc, ops in multi.items():
            best = max(ops.items(), key=lambda kv: kv[1]['hit@3'].mean)
            print(f"  {enc:50s} best={best[0]:30s} hit@3={best[1]['hit@3'].mean:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("DESIGN DOC CRITERIA")
    print("=" * 70)
    # Criterion 1 check
    from genome.corpus import build_default_corpus
    from genome.dataset import load_parent_pairs
    from genome.operators import OPERATORS as _OPS
    corpus_primary = build_default_corpus(provider=provider)
    pair_001 = load_parent_pairs()[0]
    a = provider.encode(pair_001.parent_a)
    b = provider.encode(pair_001.parent_b)
    exp_lc = {e.lower() for e in pair_001.expected_hybrids}
    par_lc = {pair_001.parent_a.lower(), pair_001.parent_b.lower()}
    c1_ops_passing = []
    for name, op in _OPS.items():
        h = op(a, b)
        res = corpus_primary.search(h, k=20)
        filt = [r for r in res if r.text.lower() not in par_lc][:5]
        if any(r.text.lower() in exp_lc for r in filt):
            c1_ops_passing.append(name)
    print(f"Criterion 1 (ML eng + PM -> tech/AI PM in top-5): "
          f"{'PASS' if c1_ops_passing else 'FAIL'} "
          f"({len(c1_ops_passing)}/{len(_OPS)} operators)")

    best_op_hit3 = max(agg.items(), key=lambda kv: kv[1]['hit@3'].mean)
    avg_hit3 = agg.get('simple_average', {}).get('hit@3')
    concat_hit3 = agg.get('concat_project', {}).get('hit@3')
    print(
        f"Criterion 3 (best op hit@3 >= 60%): "
        f"{'PASS' if best_op_hit3[1]['hit@3'].mean >= 0.60 else 'FAIL'} "
        f"({best_op_hit3[0]} = {best_op_hit3[1]['hit@3'].mean:.3f})"
    )
    if avg_hit3:
        print(
            f"  (design doc assumed averaging < 40% as contrast; actual = "
            f"{avg_hit3.mean:.3f} -- averaging is competitive for aggregate retrieval)"
        )
    if concat_hit3:
        print(
            f"Criterion 3b (concat hit@3 < 20%): "
            f"{'PASS' if concat_hit3.mean < 0.20 else 'FAIL'} "
            f"(concat_project = {concat_hit3.mean:.3f})"
        )

    if any(d['fraction_meaningfully_different'] >= 0.50 for d in div.values()):
        top = max(div.items(), key=lambda kv: kv[1]['fraction_meaningfully_different'])
        print(
            f"Criterion 2 (stochastic op produces meaningfully different hybrids "
            f">=50% of pairs): PASS ({top[0]} = {top[1]['fraction_meaningfully_different']:.3f})"
        )
    else:
        print(
            "Criterion 2 (diversity >=50% meaningfully different): FAIL "
            "(no operator meets threshold)"
        )

    print("\nAll results saved to:", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
