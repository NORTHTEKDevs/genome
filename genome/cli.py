"""Command-line runner for GENOME evaluation."""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genome.evaluate import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GENOME recombination evaluation")
    parser.add_argument("--output", "-o", type=Path, default=Path("results/evaluation.json"))
    def _positive_int(s: str) -> int:
        v = int(s)
        if v <= 0:
            raise argparse.ArgumentTypeError(
                f"--limit-pairs must be a positive integer, got {v}"
            )
        return v

    parser.add_argument("--limit-pairs", type=_positive_int, default=None)
    parser.add_argument(
        "--no-filter-parents",
        action="store_true",
        help="Disable parent filtering (parents of the hybrid stay in top-k).",
    )
    args = parser.parse_args()

    print("Running GENOME evaluation...")
    results = run_evaluation(
        limit_pairs=args.limit_pairs,
        filter_parents=not args.no_filter_parents,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")

    print("\n=== Summary (parents filtered) ===")
    print(f"{'operator':40s} {'hit@1':>7s} {'hit@3':>7s} {'hit@5':>7s} {'p@3':>7s}")
    ranked = sorted(results.items(), key=lambda kv: kv[1].get("hit@3", 0.0), reverse=True)
    for name, metrics in ranked:
        print(
            f"  {name:38s} "
            f"{metrics.get('hit@1', 0.0):>7.3f} "
            f"{metrics.get('hit@3', 0.0):>7.3f} "
            f"{metrics.get('hit@5', 0.0):>7.3f} "
            f"{metrics.get('precision@3', 0.0):>7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
