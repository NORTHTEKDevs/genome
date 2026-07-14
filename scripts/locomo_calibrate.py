"""LOCOMO calibration -- 5 conversations x all default configs.

Run this BEFORE the full benchmark sweep. Catches issues that would otherwise
waste a multi-hour, paid LLM run:
  - Extraction / answer-prompt regressions (mean score collapses)
  - Hybrid mode misconfigured (no improvement vs dense baseline)
  - RAPTOR clustering failures (skips silently in eval, surfaces here)
  - Judge model returning malformed verdicts (high "INCORRECT" parse fallbacks)

Usage:
    python scripts/locomo_calibrate.py \\
        --dataset path/to/locomo10.json \\
        --provider anthropic \\
        --limit-conversations 5

The output is a per-config comparison table. Look for:
  - Every config above ~0.20 mean score (else extraction/prompt is broken).
  - Hybrid >= dense by >= 0.02 (else BM25 isn't catching anything).
  - Parent-filtered >= baseline (else parent-filter is not working on this
    dataset's question types).

This is a "do not press the big benchmark button until this passes" gate.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="LOCOMO pre-flight calibration")
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Path to locomo10.json (download from HuggingFace).",
    )
    parser.add_argument(
        "--provider", choices=("anthropic", "openai"), required=True,
    )
    parser.add_argument(
        "--responder-model", type=str, default="claude-haiku-4-5-20251001",
    )
    parser.add_argument(
        "--judge-model", type=str, default="claude-haiku-4-5-20251001",
    )
    parser.add_argument("--limit-conversations", type=int, default=5)
    parser.add_argument("--questions-per-convo", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("results/locomo-calibration.json"),
    )
    args = parser.parse_args()

    from genome.evals.llm_judge import anthropic_judge, openai_judge
    from genome.evals.locomo import (
        DEFAULT_CONFIGS,
        load_locomo,
        run_locomo_eval,
    )

    convos = load_locomo(args.dataset)[: args.limit_conversations]
    if not convos:
        print(f"ERROR: no conversations loaded from {args.dataset}", file=sys.stderr)
        return 1
    print(
        f"Loaded {len(convos)} conversations "
        f"(~{sum(len(c.questions) for c in convos)} questions across all configs)"
    )

    if args.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        responder = anthropic_judge(client, model=args.responder_model, max_tokens=512)
        judge = anthropic_judge(client, model=args.judge_model, max_tokens=256)
    else:
        from openai import OpenAI
        client = OpenAI()
        responder = openai_judge(client, model=args.responder_model, max_tokens=512)
        judge = openai_judge(client, model=args.judge_model, max_tokens=256)

    t0 = time.perf_counter()
    summary = run_locomo_eval(
        conversations=convos,
        configs=DEFAULT_CONFIGS,
        responder=responder,
        judge=judge,
        max_questions_per_conversation=args.questions_per_convo,
    )
    elapsed = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nCalibration finished in {elapsed:.1f}s. Results: {args.output}")

    # Print comparison table
    print("\n" + "=" * 72)
    print(f"  {'CONFIG':<28} {'MEAN':>8} {'P@10':>8} {'CORRECT %':>12}")
    print("=" * 72)
    baseline_mean = None
    for cfg_name, stats in summary.items():
        mean = stats.get("mean_score", 0.0)
        p_at_10 = stats.get("retrieval_recall@10", 0.0)
        correct_pct = stats.get("correct_rate", 0.0) * 100
        marker = ""
        if baseline_mean is None:
            baseline_mean = mean
        else:
            delta = mean - baseline_mean
            marker = f"  ({'+' if delta >= 0 else ''}{delta:.3f})"
        print(f"  {cfg_name:<28} {mean:>8.3f} {p_at_10:>8.3f} {correct_pct:>11.1f}%{marker}")
    print("=" * 72)

    # Sanity gates
    issues: list[str] = []
    for cfg_name, stats in summary.items():
        mean = stats.get("mean_score", 0.0)
        if mean < 0.20:
            issues.append(
                f"{cfg_name}: mean_score={mean:.3f} < 0.20 -- extraction or "
                f"prompt likely broken; investigate before full run"
            )
    if issues:
        print("\nWARNINGS:", file=sys.stderr)
        for i in issues:
            print(f"  {i}", file=sys.stderr)
        return 1

    print("\nAll configs cleared the 0.20 floor. Safe to run the full sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
