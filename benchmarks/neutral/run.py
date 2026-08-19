# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Neutral multi-system memory benchmark: same questions, same judge, same rules.

Vendor-run memory benchmarks have a credibility problem: every vendor reports
numbers from a harness they designed, and independent reruns have disagreed with
self-reported scores by tens of points. This harness is the opposite bet: run N
systems under identical conditions - the SAME responder model, the SAME judge,
the SAME embedder tier, the SAME question slice - print per-system accuracy, a
pairwise continuity-corrected McNemar matrix, and a full-disclosure block, and
let anyone rerun it with their own key.

GENOME is one row in the table, not the house. When it loses a pairing, the
harness says so in the same font.

    # offline plumbing test - no key, no network:
    python benchmarks/neutral/run.py --smoke

    # one OpenRouter key, local embedders, two conversations:
    export OPENROUTER_API_KEY=sk-or-...
    python benchmarks/neutral/run.py --provider openrouter --n 2 --q 10 \
        --systems genome,mem0,fullcontext

Adding a system: benchmarks/neutral/adapters.py (~40 lines) + ADAPTERS.md.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from adapters import make_system  # noqa: E402
from head_to_head import (  # noqa: E402
    _MODEL_DEFAULTS,
    _build_llms,
    _headline_acc,
    _mcnemar,
    _need_dataset,
    _rows,
    _StubBaseline,
    _synthetic_conversation,
)

from genome.evals.baselines import run_baseline_eval  # noqa: E402
from genome.evals.locomo import (  # noqa: E402
    DEFAULT_CONFIGS,
    _TokenMeter,
    load_locomo,
    run_locomo_eval,
)


def _run_genome(conversations, responder, judge, *, provider, top_k, workers):
    cfg = next(c for c in DEFAULT_CONFIGS if c.name == "genome-parent-filtered")
    if provider == "openrouter":
        cfg = replace(cfg, embed_model=None, top_k=top_k)
    elif provider != "offline":
        cfg = replace(cfg, embed_model="openai:text-embedding-3-small", top_k=top_k)
    else:
        cfg = replace(cfg, top_k=top_k)
    results = run_locomo_eval(
        conversations, [cfg], responder, judge, judge_mode="mem0", workers=workers,
        progress=lambda m: print(f"    {m}"),
    )
    return _rows(results)


def _pairwise_matrix(rows_by_system: dict[str, dict]) -> list[str]:
    names = list(rows_by_system)
    lines = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            npair, b_only, a_only, p = _mcnemar(rows_by_system[a], rows_by_system[b])
            verdict = (
                "parity" if p >= 0.05
                else f"{a} leads" if a_only > b_only
                else f"{b} leads"
            )
            lines.append(
                f"  {a} vs {b}: paired n={npair}, {a}-only={a_only}, "
                f"{b}-only={b_only}, McNemar p={p:.3f} -> {verdict}"
            )
    return lines


def _report(rows_by_system: dict[str, dict], disclosure: dict[str, str]) -> None:
    print("\n" + "=" * 70)
    print("  NEUTRAL HARNESS: same questions, same judge, same rules")
    print("=" * 70)
    for name, rows in rows_by_system.items():
        acc, n = _headline_acc(rows)
        print(f"  {name:<14} accuracy {acc:.3f}   (n={n} headline questions)")
    print("\n  Pairwise McNemar (continuity-corrected):")
    for line in _pairwise_matrix(rows_by_system):
        print(line)
    print("\n  Disclosure:")
    for key, value in disclosure.items():
        print(f"    {key}: {value}")
    print(
        "\n  Read this before quoting it: small n is noisy; a 'leads' verdict on "
        "<30 paired\n  questions is weather, not climate. Rerun with your own key "
        "and more conversations."
    )


def _run_smoke() -> int:
    print("SMOKE: offline plumbing - synthetic data, echo responder/judge, stub "
          "systems.\nNo key, no network, nothing spent.\n")
    conv = _synthetic_conversation()

    def responder(_prompt: str) -> str:
        return "Berlin."

    def judge(_prompt: str) -> str:
        return '{"reasoning": "smoke", "label": "CORRECT"}'

    rows_by_system = {
        "genome": _run_genome(
            [conv], responder, judge, provider="offline", top_k=5, workers=1
        ),
        "stub-a": _rows(run_baseline_eval(
            [conv], _StubBaseline(responder), judge, judge_mode="mem0", workers=1
        )),
        "stub-b": _rows(run_baseline_eval(
            [conv], _StubBaseline(responder), judge, judge_mode="mem0", workers=1
        )),
    }
    _report(rows_by_system, {
        "mode": "smoke (offline)", "dataset": "synthetic 1 conversation",
        "responder/judge": "echo stubs",
    })
    print("\nSMOKE OK: N-system pipeline runs end-to-end.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="benchmarks/data/locomo10.json")
    ap.add_argument("--systems", default="genome,mem0",
                    help="comma-separated: genome, mem0, fullcontext (+ your adapters)")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--provider", choices=["openai", "anthropic", "openrouter"],
                    default="openrouter")
    ap.add_argument("--model", default=None)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        return _run_smoke()
    if not Path(args.dataset).exists():
        return _need_dataset(args.dataset)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    model = args.model or _MODEL_DEFAULTS[args.provider]
    conversations = load_locomo(args.dataset)[: args.n]
    if args.q:
        for c in conversations:
            c.questions = c.questions[: args.q]
    nq = sum(len(c.questions) for c in conversations)
    print(f"Systems: {', '.join(systems)} | {len(conversations)} conversations, "
          f"{nq} questions | provider={args.provider} model={model}")
    print("This makes REAL API calls for every system. Start with --n 1 --q 5.\n")

    meter_r, meter_j = _TokenMeter(), _TokenMeter()
    responder, judge = _build_llms(args.provider, model, meter_r, meter_j)

    rows_by_system: dict[str, dict] = {}
    for name in systems:
        print(f"Running {name}...")
        if name == "genome":
            rows_by_system[name] = _run_genome(
                conversations, responder, judge,
                provider=args.provider, top_k=args.top_k, workers=args.workers,
            )
            continue
        system = make_system(
            name, responder, top_k=args.top_k, provider=args.provider, model=model
        )
        try:
            rows_by_system[name] = _rows(run_baseline_eval(
                conversations, system, judge, judge_mode="mem0",
                workers=args.workers, progress=lambda m: print(f"    {m}"),
            ))
        finally:
            system.close()

    _report(rows_by_system, {
        "dataset": f"{args.dataset} (first {len(conversations)} conversations"
                   + (f", {args.q} questions each" if args.q else "") + ")",
        "responder model": model,
        "judge model": model + " (same instance as responder)",
        "embedder": "local all-MiniLM-L6-v2 on every system"
        if args.provider == "openrouter"
        else "openai:text-embedding-3-small on every system",
        "top_k": str(args.top_k),
        "provider": args.provider,
        "responder tokens": str(meter_r.total()) if hasattr(meter_r, "total") else "-",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
