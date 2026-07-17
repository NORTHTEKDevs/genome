# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the GENOME-vs-Mem0 ACCURACY comparison yourself, with your own key.

Both systems answer the same LoCoMo questions through the SAME responder, the SAME
LLM judge, the SAME embedder, and the SAME top-k -- only the memory layer is
swapped. Then it runs the paired McNemar significance test and prints a verdict.
Parity = the accuracy difference is NOT statistically significant (p >= 0.05).

This is the "don't trust our benchmark, run it" artifact. It reuses the exact eval
harness that produced benchmarks/RESULTS.md (genome.evals.locomo + baselines), so
the numbers are same-harness comparable by construction, not a fresh harness that
could quietly favor one side.

Quick start (single OpenAI key -- responder, judge, and both embedders):
    pip install genome-memory mem0ai
    export OPENAI_API_KEY=sk-...
    # LoCoMo is CC BY-NC 4.0 (Snap Inc.); we do not redistribute it -- fetch it:
    curl -L -o benchmarks/data/locomo10.json \
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
    python benchmarks/head_to_head.py --n 2 --q 10        # small + cheap to start

Faithful to RESULTS.md (Haiku responder+judge, OpenAI embedder -- needs both keys):
    python benchmarks/head_to_head.py --provider anthropic --n 10

Offline plumbing self-test (no key, no network, no Mem0 install needed):
    python benchmarks/head_to_head.py --smoke
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

# Make the repo root importable when run as `python benchmarks/head_to_head.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genome.evals.baselines import Mem0Baseline, run_baseline_eval  # noqa: E402
from genome.evals.locomo import (  # noqa: E402
    DEFAULT_CONFIGS,
    LocomoConversation,
    LocomoQuestion,
    LocomoTurn,
    _make_metered_llm,
    _TokenMeter,
    load_locomo,
    run_locomo_eval,
)

# The four "headline" LoCoMo categories the published parity claim is scored on
# (adversarial is judged separately). Mirrors benchmarks/verdict.py:21.
HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}
_LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
_MODEL_DEFAULTS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5-20251001"}


def _rows(results) -> dict[str, dict]:
    """Flatten LocomoResult list -> {question_id: {category, judge_label}} so the
    two systems can be paired by question_id (same shape verdict.py pairs on)."""
    out: dict[str, dict] = {}
    for r in results:
        for p in r.per_question:
            out[p.question_id] = {"category": p.category, "judge_label": p.judge_label}
    return out


def _headline_acc(rows: dict[str, dict]) -> tuple[float, int]:
    hq = [q for q in rows.values() if q["category"] in HEADLINE]
    if not hq:
        return 0.0, 0
    correct = sum(1 for q in hq if q["judge_label"] == "CORRECT")
    return correct / len(hq), len(hq)


def _mcnemar(g: dict[str, dict], b: dict[str, dict]) -> tuple[int, int, int, float]:
    """Continuity-corrected paired McNemar over headline questions in BOTH systems.
    Ported from benchmarks/verdict.py:49 (the one correct implementation)."""
    ids = [i for i in g if i in b and g[i]["category"] in HEADLINE]
    b_only = sum(  # Mem0 right, GENOME wrong
        1 for i in ids
        if g[i]["judge_label"] != "CORRECT" and b[i]["judge_label"] == "CORRECT"
    )
    c_only = sum(  # GENOME right, Mem0 wrong
        1 for i in ids
        if g[i]["judge_label"] == "CORRECT" and b[i]["judge_label"] != "CORRECT"
    )
    n = b_only + c_only
    if n == 0:
        return len(ids), b_only, c_only, 1.0
    chi = (abs(b_only - c_only) - 1) ** 2 / n
    return len(ids), b_only, c_only, math.erfc(math.sqrt(chi / 2))


def _require_key(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(f"Set {name} (needed for this provider). Nothing was run.")


def _build_llms(provider: str, model: str, meter_r, meter_j):
    """Build the SHARED responder + judge callables. Both systems get the exact
    same callables, so neither can be answered or judged by a different model."""
    if provider == "openai":
        _require_key("OPENAI_API_KEY")
        from openai import OpenAI
        client = OpenAI()
    elif provider == "anthropic":
        _require_key("ANTHROPIC_API_KEY")
        _require_key("OPENAI_API_KEY")  # embedder is OpenAI on both sides (matched)
        from anthropic import Anthropic
        client = Anthropic()
    else:
        raise SystemExit(f"unknown provider {provider!r}")
    responder = _make_metered_llm(provider, client, model, 512, meter_r)
    judge = _make_metered_llm(provider, client, model, 256, meter_j)
    return responder, judge


def _report(g_results, b_results, genome_name: str) -> None:
    g, b = _rows(g_results), _rows(b_results)
    gj, gn = _headline_acc(g)
    bj, bn = _headline_acc(b)
    npair, b_only, c_only, p = _mcnemar(g, b)

    print("\n" + "=" * 66)
    print("  HEAD-TO-HEAD: accuracy on the same questions, same judge/embedder")
    print("=" * 66)
    print(f"  GENOME ({genome_name}):  {gj:.3f}   (n={gn} headline questions)")
    print(f"  Mem0:{' ' * (len(genome_name) + 10)}{bj:.3f}   (n={bn})")
    print(f"\n  paired on {npair} questions | GENOME-only-correct={c_only} "
          f"Mem0-only-correct={b_only} | McNemar p={p:.3f}")

    if p >= 0.05:
        verdict = (f"PARITY -- the {abs(gj - bj):.3f} accuracy gap is NOT statistically "
                   f"significant (p={p:.3f} >= 0.05).")
    elif gj > bj:
        verdict = f"GENOME leads significantly (p={p:.3f} < 0.05)."
    else:
        verdict = f"Mem0 leads significantly (p={p:.3f} < 0.05)."
    print(f"\n  VERDICT: {verdict}")
    if npair < 30:
        print(f"  NOTE: only {npair} paired questions -- small samples are noisy. "
              f"Run more conversations (--n 10) for a stable number.")
    print("\n  This is the ACCURACY story (we claim a tie, not a win). GENOME's edge is")
    print("  cost / air-gapped / auditable -- reproduce that with `python -m genome.verify`.")


# ---------------------------------------------------------------- offline smoke

class _StubBaseline:
    """A stand-in memory system for the offline plumbing test: implements the same
    ingest/answer/close protocol as Mem0Baseline but makes no network calls, so
    --smoke exercises the whole head-to-head pipeline with no key and no Mem0."""

    name = "baseline-mem0(stub)"

    def __init__(self, responder):
        self._responder = responder

    def ingest(self, conversation):  # noqa: ANN001
        pass

    def answer(self, question):  # noqa: ANN001
        return "Berlin.", ["(stub retrieved memory)"], 1.0

    def close(self):
        pass


def _synthetic_conversation() -> LocomoConversation:
    turns = [
        LocomoTurn(speaker="Alice", text="I moved to Berlin last month.",
                   turn_id=1, dia_id="D1:1", session=1),
        LocomoTurn(speaker="Bob", text="Nice -- how is it there?",
                   turn_id=2, dia_id="D1:2", session=1),
        LocomoTurn(speaker="Alice", text="Cold, but I like my new data-scientist job.",
                   turn_id=3, dia_id="D1:3", session=1),
    ]
    questions = [
        LocomoQuestion(question="Where did Alice move?", answer="Berlin",
                       category="single-hop", evidence=["D1:1"], question_id="smoke-1"),
        LocomoQuestion(question="What is Alice's job?", answer="data scientist",
                       category="single-hop", evidence=["D1:3"], question_id="smoke-2"),
    ]
    return LocomoConversation(
        conversation_id="smoke1", turns=turns, questions=questions,
        speakers=["Alice", "Bob"], speaker_a="Alice", speaker_b="Bob",
    )


def _run_smoke() -> int:
    print("SMOKE: offline plumbing test -- synthetic data, echo responder/judge, "
          "stub Mem0.\nNo API key, no network, nothing spent. Proves the harness runs.\n")

    def responder(_prompt: str) -> str:
        return "Berlin."

    def judge(_prompt: str) -> str:  # mem0-mode judge returns JSON {reasoning,label}
        return '{"reasoning": "smoke plumbing", "label": "CORRECT"}'

    conv = _synthetic_conversation()
    cfg = replace(  # local embedder (embed_model stays None) -> fully offline
        next(c for c in DEFAULT_CONFIGS if c.name == "genome-parent-filtered"),
        top_k=5,
    )
    g_results = run_locomo_eval([conv], [cfg], responder, judge,
                                judge_mode="mem0", workers=1)
    b_results = run_baseline_eval([conv], _StubBaseline(responder), judge,
                                  judge_mode="mem0", workers=1)
    _report(g_results, b_results, cfg.name)
    print("\nSMOKE OK: pipeline runs end-to-end (ingest -> answer -> judge -> "
          "pair -> McNemar -> verdict).")
    return 0


# ---------------------------------------------------------------- real run

def _need_dataset(dataset: str) -> int:
    print(f"LoCoMo dataset not found at {dataset}.")
    print("LoCoMo is CC BY-NC 4.0 (Snap Inc.) -- non-commercial use only; this repo")
    print("does not redistribute it. Download it yourself:\n")
    print(f"  curl -L -o {dataset} \\\n    {_LOCOMO_URL}\n")
    print("Then re-run. Or try the offline plumbing test:  "
          "python benchmarks/head_to_head.py --smoke")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="benchmarks/data/locomo10.json")
    ap.add_argument("--n", type=int, default=2, help="conversations to run (max 10)")
    ap.add_argument("--q", type=int, default=None, help="cap questions per conversation")
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", default=None, help="override responder+judge model")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", default="genome-parent-filtered",
                    help="GENOME config from genome.evals.locomo.DEFAULT_CONFIGS")
    ap.add_argument("--smoke", action="store_true",
                    help="offline plumbing test: no key, no network, no Mem0")
    args = ap.parse_args()

    if args.smoke:
        return _run_smoke()

    if not Path(args.dataset).exists():
        return _need_dataset(args.dataset)

    model = args.model or _MODEL_DEFAULTS[args.provider]
    print("This run makes REAL API calls (responder + judge on both systems, plus")
    print(f"Mem0's own fact-extraction LLM). Provider={args.provider} model={model}.")
    print("Start small (--n 1 --q 5) to sanity-check spend before a full run.\n")

    conversations = load_locomo(args.dataset)[: args.n]
    if args.q:
        for c in conversations:
            c.questions = c.questions[: args.q]
    nq = sum(len(c.questions) for c in conversations)
    print(f"Loaded {len(conversations)} conversations, {nq} questions "
          f"from {args.dataset}.\n")

    meter_r, meter_j = _TokenMeter(), _TokenMeter()
    responder, judge = _build_llms(args.provider, model, meter_r, meter_j)

    genome_cfg = next((c for c in DEFAULT_CONFIGS if c.name == args.config), None)
    if genome_cfg is None:
        raise SystemExit(
            f"config {args.config!r} not found. Available: "
            f"{[c.name for c in DEFAULT_CONFIGS]}"
        )
    # Match the embedder to Mem0's (OpenAI text-embedding-3-small) so retrieval
    # quality is identical and only the memory ARCHITECTURE differs.
    genome_cfg = replace(genome_cfg, embed_model="openai:text-embedding-3-small",
                         top_k=args.top_k)

    print(f"Running GENOME ({genome_cfg.name})...")
    g_results = run_locomo_eval(conversations, [genome_cfg], responder, judge,
                                judge_mode="mem0", workers=args.workers,
                                progress=lambda m: print(f"  {m}"))

    print("Running Mem0...")
    mem0 = Mem0Baseline(responder, top_k=args.top_k, llm_model=model,
                        embed_model="text-embedding-3-small",
                        llm_provider=args.provider)
    b_results = run_baseline_eval(conversations, mem0, judge, judge_mode="mem0",
                                  workers=args.workers,
                                  progress=lambda m: print(f"  {m}"))
    mem0.close()

    _report(g_results, b_results, genome_cfg.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
