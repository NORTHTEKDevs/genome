"""(b) Efficiency + (c) length-scaling, measured from existing v2 data.

Efficiency: real context tokens/query fed to the answerer -- GENOME (top-30
retrieved, from checkpoints) vs full-context (whole transcript). tiktoken counts.
Scaling: per-conversation J for GENOME vs full-context against conversation
token size -- does GENOME's relative position improve as the transcript grows
past what fits comfortably in context? That is the memory-layer thesis.

Run: .venv/Scripts/python.exe benchmarks/efficiency_scaling.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import tiktoken

from genome.evals.baselines import _format_turn_text
from genome.evals.locomo import _sanitize_locomo_text, load_locomo

ENC = tiktoken.get_encoding("cl100k_base")  # standard proxy for token counts
DIRP = "results/locomo_claude_v2"
HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}


def ntok(s: str) -> int:
    return len(ENC.encode(s))


def per_conv(system):
    """{conv_id: [correct, n, sum_ctx_tokens]}"""
    out = {}
    for f in glob.glob(f"{DIRP}/checkpoints/{system}__*.json"):
        conv = os.path.basename(f).split("__")[1].replace(".json", "")
        d = json.load(open(f))
        c = n = ctx = 0
        for q in d["per_question"]:
            if q["category"] not in HEADLINE:
                continue
            n += 1
            c += q["judge_label"] == "CORRECT"
            rc = q.get("retrieved_contents") or []
            ctx += ntok("\n".join(f"- {x}" for x in rc))
        out[conv] = [c, n, ctx]
    return out


def main():
    convs = {c.conversation_id: c for c in load_locomo("benchmarks/data/locomo10.json")}
    # full-context transcript tokens per conversation (what it feeds every query)
    fc_ctx = {}
    for cid, c in convs.items():
        transcript = "\n".join(
            f"- {_sanitize_locomo_text(_format_turn_text(t))}" for t in c.turns
        )
        fc_ctx[cid] = ntok(transcript)

    gb = per_conv("genome-baseline")
    fc = per_conv("baseline-full-context")

    print("=== (b) EFFICIENCY: context tokens fed to the answerer per query ===")
    # GENOME per-query ctx = mean over questions
    g_ctx_total = sum(v[2] for v in gb.values())
    g_n = sum(v[1] for v in gb.values())
    g_j = sum(v[0] for v in gb.values()) / g_n
    g_ctx_per_q = g_ctx_total / g_n
    # full-context per-query ctx = transcript tokens, weighted by questions/conv
    fc_ctx_total = sum(fc_ctx[c] * fc[c][1] for c in fc)
    fc_n = sum(v[1] for v in fc.values())
    fc_j = sum(v[0] for v in fc.values()) / fc_n
    fc_ctx_per_q = fc_ctx_total / fc_n
    print(f"  genome-baseline : J={g_j:.3f}  ctx/query = {g_ctx_per_q:8.0f} tokens")
    print(f"  full-context    : J={fc_j:.3f}  ctx/query = {fc_ctx_per_q:8.0f} tokens")
    print(f"  --> GENOME uses {g_ctx_per_q/fc_ctx_per_q*100:.1f}% of full-context's "
          f"tokens ({fc_ctx_per_q/g_ctx_per_q:.1f}x less) at {g_j/fc_j*100:.1f}% "
          f"of its accuracy")

    print("\n=== (c) LENGTH-SCALING: GENOME vs full-context by transcript size ===")
    rows = []
    for cid in sorted(convs, key=lambda c: fc_ctx[c]):
        if cid not in gb or cid not in fc:
            continue
        gj = gb[cid][0] / gb[cid][1]
        fj = fc[cid][0] / fc[cid][1]
        rows.append((fc_ctx[cid], cid, gj, fj, gj - fj))
    print(f"  {'transcript_tok':>14} {'conv':>8} {'GENOME':>7} {'full-ctx':>8} {'delta':>7}")
    for tok, cid, gj, fj, dl in rows:
        print(f"  {tok:>14,} {cid:>8} {gj:>7.3f} {fj:>8.3f} {dl:>+7.3f}")
    # correlation of delta with size
    import statistics
    sizes = [r[0] for r in rows]; deltas = [r[4] for r in rows]
    if len(rows) > 2:
        mx, md = statistics.mean(sizes), statistics.mean(deltas)
        cov = sum((s-mx)*(d-md) for s, d in zip(sizes, deltas))
        sx = sum((s-mx)**2 for s in sizes) ** 0.5
        sd = sum((d-md)**2 for d in deltas) ** 0.5
        r = cov / (sx*sd) if sx and sd else 0
        half = len(rows)//2
        short_d = statistics.mean(deltas[:half]); long_d = statistics.mean(deltas[half:])
        print(f"\n  corr(transcript size, GENOME-minus-fullctx delta) = {r:+.2f}")
        print(f"  shortest half mean delta = {short_d:+.3f} | "
              f"longest half mean delta = {long_d:+.3f}")
        print("  (positive slope / less-negative on long convs = memory layer "
              "closes the gap as context grows -- the thesis)")


if __name__ == "__main__":
    raise SystemExit(main())
