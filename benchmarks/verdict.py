"""Head-to-head verdict with significance. Reads a results dir's checkpoints,
pairs systems by question_id, and reports headline J, per-category J, and a
McNemar paired significance test of GENOME vs each baseline.

'Winner' criterion (must ALL hold to publish a win claim):
  - GENOME headline J > baseline headline J, AND
  - McNemar p < 0.05 (the lead is not noise).

Run: .venv/Scripts/python.exe benchmarks/verdict.py [--dir results/locomo_claude_v2]
     [--genome genome-parent-filtered]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}


def load(dirp, system):
    rows = {}
    for f in glob.glob(f"{dirp}/checkpoints/{system}__*.json"):
        for q in json.load(open(f))["per_question"]:
            rows[q["question_id"]] = q
    return rows


def headline_j(rows):
    hq = [q for q in rows.values() if q["category"] in HEADLINE]
    if not hq:
        return 0.0, 0
    c = sum(1 for q in hq if q["judge_label"] == "CORRECT")
    return c / len(hq), len(hq)


def per_cat(rows):
    d = defaultdict(lambda: [0, 0])
    for q in rows.values():
        if q["category"] in HEADLINE:
            d[q["category"]][1] += 1
            d[q["category"]][0] += q["judge_label"] == "CORRECT"
    return d


def mcnemar(g, b):
    """Paired test on headline questions present in BOTH systems."""
    ids = [qid for qid in g if qid in b
           and g[qid]["category"] in HEADLINE]
    gc = sum(g[i]["judge_label"] == "CORRECT" for i in ids)
    bc = sum(b[i]["judge_label"] == "CORRECT" for i in ids)
    # discordant: b_only = G wrong / B right; c_only = G right / B wrong
    b_only = sum(1 for i in ids
                 if g[i]["judge_label"] != "CORRECT" and b[i]["judge_label"] == "CORRECT")
    c_only = sum(1 for i in ids
                 if g[i]["judge_label"] == "CORRECT" and b[i]["judge_label"] != "CORRECT")
    n = b_only + c_only
    if n == 0:
        return gc, bc, len(ids), b_only, c_only, 1.0
    chi = (abs(b_only - c_only) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi / 2))
    return gc, bc, len(ids), b_only, c_only, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/locomo_claude_v2")
    ap.add_argument("--genome", default="genome-parent-filtered")
    args = ap.parse_args()

    systems = sorted({
        os.path.basename(f).split("__")[0]
        for f in glob.glob(f"{args.dir}/checkpoints/*.json")
    })
    print(f"systems present: {systems}\n")
    rows = {s: load(args.dir, s) for s in systems}

    print("=== headline J ===")
    for s in sorted(systems, key=lambda s: -headline_j(rows[s])[0]):
        j, n = headline_j(rows[s])
        mark = "  <-- GENOME" if s == args.genome else ""
        print(f"  {s:26} {j:.3f}  (n={n}){mark}")

    if args.genome not in rows:
        print(f"\n(GENOME config {args.genome} not present yet)"); return 0
    g = rows[args.genome]

    print(f"\n=== {args.genome} vs each baseline (McNemar, paired) ===")
    for s in systems:
        if s == args.genome or s.startswith("genome-"):
            continue
        gc, bc, npair, b_only, c_only, p = mcnemar(g, rows[s])
        gj = gc / npair if npair else 0
        bj = bc / npair if npair else 0
        winner = (gj > bj) and (p < 0.05)
        print(f"  vs {s:24} G={gj:.3f} B={bj:.3f} delta={gj-bj:+.3f} "
              f"| G-fixed={c_only} B-fixed={b_only} p={p:.3f} "
              f"-> {'SIGNIFICANT WIN' if winner else ('lead, not sig' if gj>bj else 'NOT winning')}")

    print("\n=== per-category J (all systems) ===")
    cats = ["multi-hop", "temporal", "open-domain", "single-hop"]
    print(f"  {'system':26} " + " ".join(f"{c[:10]:>11}" for c in cats))
    for s in systems:
        pc = per_cat(rows[s])
        cells = [f"{pc[c][0]/pc[c][1]:.3f}" if pc[c][1] else "  -" for c in cats]
        print(f"  {s:26} " + " ".join(f"{x:>11}" for x in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
