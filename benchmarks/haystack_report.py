"""Crossover report for the LoCoMo-Haystack run.

Reads results/haystack/answers.jsonl (GENOME + PREFIX-truncated full-context) and,
if present, results/haystack/recency.jsonl (RECENCY-window / last-B full-context).
Reports accuracy + context tokens/query for GENOME vs BOTH truncation directions
at each budget on a ~284k-token history. The honest memory-layer result: constant-
cost retrieval holds accuracy at a fixed small cost while ANY fixed-window
truncation of an overflowing history -- first-B or last-B -- collapses.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

P = Path("results/haystack/answers.jsonl")
R = Path("results/haystack/recency.jsonl")


def main():
    rows = [json.loads(l) for l in P.open() if l.strip()]
    n = len(rows)
    if not n:
        print("no answers yet"); return 0
    budgets = sorted({int(b) for r in rows for b in r["fullctx"]})

    rec = {}
    if R.exists():
        for l in R.open():
            if l.strip():
                d = json.loads(l); rec[d["qid"]] = d["recency"]
    rec_rows = [r for r in rows if r["qid"] in rec]  # questions with a recency arm

    g_correct = sum(r["genome"]["label"] == "CORRECT" for r in rows)
    g_ctx = sum(r["genome"]["ctx_tokens"] for r in rows) / n

    print(f"=== LoCoMo-Haystack: {n} questions over a ~284k-token history "
          f"(exceeds a 200k window) ===\n")
    print(f"{'system':30} {'accuracy':>9} {'ctx tok/query':>14}")
    print(f"{'GENOME (retrieval)':30} {g_correct/n:>9.3f} {g_ctx:>14,.0f}")
    for b in budgets:
        c = sum(r["fullctx"][str(b)]["label"] == "CORRECT" for r in rows)
        ct = sum(r["fullctx"][str(b)]["ctx_tokens"] for r in rows) / n
        print(f"{'full-context PREFIX @'+f'{b//1000}k':30} {c/n:>9.3f} {ct:>14,.0f}")
    if rec:
        m = len(rec_rows)
        print(f"  -- recency arm on {m}/{n} questions --")
        for b in budgets:
            c = sum(rec[r['qid']][str(b)]["label"] == "CORRECT" for r in rec_rows)
            ct = sum(rec[r['qid']][str(b)]["ctx_tokens"] for r in rec_rows) / m
            print(f"{'full-context RECENCY @'+f'{b//1000}k':30} {c/m:>9.3f} {ct:>14,.0f}")
        # GENOME on the SAME recency-arm subset, for a matched comparison
        gc = sum(r["genome"]["label"] == "CORRECT" for r in rec_rows)
        print(f"{'GENOME (same subset)':30} {gc/m:>9.3f}")

    print("\n=== per-category accuracy (GENOME vs BEST full-context @128k) ===")
    big = str(budgets[-1])
    gc = defaultdict(lambda: [0, 0]); fc = defaultdict(lambda: [0, 0])
    rc = defaultdict(lambda: [0, 0])
    for r in rows:
        gc[r["category"]][1] += 1; gc[r["category"]][0] += r["genome"]["label"] == "CORRECT"
        fc[r["category"]][1] += 1; fc[r["category"]][0] += r["fullctx"][big]["label"] == "CORRECT"
        if r["qid"] in rec:
            rc[r["category"]][1] += 1
            rc[r["category"]][0] += rec[r["qid"]][big]["label"] == "CORRECT"
    for cat in sorted(gc):
        g = gc[cat][0]/gc[cat][1]; f = fc[cat][0]/fc[cat][1]
        rstr = ""
        if rc[cat][1]:
            rstr = f"  recency@128k {rc[cat][0]/rc[cat][1]:.3f}"
        print(f"  {cat:12} GENOME {g:.3f}  prefix@128k {f:.3f}{rstr}  (n={gc[cat][1]})")

    # headline vs the STRONGEST full-context arm across both truncation directions
    best_pref = max(sum(r["fullctx"][str(b)]["label"] == "CORRECT" for r in rows)/n
                    for b in budgets)
    line = (f"\nHEADLINE: GENOME {g_correct/n:.3f} @ {g_ctx:,.0f} tok/query vs "
            f"best PREFIX full-context {best_pref:.3f}")
    if rec:
        m = len(rec_rows)
        best_rec = max(
            sum(rec[r['qid']][str(b)]["label"] == "CORRECT" for r in rec_rows)/m
            for b in budgets)
        gsub = sum(r["genome"]["label"] == "CORRECT" for r in rec_rows)/m
        line += (f"; on the recency subset GENOME {gsub:.3f} vs best RECENCY "
                 f"full-context {best_rec:.3f}")
    print(line + ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
