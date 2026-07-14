"""Fair full-context baseline for the haystack: RECENCY window (last-B tokens).

The original haystack full-context arm used PREFIX truncation (first-B tokens),
which systematically drops later sessions. A real chat app keeps the MOST RECENT
messages, so the fair full-context baseline is the last-B tokens. This script
reuses the EXACT same 120 questions (read from the existing answers.jsonl) and
the identical concatenated transcript, and computes the recency-window arm at the
same budgets. If GENOME beats BOTH prefix and recency truncation, the 'you
truncated from the wrong end' objection is closed.

Run: .venv/Scripts/python.exe benchmarks/haystack_recency.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import tiktoken
from anthropic import Anthropic

from genome.evals.baselines import _format_turn_text
from genome.evals.llm_judge import judge_answer, preprocess_gold_mem0
from genome.evals.locomo import (
    ANSWER_PROMPT, _is_abstention, _sanitize_locomo_text, load_locomo,
)

ENC = tiktoken.get_encoding("cl100k_base")
MODEL = "claude-haiku-4-5-20251001"
OUT = Path("results/haystack")
client = Anthropic()


def haiku(prompt, mt=256):
    import time as _t
    for a in range(5):
        try:
            r = client.messages.create(model=MODEL, max_tokens=mt, temperature=0.0,
                                       messages=[{"role": "user", "content": prompt}])
            return (r.content[0].text if r.content else "") or ""
        except Exception:
            if a == 4:
                raise
            _t.sleep(2 ** a)


def answer(context, question):
    return haiku(ANSWER_PROMPT.format(context=context or "(no relevant memories)",
                                      question=_sanitize_locomo_text(question))).strip()


def judge(cat, gold, predicted):
    if cat == "adversarial":
        return "CORRECT" if _is_abstention(predicted) else "INCORRECT"
    return judge_answer(lambda p: haiku(p), "", preprocess_gold_mem0(cat, gold),
                        predicted, mode="mem0").label


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("need ANTHROPIC_API_KEY"); return 1
    src = OUT / "answers.jsonl"
    rows = [json.loads(l) for l in src.open() if l.strip()]
    # budgets = the exact keys used in the prefix arm, so the two are comparable
    budgets = sorted({int(b) for r in rows for b in r["fullctx"]})
    print(f"{len(rows)} questions, budgets {budgets}", flush=True)

    convs = load_locomo("benchmarks/data/locomo10.json")
    # rebuild the IDENTICAL concatenated transcript as haystack.py
    transcript_ids = ENC.encode("\n".join(
        f"- {_sanitize_locomo_text(_format_turn_text(t))}"
        for c in convs for t in c.turns))
    print(f"transcript = {len(transcript_ids):,} tokens", flush=True)

    ck = OUT / "recency.jsonl"
    done = set()
    if ck.exists():
        for l in ck.open():
            try: done.add(json.loads(l)["qid"])
            except Exception: pass

    with ck.open("a") as fh:
        for i, r in enumerate(rows):
            if r["qid"] in done:
                continue
            out = {"qid": r["qid"], "category": r["category"], "recency": {}}
            for B in budgets:
                ctx = ENC.decode(transcript_ids[-B:])       # LAST B tokens
                pred = answer(ctx, r["question"])
                out["recency"][str(B)] = {
                    "label": judge(r["category"], r["gold"], pred),
                    "ctx_tokens": min(B, len(transcript_ids))}
            fh.write(json.dumps(out) + "\n"); fh.flush()
            if (i + 1) % 10 == 0:
                print(f"  ...{i+1}/{len(rows)}", flush=True)
    print("DONE recency arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
