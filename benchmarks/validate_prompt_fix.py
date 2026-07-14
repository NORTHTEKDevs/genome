"""Held-out validation of the Tier-1 answer-prompt fix, in isolation.

Replays answer-generation for a stratified sample of the ORIGINAL run's
questions against their ORIGINAL retrieved context (from checkpoints), using the
NEW ANSWER_PROMPT, re-judges with the same Mem0-verbatim judge, and reports:
  - recovery rate on previously-abstained answerable questions,
  - adversarial abstention preservation (must stay high),
  - regression rate on previously-correct questions (must stay low).
No re-ingestion / no re-embedding: this isolates the prompt's effect.
Run: .venv/Scripts/python.exe benchmarks/validate_prompt_fix.py [--n-per 50]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

from anthropic import Anthropic

from genome.evals.llm_judge import judge_answer, preprocess_gold_mem0
from genome.evals.locomo import ANSWER_PROMPT, _is_abstention, _sanitize_locomo_text

MODEL = "claude-haiku-4-5-20251001"
CFG = "genome-parent-filtered"
random.seed(20260710)  # fixed sample; new Date() unavailable + reproducible

client = Anthropic()


def haiku(prompt: str, max_tokens: int) -> str:
    r = client.messages.create(
        model=MODEL, max_tokens=max_tokens, temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return (r.content[0].text if r.content else "") or ""


def answer_with_new_prompt(q_text: str, retrieved: list[str]) -> str:
    context = "\n".join(f"- {_sanitize_locomo_text(c)}" for c in retrieved)
    prompt = ANSWER_PROMPT.format(
        context=context or "(no relevant memories)",
        question=_sanitize_locomo_text(q_text),
    )
    return haiku(prompt, 256).strip()


def load_questions() -> list[dict]:
    out = []
    for f in glob.glob(f"results/locomo_claude/checkpoints/{CFG}__*.json"):
        out += json.load(open(f))["per_question"]
    return out


def judge(q: dict, predicted: str) -> str:
    if q["category"] == "adversarial":
        return "CORRECT" if _is_abstention(predicted) else "INCORRECT"
    gold = preprocess_gold_mem0(q["category"], q["gold"])
    return judge_answer(lambda p: haiku(p, 256), q["question"], gold, predicted,
                        mode="mem0").label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=50)
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY"); return 1

    qs = load_questions()
    non_adv = [q for q in qs if q["category"] != "adversarial"]
    adv = [q for q in qs if q["category"] == "adversarial"]

    # strata
    abstained = [q for q in non_adv
                 if q["judge_label"] == "INCORRECT"
                 and _is_abstention(q["predicted"])
                 and q["retrieval_hit_rate"] >= 0.5]
    temporal_wrong = [q for q in non_adv
                      if q["category"] == "temporal" and q["judge_label"] == "INCORRECT"
                      and not _is_abstention(q["predicted"])]
    prev_correct = [q for q in non_adv if q["judge_label"] == "CORRECT"]

    def samp(pool, n):
        random.shuffle(pool); return pool[:n]

    strata = {
        "recover_abstained": samp(abstained, args.n_per),
        "temporal_wrong": samp(temporal_wrong, min(args.n_per, 30)),
        "adversarial_control": samp(adv, min(args.n_per, 40)),
        "prev_correct_control": samp(prev_correct, min(args.n_per, 40)),
    }

    results = {}
    for name, pool in strata.items():
        newc = 0
        for q in pool:
            pred = answer_with_new_prompt(q["question"], q["retrieved_contents"])
            lbl = judge(q, pred)
            if lbl == "CORRECT":
                newc += 1
        results[name] = (newc, len(pool))
        print(f"{name:24} new-CORRECT {newc}/{len(pool)} "
              f"(was {sum(1 for q in pool if q['judge_label']=='CORRECT')}/{len(pool)})")
        sys.stdout.flush()

    print("\n=== READ ===")
    r, n = results["recover_abstained"]
    print(f"Abstention recovery: {r}/{n} = {r/n:.0%} of previously-refused "
          f"answerable questions now answered CORRECTLY")
    a, an = results["adversarial_control"]
    print(f"Adversarial preserved: {a}/{an} = {a/an:.0%} still correctly abstain "
          f"(must stay high -- this is the safety check)")
    p, pn = results["prev_correct_control"]
    print(f"Prev-correct held: {p}/{pn} = {p/pn:.0%} still correct "
          f"(regression = {pn-p})")
    t, tn = results["temporal_wrong"]
    print(f"Temporal-date recovery: {t}/{tn} = {t/tn:.0%} of date-format losses fixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
