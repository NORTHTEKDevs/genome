"""Main runner for the mini-LoCoMo dry-run benchmark.

Run from the repo root with:
    python -m benchmarks.mini_locomo.run

Outputs to results/mini_locomo/dry_run_<timestamp>.json plus a printable
summary table to stdout.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from benchmarks.mini_locomo.data import CONVERSATION, QUESTIONS
from benchmarks.mini_locomo.llm import ApiKeyLLM
from benchmarks.mini_locomo.scoring import (
    contains_all_keywords,
    keyword_recall,
    token_f1,
)
from benchmarks.mini_locomo.systems import GenomeFullSystem, NaiveSystem

USER_ID = "alice_dryrun"
RESULTS_DIR = Path(__file__).parents[2] / "results" / "mini_locomo"


def _score_answer(pred: str, q) -> dict:
    return {
        "token_f1": round(token_f1(pred, q.gold_answer), 3),
        "keyword_recall": round(keyword_recall(pred, q.gold_keywords), 3),
        "contains_all_keywords": contains_all_keywords(pred, q.gold_keywords),
    }


def _run_system(system_cls, llm) -> dict:
    print(f"\n=== {system_cls.__name__} ===")
    if system_cls is GenomeFullSystem:
        system = system_cls(llm)
    else:
        system = system_cls()
    t0 = time.time()
    try:
        print(f"  ingest: {len(CONVERSATION)} turns ...")
        system.ingest(CONVERSATION, user_id=USER_ID)
        ingest_seconds = time.time() - t0
        print(f"  ingest done in {ingest_seconds:.1f}s")

        per_question = []
        for q in QUESTIONS:
            tq = time.time()
            try:
                pred = system.answer(q.text, user_id=USER_ID, llm=llm)
            except Exception as e:
                pred = f"<ERROR: {e!r}>"
            scores = _score_answer(pred, q)
            elapsed = time.time() - tq
            print(
                f"    {q.qid:18s}  f1={scores['token_f1']:.2f}  "
                f"kw={scores['keyword_recall']:.2f}  ({elapsed:.1f}s) -> {pred[:80]}"
            )
            per_question.append(
                {
                    "qid": q.qid,
                    "category": q.category,
                    "question": q.text,
                    "gold": q.gold_answer,
                    "prediction": pred,
                    "scores": scores,
                    "elapsed_seconds": round(elapsed, 1),
                }
            )
        total_seconds = time.time() - t0
        return {
            "system": system_cls.__name__,
            "ingest_seconds": round(ingest_seconds, 1),
            "total_seconds": round(total_seconds, 1),
            "per_question": per_question,
            "aggregate": _aggregate(per_question),
        }
    finally:
        system.close()


def _aggregate(per_question: list[dict]) -> dict:
    if not per_question:
        return {}
    n = len(per_question)
    f1 = sum(q["scores"]["token_f1"] for q in per_question) / n
    kw = sum(q["scores"]["keyword_recall"] for q in per_question) / n
    em = sum(q["scores"]["contains_all_keywords"] for q in per_question) / n
    by_cat: dict[str, dict] = {}
    for q in per_question:
        c = q["category"]
        by_cat.setdefault(c, {"f1_sum": 0.0, "kw_sum": 0.0, "em_sum": 0.0, "n": 0})
        by_cat[c]["f1_sum"] += q["scores"]["token_f1"]
        by_cat[c]["kw_sum"] += q["scores"]["keyword_recall"]
        by_cat[c]["em_sum"] += q["scores"]["contains_all_keywords"]
        by_cat[c]["n"] += 1
    cats = {
        c: {
            "n": v["n"],
            "f1": round(v["f1_sum"] / v["n"], 3),
            "kw": round(v["kw_sum"] / v["n"], 3),
            "em": round(v["em_sum"] / v["n"], 3),
        }
        for c, v in by_cat.items()
    }
    return {
        "n": n,
        "mean_f1": round(f1, 3),
        "mean_keyword_recall": round(kw, 3),
        "exact_keyword_match_rate": round(em, 3),
        "by_category": cats,
    }


def main() -> None:
    # Seed every RNG path so embedding-quantization, k-means cluster init,
    # and any downstream sampling is reproducible across runs. LLM sampling
    # is not deterministic by default but the rest of the pipeline is.
    import random as _random

    import numpy as _np
    _random.seed(42)
    _np.random.seed(42)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"dry_run_{timestamp}.json"

    print("Mini-LoCoMo dry-run")
    print(f"Conversation turns: {len(CONVERSATION)}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Results -> {out_path}")

    llm = ApiKeyLLM(model="claude-sonnet-4-5-20250929")
    print(f"LLM: {llm.model} (Anthropic API)")

    results = {
        "run_id": timestamp,
        "model": llm.model,
        "n_turns": len(CONVERSATION),
        "n_questions": len(QUESTIONS),
        "systems": [],
        "errors": [],
    }
    try:
        for system_cls in (NaiveSystem, GenomeFullSystem):
            try:
                results["systems"].append(_run_system(system_cls, llm))
            except Exception as e:
                tb = traceback.format_exc()
                print(f"\n{system_cls.__name__} CRASHED: {e}")
                print(tb)
                results["errors"].append(
                    {"system": system_cls.__name__, "error": repr(e), "traceback": tb}
                )
    finally:
        results["llm_calls_total"] = llm.calls
        results["llm_total_seconds"] = round(llm.total_seconds, 1)
        results["llm_avg_seconds"] = (
            round(llm.total_seconds / llm.calls, 2) if llm.calls else 0.0
        )
        results["llm_input_tokens"] = getattr(llm, "input_tokens", 0)
        results["llm_output_tokens"] = getattr(llm, "output_tokens", 0)
        results["llm_cost_usd_estimate"] = round(
            getattr(llm, "cost_estimate_usd", 0.0), 4
        )
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nLLM stats: {llm.calls} calls, {llm.total_seconds:.0f}s total, "
              f"{results['llm_avg_seconds']:.1f}s/call avg")
        print(
            f"  input tokens: {results['llm_input_tokens']:,}  "
            f"output tokens: {results['llm_output_tokens']:,}  "
            f"cost: ${results['llm_cost_usd_estimate']:.3f}"
        )
        print(f"Wrote {out_path}")
        try:
            llm.close()
        except Exception as e:
            print(f"LLM teardown error (non-fatal): {e!r}")

    _print_summary(results)


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in results["systems"]:
        agg = s.get("aggregate", {})
        print(
            f"\n{s['system']}  ({s['total_seconds']}s total)\n"
            f"  mean_f1: {agg.get('mean_f1', 0):.3f}\n"
            f"  mean_keyword_recall: {agg.get('mean_keyword_recall', 0):.3f}\n"
            f"  exact_keyword_match_rate: {agg.get('exact_keyword_match_rate', 0):.3f}"
        )
        for cat, cs in agg.get("by_category", {}).items():
            print(f"    {cat:14s}  n={cs['n']}  f1={cs['f1']:.2f}  kw={cs['kw']:.2f}  em={cs['em']:.2f}")
    if results.get("errors"):
        print("\nERRORS:")
        for e in results["errors"]:
            print(f"  {e['system']}: {e['error']}")


if __name__ == "__main__":
    main()
