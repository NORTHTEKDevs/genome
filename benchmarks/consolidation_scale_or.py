"""Consolidation at scale: does AUTO-consolidation (threshold-triggered) keep
answer accuracy while shrinking the store, and does synthesis beat prune-only
when it fires on its own?

Paper-1 finding 05 tested manual consolidation at equal token budget (wash,
p=0.86). This measures the SHIPPED path: auto_consolidate_threshold fires
mid-stream during a vanilla add() loop over one long LoCoMo conversation, and
we compare three arms on the same questions:

    off        threshold=None                     (control, full store)
    synth      threshold/target with synthesize=True  (frequency_crossover)
    prune      threshold/target with synthesize=False (drop lowest-value)

Measured per arm: store-size trajectory (sampled), final store size, ingest
wall time, answer accuracy (same responder+judge protocol) + paired McNemar
vs the control arm.

Env: OPENROUTER_API_KEY required.
Run: .venv/Scripts/python.exe benchmarks/consolidation_scale_or.py
     [--conv 0] [--threshold 300] [--target 150]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from openai import OpenAI

from genome.embeddings import EmbeddingProvider
from genome.evals.llm_judge import judge_answer, preprocess_gold_mem0
from genome.evals.locomo import ANSWER_PROMPT, _sanitize_locomo_text, load_locomo
from genome.memory.facade import Memory

OR_BASE = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("OR_MODEL", "anthropic/claude-haiku-4.5")
OUT = Path("results/consolidation_scale")
_client: OpenAI | None = None


def llm(p, mt=400):
    global _client
    if _client is None:
        _client = OpenAI(base_url=OR_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    for a in range(5):
        try:
            r = _client.chat.completions.create(
                model=MODEL, max_tokens=mt, temperature=0.0,
                messages=[{"role": "user", "content": p}])
            return (r.choices[0].message.content or "") if r.choices else ""
        except Exception:
            if a == 4:
                raise
            time.sleep(2 ** a)


def run_arm(name, conv, threshold, target, synthesize):
    embed = EmbeddingProvider()  # local default
    kwargs = {}
    if threshold is not None:
        kwargs = dict(auto_consolidate_threshold=threshold,
                      auto_consolidate_target=target,
                      auto_consolidate_synthesize=synthesize)
    mem = Memory(storage=":memory:", embedding_provider=embed,
                 llm_call=llm, **kwargs)
    traj = []
    t0 = time.time()
    for i, t in enumerate(conv.turns):
        mem.add(f"[{t.session_datetime}] {t.speaker}: {t.text}", user_id="cs")
        if i % 25 == 0 or i == len(conv.turns) - 1:
            traj.append((i + 1, len(mem.list_all(user_id="cs"))))
    ingest_s = time.time() - t0
    final_n = len(mem.list_all(user_id="cs"))
    print(f"[{name}] ingested {len(conv.turns)} turns in {ingest_s:.0f}s; "
          f"final store={final_n}; trajectory={traj}", flush=True)

    labels = {}
    for q in conv.questions:
        if q.category == "adversarial":
            continue
        hits = mem.search(q.question, user_id="cs", limit=30)
        ctx = "\n".join(f"- {_sanitize_locomo_text(h.content)}" for h in hits)
        pred = llm(ANSWER_PROMPT.format(context=ctx or "(no relevant memories)",
                                        question=_sanitize_locomo_text(q.question))).strip()
        lab = judge_answer(lambda p: llm(p), "",
                           preprocess_gold_mem0(q.category, str(q.answer)),
                           pred, mode="mem0").label
        labels[q.question] = lab
    mem.close()
    acc = sum(v == "CORRECT" for v in labels.values()) / len(labels) if labels else 0.0
    print(f"[{name}] accuracy={acc:.3f} over {len(labels)} questions", flush=True)
    return {"labels": labels, "acc": acc, "final_n": final_n,
            "ingest_s": round(ingest_s, 1), "traj": traj}


def mcnemar(la, lb):
    ks = [k for k in la if k in lb]
    a_only = sum(la[k] == "CORRECT" and lb[k] != "CORRECT" for k in ks)
    b_only = sum(lb[k] == "CORRECT" and la[k] != "CORRECT" for k in ks)
    d = a_only + b_only
    p = math.erfc(math.sqrt(((abs(a_only - b_only) - 1) ** 2 / d) / 2)) if d else 1.0
    return a_only, b_only, p, len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--threshold", type=int, default=300)
    ap.add_argument("--target", type=int, default=150)
    args = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("need OPENROUTER_API_KEY"); return 1
    conv = load_locomo("benchmarks/data/locomo10.json")[args.conv]
    nq = sum(q.category != "adversarial" for q in conv.questions)
    print(f"[cfg] model={MODEL} via OpenRouter  embedder=local all-MiniLM  "
          f"conv={conv.conversation_id} ({len(conv.turns)} turns, {nq} questions)  "
          f"threshold={args.threshold} target={args.target}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    arms = [
        ("off", None, args.target, True),
        ("synth", args.threshold, args.target, True),
        ("prune", args.threshold, args.target, False),
    ]
    results = {}
    for name, th, tg, sy in arms:
        results[name] = run_arm(name, conv, th, tg, sy)
        with open(OUT / f"labels_{name}.json", "w", encoding="utf-8") as f:
            json.dump(results[name]["labels"], f)

    print("\n=== Consolidation at scale ===")
    print(f"{'arm':8}{'accuracy':>10}{'final store':>13}{'ingest s':>10}")
    for name in results:
        r = results[name]
        print(f"{name:8}{r['acc']:>10.3f}{r['final_n']:>13}{r['ingest_s']:>10.1f}")
    print("\nMcNemar vs control (off):")
    for name in ("synth", "prune"):
        ao, bo, p, n = mcnemar(results[name]["labels"], results["off"]["labels"])
        verdict = "differs" if p < 0.05 else "no significant difference"
        print(f"  {name:6} vs off  n={n}  {name}-only {ao}  off-only {bo}  "
              f"p={p:.4f} -> {verdict}")
    print("\nREAD: if accuracy holds within noise at a fraction of the store, "
          "auto-consolidation is a safe cost lever; if it drops, the shipped "
          "trigger is not a free win. synth vs prune isolates the synthesis "
          "operator under its own trigger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
