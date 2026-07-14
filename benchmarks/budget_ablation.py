"""Memory-budget ablation: does GENOME's synthesize-before-prune retain more
answerable information than plain pruning, at EQUAL final memory footprint?

This isolates GENOME's novel IP. For each conversation, compress its turns to a
fixed footprint F two ways:
  - PRUNE : keep the top-F memories by fitness, forget the rest.
  - SYNTH : keep top-(F//2) + recombine the next-worst pairs into F//2 hybrids
            (GENOME's actual consolidation: content "consolidated: A[:60] + B[:60]",
            frequency_crossover embedding). Final footprint == F, same as PRUNE.
Everything else identical (same retrieval, same new answer prompt, same judge).
Any accuracy gap is attributable ONLY to synthesis vs forgetting.

Run: .venv/Scripts/python.exe benchmarks/budget_ablation.py [--convs 3] [--budget 60]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from anthropic import Anthropic

from genome.embeddings import EmbeddingProvider
from genome.evals.llm_judge import judge_answer, preprocess_gold_mem0
from genome.evals.locomo import (
    ANSWER_PROMPT, _is_abstention, _sanitize_locomo_text, load_locomo,
)
from genome.memory.consolidation import score_memories
from genome.memory.facade import Memory
from genome.memory.schema import MemoryRecord
from genome.synthesis import recombine

MODEL = "claude-haiku-4-5-20251001"
HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}
OUT = Path("results/budget"); client = Anthropic()


def haiku(p, mt=256):
    r = client.messages.create(model=MODEL, max_tokens=mt, temperature=0.0,
                               messages=[{"role": "user", "content": p}])
    return (r.content[0].text if r.content else "") or ""


def answer(ctx, q):
    return haiku(ANSWER_PROMPT.format(context=ctx or "(no relevant memories)",
                                      question=_sanitize_locomo_text(q))).strip()


def judge(cat, gold, pred):
    if cat == "adversarial":
        return "CORRECT" if _is_abstention(pred) else "INCORRECT"
    return judge_answer(lambda p: haiku(p), "", preprocess_gold_mem0(cat, gold),
                        pred, mode="mem0").label


def build_records(conv, embed):
    texts, metas = [], []
    for t in conv.turns:
        c = (f"[{t.session_datetime}] {t.speaker}: {t.text}"
             if t.session_datetime else f"{t.speaker}: {t.text}")
        texts.append(c); metas.append(t)
    vecs = embed.encode_batch(texts)
    return [MemoryRecord(content=c, embedding=np.asarray(v, dtype=np.float32),
                         user_id="u", agent_id=conv.conversation_id,
                         metadata={"dia_id": t.dia_id})
            for c, v, t in zip(texts, vecs, metas)]


def compress_prune(records, F):
    scored = score_memories(records); scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:F]]


def compress_synth(records, F):
    scored = score_memories(records); scored.sort(key=lambda x: x[1], reverse=True)
    k = F // 2
    keepers = [r for r, _ in scored[:k]]
    rest = [r for r, _ in scored[k:]]
    hybrids = []
    for i in range(0, min(len(rest) - 1, 2 * (F - k)), 2):
        a, b = rest[i], rest[i + 1]
        try:
            emb = recombine([a.embedding, b.embedding], operator="frequency_crossover")
        except Exception:
            continue
        hybrids.append(MemoryRecord(
            content=f"consolidated: {a.content[:60]} + {b.content[:60]}",
            embedding=np.asarray(emb, dtype=np.float32),
            user_id="u", agent_id=a.agent_id, parents=[a.id, b.id],
            operator="frequency_crossover",
            metadata={"consolidation": True}))
    return keepers + hybrids


def store_from(records, embed):
    m = Memory(storage=":memory:", embedding_provider=embed)
    for r in records:
        m.store.add(r)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=3)
    ap.add_argument("--budget", type=int, default=60)
    args = ap.parse_args()
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need both keys"); return 1
    OUT.mkdir(parents=True, exist_ok=True); ck = OUT / "answers.jsonl"
    F = args.budget
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    convs = load_locomo("benchmarks/data/locomo10.json")[: args.convs]

    done = set()
    if ck.exists():
        for l in ck.open():
            try: done.add(json.loads(l)["qid"])
            except Exception: pass

    with ck.open("a") as fh:
        for conv in convs:
            recs = build_records(conv, embed)
            pruned = store_from(compress_prune(recs, F), embed)
            synthd = store_from(compress_synth(recs, F), embed)
            print(f"{conv.conversation_id}: {len(recs)} turns -> "
                  f"prune {pruned.count(user_id='u')} / synth "
                  f"{synthd.count(user_id='u')} memories (budget {F})", flush=True)
            qs = [q for q in conv.questions if q.category in HEADLINE]
            for q in qs:
                qid = f"{conv.conversation_id}:{q.question_id}"
                if qid in done:
                    continue
                kk = min(30, F)
                pc = "\n".join(f"- {_sanitize_locomo_text(r.record.content)}"
                               for r in pruned.search(q.question, user_id="u", limit=kk))
                sc = "\n".join(f"- {_sanitize_locomo_text(r.record.content)}"
                               for r in synthd.search(q.question, user_id="u", limit=kk))
                pl = judge(q.category, q.answer, answer(pc, q.question))
                sl = judge(q.category, q.answer, answer(sc, q.question))
                fh.write(json.dumps({"qid": qid, "category": q.category,
                                     "prune": pl, "synth": sl}) + "\n"); fh.flush()
            pruned.close(); synthd.close()

    rows = [json.loads(l) for l in ck.open() if l.strip()]
    n = len(rows)
    pc = sum(r["prune"] == "CORRECT" for r in rows)
    sc = sum(r["synth"] == "CORRECT" for r in rows)
    s_fix = sum(r["synth"] == "CORRECT" and r["prune"] != "CORRECT" for r in rows)
    p_fix = sum(r["prune"] == "CORRECT" and r["synth"] != "CORRECT" for r in rows)
    import math
    disc = s_fix + p_fix
    p = math.erfc(math.sqrt(((abs(s_fix-p_fix)-1)**2/disc)/2)) if disc else 1.0
    print(f"\n=== MEMORY-BUDGET ABLATION (budget F={F}, n={n}, equal footprint) ===")
    print(f"  PRUNE-only            : {pc/n:.3f} ({pc}/{n})")
    print(f"  SYNTH-before-prune    : {sc/n:.3f} ({sc}/{n})")
    print(f"  delta (synth - prune) : {(sc-pc)/n:+.3f}  "
          f"| synth-fixed {s_fix}, prune-fixed {p_fix}, McNemar p={p:.3f} "
          f"-> {'SIGNIFICANT' if p<0.05 and sc>pc else ('lead' if sc>pc else 'no gain')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
