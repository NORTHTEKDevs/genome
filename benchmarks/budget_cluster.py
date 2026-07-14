"""Rebuilt synthesis: semantic-cluster summarization vs plain pruning, at equal
memory footprint. The honest IP test done RIGHT.

For each conversation, compress its turns to footprint F two ways:
  - PRUNE   : keep top-F by fitness (forget the rest).
  - CLUSTER : keep top-(F-G) by fitness + summarize the rest into G dense
              cluster-summaries (MiniBatchKMeans over embeddings -> LLM extracts
              the concrete facts of each cluster -> re-embed). Final footprint F.
Same retrieval + answer prompt + judge. Any gap is prune-vs-cluster-synthesis.
This replaces the shipped synthesis (arbitrary-pair, 60-char truncation) that
lost to pruning (-0.04).

Run: .venv/Scripts/python.exe benchmarks/budget_cluster.py [--convs 4] [--budget 60]
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
from genome.memory.raptor import _cluster_embeddings
from genome.memory.schema import MemoryRecord

MODEL = "claude-haiku-4-5-20251001"
HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}
OUT = Path("results/budget_cluster"); client = Anthropic()

SUMMARY_PROMPT = """Extract every concrete fact from these conversation turns as
a dense bulleted list: preserve all names, dates, places, numbers, events, and
who-did-what. Keep the timestamps. Do not generalize or omit specifics.

TURNS:
{turns}

FACTS:"""


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
    s = score_memories(records); s.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in s[:F]]


def compress_cluster(records, F, embed, max_summary_members=400):
    s = score_memories(records); s.sort(key=lambda x: x[1], reverse=True)
    G = F // 2
    keepers = [r for r, _ in s[:F - G]]
    rest = [r for r, _ in s[F - G:]][:max_summary_members]
    if len(rest) < G or G < 1:
        return keepers + rest[:G]
    embs = np.stack([r.embedding for r in rest])
    labels = _cluster_embeddings(embs, k=G)
    summaries = []
    for cid in range(G):
        members = [rest[i] for i in range(len(rest)) if labels[i] == cid]
        if not members:
            continue
        turns = "\n".join(m.content for m in members[:40])
        facts = haiku(SUMMARY_PROMPT.format(turns=turns), mt=400).strip()
        if not facts:
            facts = " ".join(m.content for m in members[:5])
        v = embed.encode(facts)
        summaries.append(MemoryRecord(
            content=facts, embedding=np.asarray(v, dtype=np.float32),
            user_id="u", agent_id=members[0].agent_id,
            parents=[m.id for m in members], operator="cluster_summary",
            metadata={"consolidation": True, "n_members": len(members)}))
    return keepers + summaries


def store_from(records, embed):
    m = Memory(storage=":memory:", embedding_provider=embed)
    for r in records:
        m.store.add(r)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=4)
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
            pr = store_from(compress_prune(recs, F), embed)
            cl = store_from(compress_cluster(recs, F, embed), embed)
            print(f"{conv.conversation_id}: {len(recs)} turns -> prune "
                  f"{pr.count(user_id='u')} / cluster {cl.count(user_id='u')} "
                  f"memories (budget {F})", flush=True)
            for q in [q for q in conv.questions if q.category in HEADLINE]:
                qid = f"{conv.conversation_id}:{q.question_id}"
                if qid in done:
                    continue
                kk = min(30, F)
                pc = "\n".join(f"- {_sanitize_locomo_text(r.record.content)}"
                               for r in pr.search(q.question, user_id="u", limit=kk))
                cc = "\n".join(f"- {_sanitize_locomo_text(r.record.content)}"
                               for r in cl.search(q.question, user_id="u", limit=kk))
                fh.write(json.dumps({"qid": qid, "category": q.category,
                    "prune": judge(q.category, q.answer, answer(pc, q.question)),
                    "cluster": judge(q.category, q.answer, answer(cc, q.question))}
                    ) + "\n"); fh.flush()
            pr.close(); cl.close()

    rows = [json.loads(l) for l in ck.open() if l.strip()]
    n = len(rows)
    pc = sum(r["prune"] == "CORRECT" for r in rows)
    cc = sum(r["cluster"] == "CORRECT" for r in rows)
    cfix = sum(r["cluster"] == "CORRECT" and r["prune"] != "CORRECT" for r in rows)
    pfix = sum(r["prune"] == "CORRECT" and r["cluster"] != "CORRECT" for r in rows)
    import math
    disc = cfix + pfix
    p = math.erfc(math.sqrt(((abs(cfix-pfix)-1)**2/disc)/2)) if disc else 1.0
    print(f"\n=== CLUSTER-SYNTHESIS vs PRUNE (F={F}, n={n}, equal footprint) ===")
    print(f"  PRUNE-only          : {pc/n:.3f} ({pc}/{n})")
    print(f"  CLUSTER-summarize   : {cc/n:.3f} ({cc}/{n})")
    print(f"  delta               : {(cc-pc)/n:+.3f}  | cluster-fixed {cfix}, "
          f"prune-fixed {pfix}, McNemar p={p:.3f} -> "
          f"{'SIGNIFICANT WIN' if p<0.05 and cc>pc else ('lead' if cc>pc else 'no gain')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
