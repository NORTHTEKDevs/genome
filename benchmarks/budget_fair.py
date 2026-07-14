"""Token-FAIR memory-budget ablation. Removes the token-budget confound.

The earlier ablation equalized memory COUNT (60 vs 60), but a cluster-summary is
a dense ~400-token fact-list while a pruned turn is ~30 tokens -- so the cluster
arm was fed far more context. That inflates any 'win'. Here BOTH the storage
budget AND the answer-context budget are equalized in TOKENS, so the only
variable is whether summarization packs more answerable information per token.

  storage budget  S : keep memories (by fitness) until cumulative content tokens
                      ~= S. PRUNE keeps raw turns; CLUSTER keeps some turns +
                      cluster-summaries, both trimmed to <= S tokens.
  answer budget   B : at query time, fill the answer context with retrieved
                      items up to B tokens (identical for both arms).

If CLUSTER > PRUNE at equal S and equal B, synthesis genuinely retains more
answerable info per token -- a real, unconfounded GENOME IP win.

Run: .venv/Scripts/python.exe benchmarks/budget_fair.py [--convs 4] [--storage 3000] [--answer 1500]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tiktoken
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

ENC = tiktoken.get_encoding("cl100k_base")
MODEL = "claude-haiku-4-5-20251001"
HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}
OUT = Path("results/budget_fair"); client = Anthropic()

SUMMARY_PROMPT = """Extract every concrete fact from these conversation turns as
a dense bulleted list: preserve all names, dates, places, numbers, events, and
who-did-what. Keep the timestamps. Do not generalize or omit specifics.

TURNS:
{turns}

FACTS:"""


def ntok(s): return len(ENC.encode(s))
def haiku(p, mt=256):
    import time as _t
    for attempt in range(5):
        try:
            r = client.messages.create(model=MODEL, max_tokens=mt, temperature=0.0,
                                       messages=[{"role": "user", "content": p}])
            return (r.content[0].text if r.content else "") or ""
        except Exception as e:
            if attempt == 4:
                raise
            _t.sleep(2 ** attempt)  # 1,2,4,8s backoff on transient API errors
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


def _take_until_tokens(items, budget):
    """Greedily take records until cumulative content tokens exceed budget."""
    out, used = [], 0
    for r in items:
        tk = ntok(r.content)
        if out and used + tk > budget:
            break
        out.append(r); used += tk
    return out, used


# NOTE: score_memories()'s access*recency term is dead in offline replay
# (access_count is always 0), collapsing 'fitness' to centroid-similarity and
# selecting AGAINST specific facts. Use RECENCY instead -- a standard, sound
# sliding-window baseline. `records` are in chronological (turn) order, so
# most-recent-first = reversed.
def _by_recency(records):
    return list(reversed(records))


def compress_prune(records, S):
    keep, used = _take_until_tokens(_by_recency(records), S)
    return keep, used


def compress_cluster(records, S, embed):
    ordered = _by_recency(records)
    # Half the storage budget to raw high-fitness turns...
    keepers, used = _take_until_tokens(ordered, S // 2)
    rest = ordered[len(keepers):]
    if len(rest) < 4:
        more, u2 = _take_until_tokens(rest, S - used)
        return keepers + more, used + u2
    # ...the other half to cluster-summaries of the remainder.
    embs = np.stack([r.embedding for r in rest])
    # cluster count: aim so summaries fit the remaining budget (~200 tok each)
    remaining = S - used
    G = max(1, min(len(rest) // 2, remaining // 200))
    labels = _cluster_embeddings(embs, k=G)
    summaries = []
    for cid in range(G):
        members = [rest[i] for i in range(len(rest)) if labels[i] == cid]
        if not members:
            continue
        facts = haiku(SUMMARY_PROMPT.format(
            turns="\n".join(m.content for m in members[:40])), mt=300).strip()
        if not facts:
            continue
        v = embed.encode(facts)
        summaries.append(MemoryRecord(
            content=facts, embedding=np.asarray(v, dtype=np.float32),
            user_id="u", agent_id=members[0].agent_id,
            parents=[m.id for m in members], operator="cluster_summary",
            metadata={"consolidation": True}))
    # trim summaries to fit the remaining storage budget (token-fair)
    picked, u2 = _take_until_tokens(summaries, remaining)
    return keepers + picked, used + u2


def store_from(records, embed):
    m = Memory(storage=":memory:", embedding_provider=embed)
    for r in records:
        m.store.add(r)
    return m


def ctx_upto(store, q, B):
    """Retrieve and fill an answer context up to B tokens (identical for arms)."""
    parts, used = [], 0
    for r in store.search(q, user_id="u", limit=50, mode="dense"):
        line = f"- {_sanitize_locomo_text(r.record.content)}"
        tk = ntok(line)
        if parts and used + tk > B:
            break
        parts.append(line); used += tk
    return "\n".join(parts), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=4)
    ap.add_argument("--storage", type=int, default=3000)  # storage token budget S
    ap.add_argument("--answer", type=int, default=1500)   # answer-context budget B
    args = ap.parse_args()
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need both keys"); return 1
    OUT.mkdir(parents=True, exist_ok=True); ck = OUT / "answers.jsonl"
    S, B = args.storage, args.answer
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
            pk, ps = compress_prune(recs, S)
            ckk, cs = compress_cluster(recs, S, embed)
            pr, cl = store_from(pk, embed), store_from(ckk, embed)
            print(f"{conv.conversation_id}: {len(recs)} turns -> "
                  f"PRUNE {len(pk)} recs/{ps} tok  CLUSTER {len(ckk)} recs/{cs} tok "
                  f"(storage budget {S})", flush=True)
            for q in [q for q in conv.questions if q.category in HEADLINE]:
                qid = f"{conv.conversation_id}:{q.question_id}"
                if qid in done:
                    continue
                pc, ptok = ctx_upto(pr, q.question, B)
                cc, ctok = ctx_upto(cl, q.question, B)
                fh.write(json.dumps({
                    "qid": qid, "category": q.category,
                    "prune": judge(q.category, q.answer, answer(pc, q.question)),
                    "cluster": judge(q.category, q.answer, answer(cc, q.question)),
                    "prune_ctx_tok": ptok, "cluster_ctx_tok": ctok,
                }) + "\n"); fh.flush()
            pr.close(); cl.close()

    rows = [json.loads(l) for l in ck.open() if l.strip()]
    n = len(rows)
    pc = sum(r["prune"] == "CORRECT" for r in rows)
    cc = sum(r["cluster"] == "CORRECT" for r in rows)
    cfix = sum(r["cluster"] == "CORRECT" and r["prune"] != "CORRECT" for r in rows)
    pfix = sum(r["prune"] == "CORRECT" and r["cluster"] != "CORRECT" for r in rows)
    ap_tok = sum(r["prune_ctx_tok"] for r in rows) / n
    ac_tok = sum(r["cluster_ctx_tok"] for r in rows) / n
    import math
    disc = cfix + pfix
    p = math.erfc(math.sqrt(((abs(cfix-pfix)-1)**2/disc)/2)) if disc else 1.0
    print(f"\n=== TOKEN-FAIR ablation (storage S={S}, answer B={B}, n={n}) ===")
    print(f"  mean answer-context tokens: PRUNE {ap_tok:.0f}  CLUSTER {ac_tok:.0f} "
          f"(should be ~equal for a fair test)")
    print(f"  PRUNE-only        : {pc/n:.3f} ({pc}/{n})")
    print(f"  CLUSTER-summarize : {cc/n:.3f} ({cc}/{n})")
    print(f"  delta             : {(cc-pc)/n:+.3f} | cluster-fixed {cfix}, "
          f"prune-fixed {pfix}, McNemar p={p:.3f} -> "
          f"{'SIGNIFICANT WIN' if p<0.05 and cc>pc else ('lead' if cc>pc else 'no gain')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
