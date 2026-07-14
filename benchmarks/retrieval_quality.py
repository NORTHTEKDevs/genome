"""Offline retrieval-quality harness (NO LLM / NO judge -> ~free, embeddings only).

Measures whether GENOME's retrieval MODES surface more gold evidence than plain
dense retrieval, using LoCoMo's annotated evidence dia_ids as ground truth. This
isolates the RETRIEVAL architecture from the (saturated) answer accuracy: a mode
that puts more evidence turns in the top-k is objectively better retrieval,
regardless of whether the ~0.85-ceiling answer model then uses it.

metric: hit-rate@k = |retrieved_dia_ids intersect evidence| / |evidence|,
        averaged over questions that HAVE evidence annotations.
also:   full-recall@k = fraction of questions where ALL evidence was retrieved.

modes compared: dense, hybrid, graph (GENOME facade modes) + a recency-reranked
dense variant (dense pool re-scored by 0.7*sim + 0.3*recency).

Run: .venv/Scripts/python.exe benchmarks/retrieval_quality.py [--convs 10]
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

from genome.embeddings import EmbeddingProvider
from genome.evals.locomo import load_locomo
from genome.memory.facade import Memory
from genome.memory.schema import MemoryRecord

HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}
KS = [5, 10, 20]
MODES = ["dense", "hybrid", "graph"]


def build(conv, embed):
    texts, metas = [], []
    for i, t in enumerate(conv.turns):
        c = (f"[{t.session_datetime}] {t.speaker}: {t.text}"
             if t.session_datetime else f"{t.speaker}: {t.text}")
        texts.append(c); metas.append((i, t.dia_id))
    vecs = embed.encode_batch(texts)
    m = Memory(storage=":memory:", embedding_provider=embed)
    for c, v, (i, dia) in zip(texts, vecs, metas):
        m.store.add(MemoryRecord(
            content=c, embedding=np.asarray(v, dtype=np.float32),
            user_id="u", agent_id=conv.conversation_id,
            metadata={"dia_id": dia, "ord": i}))
    return m, len(conv.turns)


def dia_of(r):
    md = r.record.metadata or {}
    return str(md.get("dia_id", ""))


def recency_rerank(results, n_turns, k):
    """Re-score a retrieved pool by 0.7*sim_rank + 0.3*recency, take top-k."""
    scored = []
    for rank, r in enumerate(results):
        sim = 1.0 - rank / max(len(results), 1)            # rank-based sim proxy
        rec = (r.record.metadata or {}).get("ord", 0) / max(n_turns - 1, 1)
        scored.append((0.7 * sim + 0.3 * rec, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=10)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("need OPENAI_API_KEY"); return 1
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    convs = load_locomo("benchmarks/data/locomo10.json")[: args.convs]

    # accumulators: strat -> k -> [hit_rates], and full-recall counts
    hit = defaultdict(lambda: defaultdict(list))
    full = defaultdict(lambda: defaultdict(int))
    cat_hit = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nq = 0

    for conv in convs:
        m, n_turns = build(conv, embed)
        for q in conv.questions:
            if q.category not in HEADLINE:
                continue
            ev = {str(e).strip() for e in q.evidence if str(e).strip()}
            if not ev:
                continue
            nq += 1
            pool = list(m.search(q.question, user_id="u", limit=max(KS), mode="dense"))
            for mode in MODES:
                res = list(m.search(q.question, user_id="u", limit=max(KS), mode=mode))
                for k in KS:
                    got = {dia_of(r) for r in res[:k]}
                    hr = len(got & ev) / len(ev)
                    hit[mode][k].append(hr)
                    cat_hit[q.category][mode][k].append(hr)
                    if ev <= got:
                        full[mode][k] += 1
            for k in KS:                        # recency-reranked dense
                got = {dia_of(r) for r in recency_rerank(pool, n_turns, k)}
                hr = len(got & ev) / len(ev)
                hit["dense+recency"][k].append(hr)
                if ev <= got:
                    full["dense+recency"][k] += 1
        m.close()

    print(f"\n=== RETRIEVAL QUALITY over {len(convs)} convs, "
          f"{nq} evidence-annotated headline questions ===")
    print(f"(metric: mean hit-rate@k = fraction of gold evidence turns retrieved)\n")
    str2 = MODES + ["dense+recency"]
    print(f"{'mode':16}" + "".join(f"  hit@{k:<3}" for k in KS)
          + "   " + "".join(f" full@{k:<3}" for k in KS))
    for s in str2:
        row = f"{s:16}"
        for k in KS:
            row += f"  {np.mean(hit[s][k]):.3f} "
        for k in KS:
            row += f"  {full[s][k]/nq:.3f}"
        print(row)

    print(f"\n=== per-category mean hit-rate@10 (dense vs hybrid vs graph) ===")
    for cat in sorted(cat_hit):
        d = np.mean(cat_hit[cat]["dense"][10])
        h = np.mean(cat_hit[cat]["hybrid"][10])
        g = np.mean(cat_hit[cat]["graph"][10])
        n = len(cat_hit[cat]["dense"][10])
        best = max([("dense", d), ("hybrid", h), ("graph", g)], key=lambda x: x[1])
        print(f"  {cat:12} dense {d:.3f}  hybrid {h:.3f}  graph {g:.3f}  "
              f"(n={n}, best={best[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
