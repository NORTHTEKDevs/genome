"""LongMemEval retrieval-quality test: does GENOME's reranking gain hold on a HARDER,
non-saturated benchmark than LoCoMo? Free loop (embeddings + local cross-encoder, no
answer/judge LLM). Gold = turns flagged has_answer=True; hit-rate@k = fraction retrieved.

longmemeval_s has ~40 sessions/question with distractors (retrieval actually matters);
oracle has only evidence sessions (easy -- for plumbing validation).

Run: .venv/Scripts/python.exe benchmarks/lme_retrieval.py --data benchmarks/data/lme/longmemeval_s --n 60
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from genome.embeddings import EmbeddingProvider

KS = [5, 10, 20]
POOL = 50


def flatten(q):
    """-> list of (turn_text, is_gold). One memory per turn."""
    turns = []
    for sess in q["haystack_sessions"]:
        for t in sess:
            if not isinstance(t, dict):
                continue
            txt = f"{t.get('role','user')}: {t.get('content','')}"
            turns.append((txt[:6000], bool(t.get("has_answer"))))  # cap: embedder limit 8192 tok
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmarks/data/lme/longmemeval_s")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("need OPENAI_API_KEY"); return 1
    data = json.load(open(args.data, encoding="utf-8"))
    # balanced-ish subset favoring retrieval-relevant types, only Qs that HAVE gold turns
    order = ["temporal-reasoning", "knowledge-update", "multi-session",
             "single-session-preference", "single-session-assistant", "single-session-user"]
    by = defaultdict(list)
    for q in data:
        if any(t[1] for t in flatten(q)):
            by[q["question_type"]].append(q)
    sample, per = [], max(1, args.n // len(order))
    for qt in order:
        sample += by.get(qt, [])[:per]
    sample = sample[: args.n]

    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

    STRATS = ["dense", "rerank", "rrf_w2", "rrf_w3"]
    hit = {s: {k: [] for k in KS} for s in STRATS}
    cat_hit = defaultdict(lambda: {s: [] for s in STRATS})
    n_sess = []

    def wrrf(dense_order, rr_order, w_rr, k=60):
        sc = defaultdict(float)
        for r, i in enumerate(dense_order, 1):
            sc[i] += 1.0 / (k + r)
        for r, i in enumerate(rr_order, 1):
            sc[i] += w_rr / (k + r)
        return sorted(sc, key=lambda x: sc[x], reverse=True)

    for qi, q in enumerate(sample):
        turns = flatten(q)
        gold = {i for i, (_, g) in enumerate(turns) if g}
        if not gold:
            continue
        n_sess.append(len(q["haystack_sessions"]))
        texts = [t for t, _ in turns]
        vecs = np.asarray(embed.encode_batch(texts), dtype=np.float32)
        vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        qv = np.asarray(embed.encode(q["question"]), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        cos = vn @ qv
        dense_order = list(np.argsort(-cos))
        dpool = dense_order[:POOL]
        pairs = [(q["question"], texts[i]) for i in dpool]
        rr = [dpool[i] for i in np.argsort(-np.asarray(ce.predict(pairs, show_progress_bar=False)))]
        orders = {"dense": dense_order, "rerank": rr,
                  "rrf_w2": wrrf(dpool, rr, 2.0), "rrf_w3": wrrf(dpool, rr, 3.0)}
        for name, order_ in orders.items():
            for k in KS:
                h = len(set(order_[:k]) & gold) / len(gold)
                hit[name][k].append(h)
                cat_hit[q["question_type"]][name].append(h if k == 10 else None)
        if (qi + 1) % 10 == 0:
            print(f"  ...{qi+1}/{len(sample)}", flush=True)

    n = len(hit["dense"][10])
    print(f"\n=== LongMemEval retrieval ({os.path.basename(args.data)}, {n} Q, "
          f"avg {np.mean(n_sess):.0f} sessions/Q) ===")
    print(f"{'strategy':13}" + "".join(f"  hit@{k:<3}" for k in KS))
    for s in STRATS:
        print(f"{s:13}" + "".join(f"  {np.mean(hit[s][k]):.3f} " for k in KS))
    print("\nper question_type (hit@10):  " + " / ".join(STRATS))
    for qt in order:
        vals = {s: [x for x in cat_hit[qt][s] if x is not None] for s in STRATS}
        d = vals["dense"]
        if d:
            print(f"  {qt:26} " + " / ".join(f"{np.mean(vals[s]):.3f}" for s in STRATS) + f"  (n={len(d)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
