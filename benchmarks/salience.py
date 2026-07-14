"""Salience lab (free: embeddings + local cross-encoder). Does weighting retrieval by
memory IMPORTANCE or DIVERSITY beat the shipped dense+rerank(RRF)? Honest test on
LongMemEval-S (where multi-evidence recall matters). Gold = has_answer turns.

Strategies:
  rerank_rrf   : the shipped fusion (dense + cross-encoder), baseline
  mmr          : rerank_rrf pool, then Maximal Marginal Relevance reselection
                 (diversify so top-k isn't near-duplicate turns -> more distinct gold)
  importance   : blend cosine with an importance heuristic (entity/number/length),
                 downweighting filler turns

Run: .venv/Scripts/python.exe benchmarks/salience.py --n 60
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

from genome.embeddings import EmbeddingProvider

KS = [5, 10, 20]
POOL = 50
_CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NUM = re.compile(r"\b\d{2,}\b")


def importance(text: str) -> float:
    ents = len(set(_CAP.findall(text)))
    nums = len(_NUM.findall(text))
    length = min(len(text) / 200.0, 1.0)
    return 1.0 + 0.15 * min(ents, 4) + 0.1 * min(nums, 3) + 0.2 * length


def rrf(a, b, k=60):
    sc = defaultdict(float)
    for r, i in enumerate(a, 1):
        sc[i] += 1.0 / (k + r)
    for r, i in enumerate(b, 1):
        sc[i] += 1.0 / (k + r)
    return sorted(sc, key=lambda x: sc[x], reverse=True)


def mmr(cand, vn, rel_rank, k, lam=0.7):
    """Reselect from cand (idx) balancing relevance-rank and diversity."""
    rel = {c: 1.0 - rel_rank.index(c) / max(len(rel_rank), 1) for c in cand}
    chosen, pool = [], list(cand)
    while pool and len(chosen) < k:
        if not chosen:
            best = max(pool, key=lambda c: rel[c])
        else:
            def score(c):
                sim = max(float(vn[c] @ vn[j]) for j in chosen)
                return lam * rel[c] - (1 - lam) * sim
            best = max(pool, key=score)
        chosen.append(best); pool.remove(best)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmarks/data/lme/longmemeval_s")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("need OPENAI_API_KEY"); return 1
    data = json.load(open(args.data, encoding="utf-8"))
    order_t = ["temporal-reasoning", "knowledge-update", "multi-session",
               "single-session-preference", "single-session-assistant", "single-session-user"]
    by = defaultdict(list)
    for q in data:
        if any(t.get("has_answer") for s in q["haystack_sessions"] for t in s if isinstance(t, dict)):
            by[q["question_type"]].append(q)
    per = max(1, args.n // len(order_t))
    sample = [q for qt in order_t for q in by.get(qt, [])[:per]][: args.n]

    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

    STR = ["rerank_rrf", "mmr", "importance"]
    hit = {s: {k: [] for k in KS} for s in STR}
    cat = defaultdict(lambda: {s: [] for s in STR})

    for qi, q in enumerate(sample):
        turns = [(f"{t.get('role','user')}: {t.get('content','')[:6000]}", bool(t.get("has_answer")))
                 for s in q["haystack_sessions"] for t in s if isinstance(t, dict)]
        gold = {i for i, (_, g) in enumerate(turns) if g}
        if not gold:
            continue
        texts = [t for t, _ in turns]
        vecs = np.asarray(embed.encode_batch(texts), dtype=np.float32)
        vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        qv = np.asarray(embed.encode(q["question"]), dtype=np.float32); qv /= (np.linalg.norm(qv) + 1e-9)
        cos = vn @ qv
        dense_order = list(np.argsort(-cos))
        dpool = dense_order[:POOL]
        ce_ord = [dpool[i] for i in np.argsort(-np.asarray(
            ce.predict([(q["question"], texts[j]) for j in dpool], show_progress_bar=False)))]
        rrf_order = rrf(dpool, ce_ord)
        imp = np.array([importance(t) for t in texts])
        imp_order = list(np.argsort(-(cos * imp)))
        orders = {"rerank_rrf": rrf_order,
                  "mmr": mmr(rrf_order[:30], vn, rrf_order, max(KS)),
                  "importance": imp_order}
        for s in STR:
            for k in KS:
                h = len(set(orders[s][:k]) & gold) / len(gold)
                hit[s][k].append(h)
                if k == 10:
                    cat[q["question_type"]][s].append(h)
        if (qi + 1) % 10 == 0:
            print(f"  ...{qi+1}/{len(sample)}", flush=True)

    n = len(hit["rerank_rrf"][10])
    print(f"\n=== SALIENCE LAB (LME-S, {n} Q) ===")
    print(f"{'strategy':12}" + "".join(f"  hit@{k:<3}" for k in KS))
    base = np.mean(hit["rerank_rrf"][10])
    for s in STR:
        print(f"{s:12}" + "".join(f"  {np.mean(hit[s][k]):.3f} " for k in KS)
              + f"   ({np.mean(hit[s][10])-base:+.3f} vs rrf@10)")
    print("\nmulti-session hit@10:", " ".join(f"{s}={np.mean(cat['multi-session'][s]):.3f}" for s in STR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
