"""Retrieval lab v2: push hit-rate past the rerank baseline (0.79 hit@10) with the
strongest known recall levers, measured against LoCoMo gold evidence.

Strategies (all reranked with a cross-encoder unless noted):
  dense                : cosine top-k (old default, ~0.73)
  rerank_L6            : dense pool -> ms-marco-MiniLM-L-6 rerank (current, ~0.79)
  rerank_L12           : stronger cross-encoder (ms-marco-MiniLM-L-12)
  hyde_rerank          : LLM writes a hypothetical answer -> embed it -> retrieve -> rerank
  multiquery_rerank    : LLM writes 3 query paraphrases -> union their pools -> rerank

Free levers (dense, rerank_L6/L12) cost only embeddings + local CE. HyDE/multi-query
cost 1 cheap LLM call/query. Keep whatever measurably wins; discard the rest.

Run: .venv/Scripts/python.exe benchmarks/retrieval_improve2.py [--convs 3]
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
from anthropic import Anthropic

from genome.embeddings import EmbeddingProvider
from genome.evals.locomo import load_locomo

HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}
KS = [5, 10, 20]
POOL = 50
MODEL = "claude-haiku-4-5-20251001"
client = Anthropic()


def haiku(p, mt=200):
    import time as _t
    for a in range(4):
        try:
            r = client.messages.create(model=MODEL, max_tokens=mt, temperature=0.0,
                                       messages=[{"role": "user", "content": p}])
            return (r.content[0].text if r.content else "") or ""
        except Exception:
            if a == 3:
                return ""
            _t.sleep(2 ** a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=3)
    args = ap.parse_args()
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need both keys"); return 1
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    convs = load_locomo("benchmarks/data/locomo10.json")[: args.convs]

    print("loading cross-encoders...", flush=True)
    from sentence_transformers import CrossEncoder
    ce6 = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    ce12 = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=256)

    strategies = ["dense", "rerank_L6", "rerank_L12", "hyde_rerank", "multiquery_rerank"]
    hit = {s: {k: [] for k in KS} for s in strategies}
    nq = 0

    def topk_by_vec(vn, qv, k):
        return list(np.argsort(-(vn @ qv)))[:k]

    for conv in convs:
        texts, dias = [], []
        for t in conv.turns:
            c = (f"[{t.session_datetime}] {t.speaker}: {t.text}"
                 if t.session_datetime else f"{t.speaker}: {t.text}")
            texts.append(c); dias.append(t.dia_id)
        vecs = np.asarray(embed.encode_batch(texts), dtype=np.float32)
        vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

        qs = [q for q in conv.questions
              if q.category in HEADLINE and {str(e).strip() for e in q.evidence if str(e).strip()}]
        qvecs = np.asarray(embed.encode_batch([q.question for q in qs]), dtype=np.float32)
        qvecs = qvecs / (np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9)
        print(f"  {conv.conversation_id}: {len(qs)} q", flush=True)

        # HyDE + multi-query need LLM text -> embed in batches per conv
        hyde_texts = [haiku(f"Write a plausible 1-2 sentence answer to this question, as if "
                            f"recalling it from a conversation. Question: {q.question}") for q in qs]
        hv = np.asarray(embed.encode_batch([h or qs[i].question for i, h in enumerate(hyde_texts)]),
                        dtype=np.float32)
        hv = hv / (np.linalg.norm(hv, axis=1, keepdims=True) + 1e-9)
        mq_raw = [haiku(f"Rewrite this question as 3 differently-worded search queries, one per "
                        f"line, no numbering. Question: {q.question}") for q in qs]

        def rr(ce, pool_idx, query):
            pairs = [(query, texts[i]) for i in pool_idx]
            sc = ce.predict(pairs, show_progress_bar=False)
            return [pool_idx[i] for i in np.argsort(-np.asarray(sc))]

        for qi, q in enumerate(qs):
            ev = {str(e).strip() for e in q.evidence if str(e).strip()}
            nq += 1
            dense_pool = topk_by_vec(vn, qvecs[qi], POOL)
            orders = {"dense": dense_pool,
                      "rerank_L6": rr(ce6, dense_pool, q.question),
                      "rerank_L12": rr(ce12, dense_pool, q.question)}
            # HyDE: retrieve with hypothetical-answer vector, rerank on the real question
            hyde_pool = topk_by_vec(vn, hv[qi], POOL)
            orders["hyde_rerank"] = rr(ce6, hyde_pool, q.question)
            # multi-query: union of pools from each paraphrase + original
            variants = [v.strip() for v in (mq_raw[qi] or "").splitlines() if v.strip()][:3]
            union = list(dict.fromkeys(dense_pool[:20]))
            if variants:
                vv = np.asarray(embed.encode_batch(variants), dtype=np.float32)
                vv = vv / (np.linalg.norm(vv, axis=1, keepdims=True) + 1e-9)
                for j in range(len(variants)):
                    for idx in topk_by_vec(vn, vv[j], 20):
                        if idx not in union:
                            union.append(idx)
            orders["multiquery_rerank"] = rr(ce6, union, q.question)
            for s in strategies:
                for k in KS:
                    got = {dias[i] for i in orders[s][:k]}
                    hit[s][k].append(len(got & ev) / len(ev))

    print(f"\n=== RETRIEVAL LAB v2: {len(convs)} convs, {nq} evidence Q ===")
    print(f"{'strategy':18}" + "".join(f"  hit@{k:<3}" for k in KS))
    base = np.mean(hit["dense"][10])
    for s in strategies:
        row = f"{s:18}" + "".join(f"  {np.mean(hit[s][k]):.3f} " for k in KS)
        print(row + (f"   ({np.mean(hit[s][10])-base:+.3f} vs dense@10)"))
    best = max(strategies, key=lambda s: np.mean(hit[s][10]))
    print(f"\nBEST @10: {best} {np.mean(hit[best][10]):.3f} ({np.mean(hit[best][10])-base:+.3f} vs dense)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
