"""Retrieval-quality improvement lab (free feedback loop: embeddings + LOCAL rerank,
no answer/judge LLM calls). Optimizes hit-rate@k against LoCoMo gold evidence.

Baselines to beat: dense 0.715 hit@10, hybrid 0.564 (currently WORSE than dense -- a bug).

Strategies measured:
  dense              : plain cosine top-k (the current default)
  hybrid_fixed       : BM25 (stopword-stripped) + dense via WEIGHTED RRF, sparse capped
  rerank             : dense pool -> cross-encoder rerank -> top-k
  hybrid_rerank      : hybrid_fixed pool -> cross-encoder rerank -> top-k

Run: .venv/Scripts/python.exe benchmarks/retrieval_improve.py [--convs 5]
"""
from __future__ import annotations

import argparse
import os
import re
import string
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from genome.embeddings import EmbeddingProvider
from genome.evals.locomo import load_locomo

HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}
KS = [5, 10, 20]
POOL = 50
_PUNCT = str.maketrans("", "", string.punctuation)
_STOP = set(ENGLISH_STOP_WORDS) | {"speaker", "said", "im", "ill", "ive"}


def sharp_tokenize(text: str) -> list[str]:
    toks = text.lower().translate(_PUNCT).split()
    return [t for t in toks if t and t not in _STOP and not t.isdigit() or (t.isdigit() and len(t) >= 4)]


def weighted_rrf(dense_ids, sparse_ids, w_dense=2.0, w_sparse=1.0, k=60):
    score = defaultdict(float)
    for r, i in enumerate(dense_ids, 1):
        score[i] += w_dense / (k + r)
    for r, i in enumerate(sparse_ids, 1):
        score[i] += w_sparse / (k + r)
    return sorted(score, key=lambda x: score[x], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=5)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("need OPENAI_API_KEY"); return 1
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    convs = load_locomo("benchmarks/data/locomo10.json")[: args.convs]

    print("loading cross-encoder (local, one-time download ~80MB)...", flush=True)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

    strategies = ["dense", "hybrid_fixed", "rerank", "hybrid_rerank"]
    hit = {s: {k: [] for k in KS} for s in strategies}
    nq = 0

    for conv in convs:
        texts, dias = [], []
        for t in conv.turns:
            c = (f"[{t.session_datetime}] {t.speaker}: {t.text}"
                 if t.session_datetime else f"{t.speaker}: {t.text}")
            texts.append(c); dias.append(t.dia_id)
        vecs = np.asarray(embed.encode_batch(texts), dtype=np.float32)
        vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        bm25 = BM25Okapi([sharp_tokenize(t) for t in texts])

        qs = [q for q in conv.questions
              if q.category in HEADLINE and {str(e).strip() for e in q.evidence if str(e).strip()}]
        qvecs = np.asarray(embed.encode_batch([q.question for q in qs]), dtype=np.float32)  # batch!
        qvecs = qvecs / (np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9)
        print(f"  {conv.conversation_id}: {len(qs)} q, reranking...", flush=True)

        for qi, q in enumerate(qs):
            ev = {str(e).strip() for e in q.evidence if str(e).strip()}
            nq += 1
            qv = qvecs[qi]
            cos = vn @ qv
            dense_order = list(np.argsort(-cos))            # all idx, best first
            dense_pool = dense_order[:POOL]

            bm = bm25.get_scores(sharp_tokenize(q.question))
            sparse_order = list(np.argsort(-bm))[:20]        # sharp: only top-20 lexical

            hybrid_order = weighted_rrf([dense_order[i] for i in range(len(dense_order))][:POOL],
                                        sparse_order)

            # cross-encoder rerank over the dense pool / hybrid pool
            def rerank(pool_idx):
                pairs = [(q.question, texts[i]) for i in pool_idx]
                scores = ce.predict(pairs, show_progress_bar=False)
                return [pool_idx[i] for i in np.argsort(-np.asarray(scores))]

            hyb_pool = hybrid_order[:POOL]
            orders = {
                "dense": dense_pool,
                "hybrid_fixed": hyb_pool,
                "rerank": rerank(dense_pool),
                "hybrid_rerank": rerank(hyb_pool),
            }
            for s in strategies:
                for k in KS:
                    got = {dias[i] for i in orders[s][:k]}
                    hit[s][k].append(len(got & ev) / len(ev))

    print(f"\n=== RETRIEVAL HIT-RATE over {len(convs)} convs, {nq} evidence Q "
          f"(free: embeddings + local cross-encoder) ===")
    print(f"{'strategy':16}" + "".join(f"  hit@{k:<3}" for k in KS))
    for s in strategies:
        row = f"{s:16}"
        for k in KS:
            row += f"  {np.mean(hit[s][k]):.3f} "
        print(row)
    base = np.mean(hit["dense"][10])
    best = max(strategies, key=lambda s: np.mean(hit[s][10]))
    print(f"\ndense hit@10 baseline = {base:.3f}; best = {best} "
          f"{np.mean(hit[best][10]):.3f} ({np.mean(hit[best][10])-base:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
