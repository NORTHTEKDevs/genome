"""Lean HyDE / multi-query retrieval test (ONE cross-encoder, small scope, to avoid
the machine-crash we hit loading two). Fitness = gold-evidence hit@k. Keep only what
beats the rerank baseline (~0.767 hit@10).

Strategies:
  dense               : cosine top-k
  rerank              : dense pool -> ms-marco-MiniLM-L-6 rerank (current default)
  hyde_rerank         : LLM hypothetical answer -> embed -> retrieve -> rerank
  multiquery_rerank   : LLM 3 paraphrases -> union pools -> rerank
  hyde+mq_rerank      : union of dense + hyde + paraphrase pools -> rerank

Run: .venv/Scripts/python.exe benchmarks/retrieval_hyde.py [--convs 1]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from anthropic import Anthropic

from genome.embeddings import EmbeddingProvider
from genome.evals.locomo import load_locomo

HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}
KS = [5, 10, 20]
POOL = 50
MODEL = "claude-haiku-4-5-20251001"
client = Anthropic()


def haiku(p, mt=180):
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
    ap.add_argument("--convs", type=int, default=1)
    args = ap.parse_args()
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need both keys"); return 1
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    convs = load_locomo("benchmarks/data/locomo10.json")[: args.convs]

    print("loading cross-encoder (L-6)...", flush=True)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

    strat = ["dense", "rerank", "hyde_rerank", "multiquery_rerank", "hyde+mq_rerank"]
    hit = {s: {k: [] for k in KS} for s in strat}
    nq = 0

    def topk(vn, qv, k):
        return list(np.argsort(-(vn @ qv)))[:k]

    def rr(pool_idx, query, texts):
        sc = ce.predict([(query, texts[i]) for i in pool_idx], show_progress_bar=False)
        return [pool_idx[i] for i in np.argsort(-np.asarray(sc))]

    for conv in convs:
        texts = [(f"[{t.session_datetime}] {t.speaker}: {t.text}"
                  if t.session_datetime else f"{t.speaker}: {t.text}") for t in conv.turns]
        dias = [t.dia_id for t in conv.turns]
        vn = np.asarray(embed.encode_batch(texts), dtype=np.float32)
        vn = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)
        qs = [q for q in conv.questions
              if q.category in HEADLINE and {str(e).strip() for e in q.evidence if str(e).strip()}]
        qvecs = np.asarray(embed.encode_batch([q.question for q in qs]), dtype=np.float32)
        qvecs = qvecs / (np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9)
        print(f"  {conv.conversation_id}: {len(qs)} q — generating HyDE...", flush=True)
        hyde = [haiku(f"Write a plausible 1-2 sentence answer to this question as if recalling "
                      f"it from a conversation. Q: {q.question}") for q in qs]
        hv = np.asarray(embed.encode_batch([hyde[i] or qs[i].question for i in range(len(qs))]),
                        dtype=np.float32)
        hv = hv / (np.linalg.norm(hv, axis=1, keepdims=True) + 1e-9)
        print("    generating multi-query...", flush=True)
        mq = [haiku(f"Rewrite as 3 differently-worded search queries, one per line, no numbering. "
                    f"Q: {q.question}") for q in qs]

        for qi, q in enumerate(qs):
            ev = {str(e).strip() for e in q.evidence if str(e).strip()}
            nq += 1
            dpool = topk(vn, qvecs[qi], POOL)
            hpool = topk(vn, hv[qi], POOL)
            variants = [v.strip() for v in (mq[qi] or "").splitlines() if v.strip()][:3]
            mqunion = list(dict.fromkeys(dpool[:20]))
            if variants:
                vv = np.asarray(embed.encode_batch(variants), dtype=np.float32)
                vv = vv / (np.linalg.norm(vv, axis=1, keepdims=True) + 1e-9)
                for j in range(len(variants)):
                    for idx in topk(vn, vv[j], 20):
                        if idx not in mqunion:
                            mqunion.append(idx)
            allunion = list(dict.fromkeys(dpool[:20] + hpool[:20] + mqunion))
            orders = {
                "dense": dpool,
                "rerank": rr(dpool, q.question, texts),
                "hyde_rerank": rr(hpool, q.question, texts),
                "multiquery_rerank": rr(mqunion, q.question, texts),
                "hyde+mq_rerank": rr(allunion, q.question, texts),
            }
            for s in strat:
                for k in KS:
                    hit[s][k].append(len({dias[i] for i in orders[s][:k]} & ev) / len(ev))

    print(f"\n=== HyDE/MULTI-QUERY LAB: {len(convs)} convs, {nq} evidence Q ===")
    base = np.mean(hit["rerank"][10])
    print(f"{'strategy':18}" + "".join(f"  hit@{k:<3}" for k in KS) + "   vs rerank@10")
    for s in strat:
        print(f"{s:18}" + "".join(f"  {np.mean(hit[s][k]):.3f} " for k in KS)
              + f"   {np.mean(hit[s][10])-base:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
