"""LongMemEval QA head-to-head via OpenRouter (LLM) + LOCAL embedder, for a
larger-n significance test when direct Anthropic/OpenAI keys aren't available.

Backend differs from lme_qa.py (Sonnet + OpenAI embeddings), so this is a FRESH,
self-consistent run -- NOT poolable with results/lme_qa/answers_pow.jsonl. Both
GENOME and Mem0 use the SAME OpenRouter model + the SAME local embedder, so the
paired GENOME-vs-Mem0 comparison and its McNemar significance test are valid.

Env: OPENROUTER_API_KEY required. Optional: OR_RESP (responder model),
OR_JUDGE (judge + Mem0 extraction model).

Run: .venv/Scripts/python.exe benchmarks/lme_qa_or.py [--n 210] [--distractors 4]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict

import numpy as np
from openai import OpenAI

from genome.embeddings import EmbeddingProvider
from genome.evals.llm_judge import judge_answer, preprocess_gold_mem0
from genome.evals.locomo import ANSWER_PROMPT, _is_abstention, _sanitize_locomo_text
from genome.memory.facade import Memory
from genome.memory.rerank import CrossEncoderReranker
from genome.memory.schema import MemoryRecord

OR_BASE = "https://openrouter.ai/api/v1"
RESP = os.environ.get("OR_RESP", "anthropic/claude-sonnet-4")       # responder
JUDGE = os.environ.get("OR_JUDGE", "anthropic/claude-haiku-4.5")     # judge + Mem0 extraction
OUT = "results/lme_qa/answers_or.jsonl"
_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OR_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    return _client


def _call(model, p, mt=512):
    for a in range(5):
        try:
            r = client().chat.completions.create(
                model=model, max_tokens=mt, temperature=0.0,
                messages=[{"role": "user", "content": p}])
            return (r.choices[0].message.content or "").strip()
        except Exception:
            if a == 4:
                raise
            time.sleep(2 ** a)


def judge_llm(p, mt=256):
    return _call(JUDGE, p, mt)


def answer(ctx, q):
    return _call(RESP, ANSWER_PROMPT.format(context=ctx or "(no relevant memories)",
                                            question=_sanitize_locomo_text(q)), 512).strip()


def judge(is_abs, gold, pred):
    if is_abs:
        return "CORRECT" if _is_abstention(pred) else "INCORRECT"
    return judge_answer(lambda p: judge_llm(p), "", preprocess_gold_mem0("", str(gold)),
                        pred, mode="mem0").label


def capped_sessions(q, n_distractor):
    gold_ids = set(q["answer_session_ids"])
    sessions = list(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    gold = [(sid, s) for sid, s in sessions if sid in gold_ids]
    distract = [(sid, s) for sid, s in sessions if sid not in gold_ids][:n_distractor]
    keep = gold + distract
    order = {sid: i for i, (sid, _) in enumerate(sessions)}
    keep.sort(key=lambda x: order[x[0]])
    return [s for _, s in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=210)
    ap.add_argument("--distractors", type=int, default=4)
    args = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("need OPENROUTER_API_KEY"); return 1
    # Route Mem0's OpenAI-provider LLM through OpenRouter via env (config-level base_url
    # is ignored by mem0's openai provider). Our own responder/judge client is explicit.
    os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = OR_BASE
    print(f"[cfg] responder={RESP}  judge={JUDGE}  embedder=local all-MiniLM (384d)", flush=True)

    data = json.load(open("benchmarks/data/lme/longmemeval_s", encoding="utf-8"))
    order = ["temporal-reasoning", "knowledge-update", "multi-session",
             "single-session-preference", "single-session-assistant", "single-session-user"]
    by = defaultdict(list)
    for q in data:
        by[q["question_type"]].append(q)
    per = max(1, args.n // len(order))
    sample = [q for qt in order for q in by.get(qt, [])[:per]][: args.n]

    os.makedirs("results/lme_qa", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                d = json.loads(l); done.add((d["system"], d["qid"]))
            except Exception:
                pass

    embed = EmbeddingProvider()   # LOCAL default (all-MiniLM), no API
    rr = CrossEncoderReranker()
    from mem0 import Memory as Mem0
    shutil.rmtree("/tmp/qdrant_or", ignore_errors=True)
    m0 = Mem0.from_config({
        "llm": {"provider": "openai", "config": {"model": JUDGE, "temperature": 0.0}},
        "embedder": {"provider": "huggingface", "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"}},
        "vector_store": {"provider": "qdrant", "config": {
            "embedding_model_dims": 384, "path": "/tmp/qdrant_or"}}})

    fh = open(OUT, "a")
    for qi, q in enumerate(sample):
        qid = q["question_id"]
        is_abs = qid.endswith("_abs")
        sess = capped_sessions(q, args.distractors)
        turns = [(f"{t.get('role','user')}: {t.get('content','')[:6000]}", bool(t.get("has_answer")))
                 for s in sess for t in s if isinstance(t, dict)]
        gold_turns = {i for i, (_, g) in enumerate(turns) if g}
        texts = [t for t, _ in turns]

        need_g = any((s, qid) not in done for s in ("genome_dense", "genome_rerank"))
        if need_g:
            vecs = embed.encode_batch(texts)
            g = Memory(storage=":memory:", embedding_provider=embed)
            for txt, v in zip(texts, vecs):
                g.store.add(MemoryRecord(content=txt, embedding=np.asarray(v, dtype=np.float32),
                                        user_id="u"))
        need_m = ("mem0", qid) not in done
        if need_m:
            uid = f"q{qi}"
            for s in sess:
                msgs = [{"role": t.get("role", "user"), "content": t.get("content", "")[:6000]}
                        for t in s if isinstance(t, dict)]
                try:
                    m0.add(msgs, user_id=uid, infer=True)
                except Exception:
                    pass

        def hitrate(idxs):
            return len(set(idxs) & gold_turns) / len(gold_turns) if gold_turns else 0.0

        for system in ("genome_dense", "genome_rerank", "mem0"):
            if (system, qid) in done:
                continue
            hr = None
            if system == "genome_dense":
                res = g.search(q["question"], user_id="u", limit=10, mode="dense")
                idxs = [texts.index(r.record.content) for r in res if r.record.content in texts]
                ctx = "\n".join(f"- {_sanitize_locomo_text(r.record.content)}" for r in res)
                hr = hitrate(idxs)
            elif system == "genome_rerank":
                res = g.search(q["question"], user_id="u", limit=10, mode="dense", reranker=rr)
                idxs = [texts.index(r.record.content) for r in res if r.record.content in texts]
                ctx = "\n".join(f"- {_sanitize_locomo_text(r.record.content)}" for r in res)
                hr = hitrate(idxs)
            else:
                parts = []
                try:
                    r = m0.search(q["question"], top_k=10, filters={"user_id": f"q{qi}"})
                    items = r.get("results", r) if isinstance(r, dict) else r
                    parts = [f"- {it.get('memory') or it.get('text') or ''}" for it in (items or [])]
                except Exception:
                    pass
                ctx = "\n".join(parts)
            pred = answer(ctx, q["question"])
            label = judge(is_abs, q["answer"], pred)
            fh.write(json.dumps({"system": system, "qid": qid, "qtype": q["question_type"],
                                 "is_abs": is_abs, "label": label, "hit": hr}) + "\n"); fh.flush()
        if need_g:
            g.close()
        print(f"  {qi+1}/{len(sample)} {q['question_type']}", flush=True)
    fh.close()

    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    systems = ["genome_dense", "genome_rerank", "mem0"]
    print(f"\n=== LongMemEval-S QA via OpenRouter ({len({r['qid'] for r in rows})} Q, "
          f"gold+{args.distractors} distractors, local embedder) ===")
    print(f"{'system':16} {'accuracy':>9} {'hit@10':>8}")
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        acc = sum(r["label"] == "CORRECT" for r in rs) / len(rs) if rs else 0
        hrs = [r["hit"] for r in rs if r.get("hit") is not None]
        hr = sum(hrs) / len(hrs) if hrs else float("nan")
        print(f"{s:16} {acc:>9.3f} {hr:>8.3f}")

    import math
    lab = defaultdict(dict)
    for r in rows:
        lab[r["qid"]][r["system"]] = r["label"]
    print("\nMcNemar (GENOME vs Mem0, paired):")
    for gsys in ("genome_dense", "genome_rerank"):
        pr = [(lab[q].get(gsys), lab[q].get("mem0")) for q in lab
              if lab[q].get(gsys) and lab[q].get("mem0")]
        go = sum(a == "CORRECT" and b != "CORRECT" for a, b in pr)
        mo = sum(b == "CORRECT" and a != "CORRECT" for a, b in pr)
        d = go + mo
        p = math.erfc(math.sqrt(((abs(go - mo) - 1) ** 2 / d) / 2)) if d else 1.0
        verdict = "GENOME WINS (sig)" if go > mo and p < 0.05 else ("leads" if go > mo else "no")
        print(f"  {gsys:14} vs mem0  n={len(pr)}  {gsys}-only {go}  mem0-only {mo}  p={p:.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
