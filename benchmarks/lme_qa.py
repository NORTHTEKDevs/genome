"""LongMemEval answer-accuracy head-to-head: GENOME (dense / dense+rerank) vs Mem0, on
a harder-than-LoCoMo benchmark. Within-harness (same responder + judge + embedder).

Feasibility: each question's history is CAPPED to its gold (answer) sessions + a bounded
number of distractor sessions, and Mem0 ingests SESSION-BY-SESSION (one infer=True add()
per session, not per turn) so its LLM cost stays bounded. This keeps a real retrieval
challenge (distractors present) while making the Mem0 comparison runnable.

Metrics: answer accuracy per question_type + retrieval hit-rate (gold = has_answer turns).
Abstention questions (question_id endswith '_abs') require the system to say it doesn't know.

Run: .venv/Scripts/python.exe benchmarks/lme_qa.py [--n 30] [--distractors 6]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict

import numpy as np
from anthropic import Anthropic

from genome.embeddings import EmbeddingProvider
from genome.evals.llm_judge import judge_answer, preprocess_gold_mem0
from genome.evals.locomo import ANSWER_PROMPT, _is_abstention, _sanitize_locomo_text
from genome.memory.facade import Memory
from genome.memory.rerank import CrossEncoderReranker
from genome.memory.schema import MemoryRecord

MODEL = "claude-haiku-4-5-20251001"          # Mem0 extraction + judge
RESP = "claude-sonnet-5"                       # RESPONDER: stronger, to un-bottleneck answers
OUT = "results/lme_qa/answers_pow.jsonl"        # powered run: Sonnet responder
client = Anthropic()


def _call(model, p, mt=256):
    import time as _t
    kw = {"model": model, "max_tokens": mt, "messages": [{"role": "user", "content": p}]}
    if not model.startswith("claude-sonnet-5"):   # sonnet-5 deprecates temperature
        kw["temperature"] = 0.0
    for a in range(5):
        try:
            r = client.messages.create(**kw)
            return "".join(getattr(b, "text", "") for b in (r.content or [])
                           if getattr(b, "type", "") == "text").strip()
        except Exception:
            if a == 4:
                raise
            _t.sleep(2 ** a)


def haiku(p, mt=256):
    return _call(MODEL, p, mt)


def answer(ctx, q):
    # generous budget: sonnet-5 spends output tokens on reasoning before the answer
    return _call(RESP, ANSWER_PROMPT.format(context=ctx or "(no relevant memories)",
                                            question=_sanitize_locomo_text(q)), 1500).strip()


def judge(is_abs, gold, pred):
    if is_abs:
        return "CORRECT" if _is_abstention(pred) else "INCORRECT"
    return judge_answer(lambda p: haiku(p), "", preprocess_gold_mem0("", str(gold)),
                        pred, mode="mem0").label


def capped_sessions(q, n_distractor):
    """Return list of sessions = gold sessions + up to n_distractor distractors,
    each session a list of turn dicts. Preserves chronological order by haystack index."""
    gold_ids = set(q["answer_session_ids"])
    sessions = list(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    gold = [(sid, s) for sid, s in sessions if sid in gold_ids]
    distract = [(sid, s) for sid, s in sessions if sid not in gold_ids][:n_distractor]
    keep = gold + distract
    # keep original order
    order = {sid: i for i, (sid, _) in enumerate(sessions)}
    keep.sort(key=lambda x: order[x[0]])
    return [s for _, s in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--distractors", type=int, default=4)
    args = ap.parse_args()
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need both keys"); return 1
    data = json.load(open("benchmarks/data/lme/longmemeval_s", encoding="utf-8"))
    order = ["temporal-reasoning", "knowledge-update", "multi-session",
             "single-session-preference", "single-session-assistant", "single-session-user"]
    by = defaultdict(list)
    for q in data:
        by[q["question_type"]].append(q)
    per = max(1, args.n // len(order))
    sample = [q for qt in order for q in by.get(qt, [])[:per]][: args.n]

    os.makedirs("results/lme_qa", exist_ok=True)
    ck = OUT
    done = set()
    if os.path.exists(ck):
        for l in open(ck):
            try:
                d = json.loads(l); done.add((d["system"], d["qid"]))
            except Exception:
                pass

    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    rr = CrossEncoderReranker()
    from mem0 import Memory as Mem0
    shutil.rmtree("/tmp/qdrant", ignore_errors=True)
    m0 = Mem0.from_config({
        "llm": {"provider": "anthropic", "config": {"model": MODEL, "temperature": 0.0}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}}})

    fh = open(ck, "a")
    for qi, q in enumerate(sample):
        qid = q["question_id"]
        is_abs = qid.endswith("_abs")
        sess = capped_sessions(q, args.distractors)
        turns = [(f"{t.get('role','user')}: {t.get('content','')[:6000]}", bool(t.get("has_answer")))
                 for s in sess for t in s if isinstance(t, dict)]
        gold_turns = {i for i, (_, g) in enumerate(turns) if g}
        texts = [t for t, _ in turns]

        # GENOME store (dense; rerank shares it)
        need_g = any((s, qid) not in done for s in ("genome_dense", "genome_rerank"))
        if need_g:
            vecs = embed.encode_batch(texts)
            g = Memory(storage=":memory:", embedding_provider=embed)
            for txt, v in zip(texts, vecs):
                g.store.add(MemoryRecord(content=txt, embedding=np.asarray(v, dtype=np.float32),
                                        user_id="u"))
        # Mem0 ingest session-by-session (bounded LLM calls)
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

        # answer with each system
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
            else:  # mem0
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

    # report
    rows = [json.loads(l) for l in open(ck) if l.strip()]
    systems = ["genome_dense", "genome_rerank", "mem0"]
    print(f"\n=== LongMemEval-S QA head-to-head ({len({r['qid'] for r in rows})} Q, "
          f"capped gold+{args.distractors} distractors) ===")
    print(f"{'system':16} {'accuracy':>9} {'hit@10':>8}")
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        acc = sum(r["label"] == "CORRECT" for r in rs) / len(rs) if rs else 0
        hrs = [r["hit"] for r in rs if r.get("hit") is not None]
        hr = sum(hrs) / len(hrs) if hrs else float("nan")
        print(f"{s:16} {acc:>9.3f} {hr:>8.3f}")
    print("\naccuracy per question_type:")
    types = sorted({r["qtype"] for r in rows})
    print(f"{'type':26}" + "".join(f"{s.split('_')[-1]:>12}" for s in systems))
    for t in types:
        line = f"{t:26}"
        for s in systems:
            rs = [r for r in rows if r["system"] == s and r["qtype"] == t]
            line += f"{(sum(r['label']=='CORRECT' for r in rs)/len(rs) if rs else 0):>12.3f}"
        print(line)

    # McNemar: GENOME vs Mem0, paired per question
    import math
    lab = defaultdict(dict)
    for r in rows:
        lab[r["qid"]][r["system"]] = r["label"]
    print("\nMcNemar (GENOME vs Mem0, paired):")
    for g in ("genome_dense", "genome_rerank"):
        pr = [(lab[q].get(g), lab[q].get("mem0")) for q in lab
              if lab[q].get(g) and lab[q].get("mem0")]
        go = sum(a == "CORRECT" and b != "CORRECT" for a, b in pr)
        mo = sum(b == "CORRECT" and a != "CORRECT" for a, b in pr)
        d = go + mo
        p = math.erfc(math.sqrt(((abs(go - mo) - 1) ** 2 / d) / 2)) if d else 1.0
        verdict = "GENOME WINS" if go > mo and p < 0.05 else ("leads" if go > mo else "no")
        print(f"  {g:14} vs mem0  n={len(pr)}  {g}-only {go}  mem0-only {mo}  p={p:.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
