"""Run the TempBelief benchmark: GENOME belief-state vs full-context, dense retrieval,
and Mem0 (infer=True + history traversal). Within-harness: identical responder
(Haiku 4.5), judge (mem0-verbatim), embedder. Only {context} varies (retrieval top-k
differs slightly: dense 30, mem0 20, belief raw-fallback 15 -- not directional, dense
has the most hits and scores worst on as-of).

Splits: current-value, as-of, history (headline) + as-of-abstention (diagnostic).
Checkpointed per (system, qid). Also measures GENOME belief-fact extraction recall
against the ground-truth event log (kill-criterion #4: don't trust QA if extraction
is broken).

Run: .venv/Scripts/python.exe benchmarks/tempbelief_run.py [--convs 12] [--smoke]
     [--systems belief,full-context,dense,mem0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
from anthropic import Anthropic

from genome.embeddings import EmbeddingProvider
from genome.evals.llm_judge import judge_answer, preprocess_gold_mem0
from genome.evals.locomo import ANSWER_PROMPT, _is_abstention, _sanitize_locomo_text
from genome.evals.tempbelief import load
from genome.memory.belief import answer_belief_context, ingest_belief_turn, parse_when
from genome.memory.facade import Memory

MODEL = "claude-haiku-4-5-20251001"
HEADLINE = {"current-value", "as-of", "history"}
OUT = Path("results/tempbelief")
client = Anthropic()


def haiku(p, mt=400):
    import time as _t
    for a in range(5):
        try:
            r = client.messages.create(model=MODEL, max_tokens=mt, temperature=0.0,
                                       messages=[{"role": "user", "content": p}])
            return (r.content[0].text if r.content else "") or ""
        except Exception:
            if a == 4:
                raise
            _t.sleep(2 ** a)


def answer(ctx, q):
    return haiku(ANSWER_PROMPT.format(context=ctx or "(no relevant memories)",
                                      question=_sanitize_locomo_text(q))).strip()


def judge(cat, gold, pred):
    if cat == "as-of-abstention":
        return "CORRECT" if _is_abstention(pred) else "INCORRECT"
    return judge_answer(lambda p: haiku(p), "", preprocess_gold_mem0(cat, gold),
                        pred, mode="mem0").label


def turn_line(t):
    return f"[{t.session_datetime}] {t.speaker}: {t.text}"


# ---- per-system ingest -> returns a state; and context(state, question) ----
def build_fullcontext(conv, embed):
    return "\n".join(f"- {_sanitize_locomo_text(turn_line(t))}" for t in conv.turns)

def build_dense(conv, embed):
    m = Memory(storage=":memory:", embedding_provider=embed)   # IdentityExtractor, no LLM
    for t in conv.turns:
        m.add(turn_line(t), user_id="tb")
    return m

def build_belief(conv, embed):
    m = Memory(storage=":memory:", embedding_provider=embed,
               llm_call=lambda p: haiku(p))
    recorded = 0
    for t in conv.turns:
        st = parse_when(t.session_datetime, time.time())
        recorded += ingest_belief_turn(m, t.text, session_time=st, user_id="tb", llm=lambda p: haiku(p))
    return {"mem": m, "recorded": recorded}

_MEM0_CLIENT = None  # one client for the whole run; convs isolated by user_id

def _mem0_client():
    global _MEM0_CLIENT
    if _MEM0_CLIENT is None:
        import shutil
        from mem0 import Memory as Mem0
        shutil.rmtree("/tmp/qdrant", ignore_errors=True)
        _MEM0_CLIENT = Mem0.from_config({
            "llm": {"provider": "anthropic", "config": {"model": MODEL, "temperature": 0.0}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}}})
    return _MEM0_CLIENT

def build_mem0(conv, embed):
    m = _mem0_client()
    uid = f"tb_{conv.conversation_id}"
    for t in conv.turns:
        role = "user" if t.speaker == conv.speaker_a else "assistant"
        try:
            m.add([{"role": role, "content": turn_line(t)}], user_id=uid, infer=True)
        except Exception:
            pass
    return {"client": m, "uid": uid}


def ctx_fullcontext(state, q):
    return state

def ctx_dense(state, q):
    hits = state.search(q.question, user_id="tb", limit=30, mode="dense")
    return "\n".join(f"- {_sanitize_locomo_text(h.content)}" for h in hits)

def ctx_belief(state, q):
    c = answer_belief_context(state["mem"], q.question, user_id="tb", llm=lambda p: haiku(p))
    if c:
        return c
    # belt-and-suspenders: fall back to dense over the raw store if KG miss
    hits = state["mem"].search(q.question, user_id="tb", limit=15, mode="dense")
    return "\n".join(f"- {_sanitize_locomo_text(h.content)}" for h in hits)

def ctx_mem0(state, q):
    client, uid = state["client"], state["uid"]
    parts = []
    try:
        res = client.search(q.question, top_k=20, filters={"user_id": uid})
        items = res.get("results", res) if isinstance(res, dict) else res
        for it in (items or []):
            mem = it.get("memory") or it.get("text") or ""
            parts.append(f"- {mem}")
            # fair: give Mem0 its own event log (history) for the matched memory
            mid = it.get("id")
            if mid:
                try:
                    for h in client.history(mid) or []:
                        pv, nv = h.get("old_memory"), h.get("new_memory")
                        ts = h.get("updated_at") or h.get("created_at") or ""
                        if nv:
                            parts.append(f"  (history: {pv} -> {nv} at {ts})")
                except Exception:
                    pass
    except Exception:
        pass
    return "\n".join(parts)


SYSTEMS = {
    "full-context": (build_fullcontext, ctx_fullcontext),
    "dense": (build_dense, ctx_dense),
    "belief": (build_belief, ctx_belief),
    "mem0": (build_mem0, ctx_mem0),
}


def mcnemar(rows, a, b):
    a_only = sum(r[a] == "CORRECT" and r[b] != "CORRECT" for r in rows)
    b_only = sum(r[b] == "CORRECT" and r[a] != "CORRECT" for r in rows)
    d = a_only + b_only
    p = math.erfc(math.sqrt(((abs(a_only - b_only) - 1) ** 2 / d) / 2)) if d else 1.0
    return a_only, b_only, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--systems", default="belief,full-context,dense,mem0")
    ap.add_argument("--maxq", type=int, default=0, help="cap questions per conv (smoke)")
    ap.add_argument("--data", default="benchmarks/data/tempbelief.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need both keys"); return 1
    systems = [s for s in args.systems.split(",") if s in SYSTEMS]
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = (args.tag or ("smoke" if args.smoke else "")) or ""
    ck = OUT / (f"answers_{suffix}.jsonl" if suffix else "answers.jsonl")
    rk = OUT / (f"recall_{suffix}.jsonl" if suffix else "recall.jsonl")

    convs = load(args.data)[: (2 if args.smoke else args.convs)]
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")

    done = set()
    if ck.exists():
        for l in ck.open():
            try:
                d = json.loads(l); done.add((d["system"], d["qid"]))
            except Exception:
                pass

    rec_fh = rk.open("a")
    with ck.open("a") as fh:
        for sysname in systems:
            build, ctxfn = SYSTEMS[sysname]
            for conv in convs:
                qs = [q for q in conv.questions]
                if args.maxq:
                    # keep a balanced slice across splits
                    bycat = defaultdict(list)
                    for q in qs:
                        bycat[q.category].append(q)
                    qs = [q for cat in bycat for q in bycat[cat][: args.maxq]]
                pending = [q for q in qs if (sysname, f"{conv.conversation_id}:{q.question_id}") not in done]
                if not pending:
                    continue
                t0 = time.time()
                state = build(conv, embed)
                ingest_s = time.time() - t0
                if sysname == "belief":
                    rec_fh.write(json.dumps({"conv": conv.conversation_id,
                        "recorded": state["recorded"], "gold_slots": len({(q.entity, q.attribute) for q in conv.questions}),
                        "ingest_s": round(ingest_s, 1)}) + "\n"); rec_fh.flush()
                for q in pending:
                    qid = f"{conv.conversation_id}:{q.question_id}"
                    ctx = ctxfn(state, q)
                    pred = answer(ctx, q.question)
                    label = judge(q.category, q.answer, pred)
                    fh.write(json.dumps({"system": sysname, "qid": qid,
                        "category": q.category, "label": label,
                        "ctx_tok": len(ctx.split())}) + "\n"); fh.flush()
                if hasattr(state, "close"):
                    state.close()
                elif isinstance(state, dict) and hasattr(state.get("mem"), "close"):
                    state["mem"].close()
                print(f"{sysname} {conv.conversation_id}: {len(pending)} q done", flush=True)
    rec_fh.close()

    # ---- report ----
    rows = [json.loads(l) for l in ck.open() if l.strip()]
    by = defaultdict(dict)  # qid -> {system: label, category}
    for r in rows:
        by[r["qid"]]["category"] = r["category"]
        by[r["qid"]][r["system"]] = r["label"]
    present = [s for s in systems]
    print(f"\n=== TempBelief ({'smoke' if args.smoke else 'full'}): per-split accuracy ===")
    splits = ["current-value", "as-of", "history", "as-of-abstention"]
    hdr = f"{'split':18}" + "".join(f"{s:>14}" for s in present)
    print(hdr)
    for split in splits:
        qids = [q for q, v in by.items() if v.get("category") == split]
        line = f"{split:18}"
        for s in present:
            vals = [by[q].get(s) for q in qids if by[q].get(s)]
            acc = sum(v == "CORRECT" for v in vals) / len(vals) if vals else float("nan")
            line += f"{acc:>13.3f} "
        print(line)
    # macro-J over 3 headline splits
    print("\nmacro-J (mean of 3 headline splits):")
    for s in present:
        accs = []
        for split in HEADLINE:
            qids = [q for q, v in by.items() if v.get("category") == split]
            vals = [by[q].get(s) for q in qids if by[q].get(s)]
            if vals:
                accs.append(sum(v == "CORRECT" for v in vals) / len(vals))
        print(f"  {s:14} {sum(accs)/len(accs):.3f}" if accs else f"  {s}: n/a")
    # McNemar belief vs each baseline, per headline split
    if "belief" in present:
        print("\n=== McNemar: belief vs baselines (headline splits) ===")
        for split in ["current-value", "as-of", "history"]:
            qids = [q for q, v in by.items() if v.get("category") == split]
            for b in present:
                if b == "belief":
                    continue
                pr = [{"belief": by[q].get("belief"), b: by[q].get(b)} for q in qids
                      if by[q].get("belief") and by[q].get(b)]
                if not pr:
                    continue
                bo, ao, p = mcnemar(pr, "belief", b)
                verdict = "belief WINS" if (bo > ao and p < 0.05) else ("lead" if bo > ao else "no")
                print(f"  {split:14} belief vs {b:13} belief-only {bo:3} {b}-only {ao:3} p={p:.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
