"""LoCoMo-Haystack: the honest long-history benchmark where a memory layer can
actually beat full-context.

All 10 LoCoMo conversations are ingested into ONE memory store (~250k tokens).
Every question is answered against that union:
  - GENOME: retrieve top-k (constant ~1.4k-token context) from the 250k store.
  - full-context @ budget B: the first B tokens of the concatenated store; when
    the 250k history exceeds B, the rest is DROPPED, so questions whose evidence
    fell outside the window fail. (No model can hold 250k in a 200k window --
    that's the whole point: real histories overflow.)
Sweep B and measure judged accuracy + context tokens/query. Constant-cost
retrieval vs budget-truncated full context = the crossover.

Checkpointed per question (this box kills long runs). Same judge/model/prompt
as the LoCoMo runs -> within-harness, comparable.

Run: .venv/Scripts/python.exe benchmarks/haystack.py [--n 120] [--budgets 4000,16000,64000]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import tiktoken
from anthropic import Anthropic

from genome.embeddings import EmbeddingProvider
from genome.evals.baselines import _format_turn_text
from genome.evals.llm_judge import judge_answer, preprocess_gold_mem0
from genome.evals.locomo import (
    ANSWER_PROMPT,
    LocomoConfig,
    _is_abstention,
    _sanitize_locomo_text,
    load_locomo,
    replay_conversation,
)
from genome.memory.facade import Memory

ENC = tiktoken.get_encoding("cl100k_base")
MODEL = "claude-haiku-4-5-20251001"
HEADLINE = {"multi-hop", "temporal", "open-domain", "single-hop"}
OUT = Path("results/haystack")
random.seed(20260711)
client = Anthropic()


def haiku(prompt, mt=256):
    r = client.messages.create(model=MODEL, max_tokens=mt, temperature=0.0,
                               messages=[{"role": "user", "content": prompt}])
    return (r.content[0].text if r.content else "") or ""


def answer(context, question):
    p = ANSWER_PROMPT.format(context=context or "(no relevant memories)",
                             question=_sanitize_locomo_text(question))
    return haiku(p).strip()


def judge(cat, gold, predicted):
    if cat == "adversarial":
        return "CORRECT" if _is_abstention(predicted) else "INCORRECT"
    g = preprocess_gold_mem0(cat, gold)
    return judge_answer(lambda p: haiku(p), "", g, predicted, mode="mem0").label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--budgets", default="4000,16000,64000")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need both keys"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    ck = OUT / "answers.jsonl"

    convs = load_locomo("benchmarks/data/locomo10.json")

    # Build the union GENOME store (dense + parent filter, no extraction).
    # Batch-embed ALL turns at once (~3 API calls) instead of one-per-turn --
    # the same stored content/metadata as replay_conversation, 100x faster to
    # ingest 5,882 turns. Every turn shares user_id='hay'; searching with
    # agent_id omitted spans ALL conversations (the union store).
    print("ingesting all 10 conversations into ONE store (batched)...", flush=True)
    from genome.memory.schema import MemoryRecord
    import numpy as _np
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    mem = Memory(storage=":memory:", embedding_provider=embed)
    turns_meta, texts = [], []
    for c in convs:
        for t in c.turns:
            content = (f"[{t.session_datetime}] {t.speaker}: {t.text}"
                       if t.session_datetime else f"{t.speaker}: {t.text}")
            texts.append(content)
            turns_meta.append((content, c.conversation_id, t))
    vecs = embed.encode_batch(texts)
    for (content, cid, t), v in zip(turns_meta, vecs):
        mem.store.add(MemoryRecord(
            content=content, embedding=_np.asarray(v, dtype=_np.float32),
            user_id="hay", agent_id=cid,
            metadata={"turn_id": t.turn_id, "dia_id": t.dia_id,
                      "speaker": t.speaker, "session": t.session,
                      "session_datetime": t.session_datetime}))
    total_store_tokens = sum(
        len(ENC.encode(f"[{t.session_datetime}] {t.speaker}: {t.text}"))
        for c in convs for t in c.turns
    )
    print(f"  store ~{total_store_tokens:,} tokens across all conversations", flush=True)

    # Concatenated transcript (chronological by conversation) for full-context.
    transcript_ids = ENC.encode("\n".join(
        f"- {_sanitize_locomo_text(_format_turn_text(t))}"
        for c in convs for t in c.turns
    ))
    print(f"  full concatenated transcript = {len(transcript_ids):,} tokens "
          f"(exceeds a 200k window: full-context CANNOT hold it all)", flush=True)

    # Sample questions across conversations (headline categories only).
    pool = [(c.conversation_id, q) for c in convs for q in c.questions
            if q.category in HEADLINE]
    random.shuffle(pool)
    sample = pool[: args.n]

    done = set()
    if ck.exists():
        for line in ck.open():
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                pass

    with ck.open("a") as fh:
        for i, (cid, q) in enumerate(sample):
            qid = f"{cid}:{q.question_id or q.question[:40]}"
            if qid in done:
                continue
            row = {"qid": qid, "cid": cid, "category": q.category,
                   "question": q.question, "gold": q.answer}
            # GENOME: retrieve from the union store (search spans all convs since
            # every turn shares user_id='hay'); agent_id omitted -> whole scope.
            g = mem.search(q.question, user_id="hay", limit=30,
                           filter_parents=True, mode="dense")
            gctx = "\n".join(f"- {_sanitize_locomo_text(r.content)}" for r in g)
            gpred = answer(gctx, q.question)
            row["genome"] = {"label": judge(q.category, q.answer, gpred),
                             "ctx_tokens": len(ENC.encode(gctx))}
            # full-context @ each budget (first-B tokens of the 250k store)
            row["fullctx"] = {}
            for B in budgets:
                ctx = ENC.decode(transcript_ids[:B])
                pred = answer(ctx, q.question)
                row["fullctx"][str(B)] = {
                    "label": judge(q.category, q.answer, pred),
                    "ctx_tokens": min(B, len(transcript_ids)),
                }
            fh.write(json.dumps(row) + "\n"); fh.flush()
            if (i + 1) % 10 == 0:
                print(f"  ...{i+1}/{len(sample)}", flush=True)
    mem.close()
    print("DONE. Run benchmarks/haystack_report.py for the crossover table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
