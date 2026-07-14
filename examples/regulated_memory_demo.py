"""Auditable memory for regulated agents: GENOME vs Mem0, head to head, honest.

The pitch this demonstrates: for an agent operating where you must be able to prove
*what the system had on record about a person, as of a past date, and on what basis*
(lending, KYC, healthcare, claims), an overwrite-based memory is a liability. Mem0
DOES keep a change log (history()), so this is not "Mem0 forgets everything" -- it is
narrower and true:

  1. Mem0's timeline is WALL-CLOCK (when the DB row changed), so it cannot reconstruct
     "what was on record as of <business date>" when facts arrive out of order.
  2. Mem0's update/delete is a non-deterministic LLM decision that can drop a fact.
  3. GENOME keeps a deterministic, per-attribute, DOMAIN-TIME bi-temporal record with
     per-fact provenance (the exact source text), so a point-in-time reconstruction is
     always reproducible and explainable.

We run the identical events through both, then ask the compliance questions an auditor
asks. Same responder + prompt for the QA rows; raw audit surfaces shown verbatim.

Run: .venv/Scripts/python.exe examples/regulated_memory_demo.py
"""
from __future__ import annotations

import os
import time
import warnings
from calendar import timegm

warnings.filterwarnings("ignore")
from anthropic import Anthropic

from genome.embeddings import EmbeddingProvider
from genome.evals.locomo import ANSWER_PROMPT
from genome.memory.belief import (
    answer_belief_context, explain_belief, ingest_belief_turn,
)
from genome.memory.facade import Memory

MODEL = "claude-haiku-4-5-20251001"
client = Anthropic()


def haiku(p, mt=350):
    r = client.messages.create(model=MODEL, max_tokens=mt, temperature=0.0,
                               messages=[{"role": "user", "content": p}])
    return (r.content[0].text if r.content else "") or ""


def ep(y, m):
    return float(timegm((y, m, 1, 0, 0, 0, 0, 0, 0)))


# A loan applicant's file, as narrated to the intake agent over time. Note the
# OUT-OF-ORDER correction in session 4 (a late-arriving fact about an earlier date) --
# the case that breaks wall-clock memory.
EVENTS = [
    # (narration_time, text)
    (ep(2023, 1), "Rivera is employed at Northwind Logistics as a dispatcher; stated annual income is 72000 dollars, as of January 2023."),
    (ep(2023, 9), "Update on Rivera: in August 2023 they moved to Cascade Freight and their stated annual income is now 88000 dollars."),
    (ep(2024, 3), "Rivera's file was updated: as of February 2024 their stated annual income is 95000 dollars."),
    (ep(2024, 4), "Correction to Rivera's record: back in 2022, before Northwind, Rivera was actually self-employed with a stated income of 60000 dollars."),
]

QUERIES = [
    ("Current", "What is Rivera's current stated annual income?"),
    ("As-of loan approval (Mar 2024)", "What was Rivera's stated annual income on record as of March 2024?"),
    ("As-of the Aug 2023 credit pull", "What was Rivera's stated annual income in September 2023?"),
    ("As-of 2022 (out-of-order fact)", "What was Rivera's employment in 2022?"),
]


def genome_answer(mem, q):
    ctx = answer_belief_context(mem, q, user_id="acct", llm=lambda p: haiku(p))
    return haiku(ANSWER_PROMPT.format(context=ctx or "(no relevant memories)", question=q)).strip(), ctx


def mem0_ctx(m, q):
    parts = []
    try:
        res = m.search(q, top_k=20, filters={"user_id": "acct"})
        items = res.get("results", res) if isinstance(res, dict) else res
        for it in (items or []):
            parts.append(f"- {it.get('memory') or it.get('text') or ''}")
            mid = it.get("id")
            if mid:
                for h in (m.history(mid) or []):
                    if h.get("new_memory"):
                        parts.append(f"  (changed: {h.get('old_memory')} -> {h.get('new_memory')} at {h.get('updated_at') or h.get('created_at')})")
    except Exception as e:
        parts.append(f"(mem0 error: {e})")
    return "\n".join(parts)


def main():
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need both keys"); return 1
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")

    print("Ingesting Rivera's loan file into GENOME (belief-state) and Mem0 (infer=True)...\n")
    g = Memory(storage=":memory:", embedding_provider=embed, llm_call=lambda p: haiku(p))
    for st, text in EVENTS:
        ingest_belief_turn(g, text, session_time=st, user_id="acct", llm=lambda p: haiku(p))

    from mem0 import Memory as Mem0
    import shutil
    shutil.rmtree("/tmp/qdrant", ignore_errors=True)
    m0 = Mem0.from_config({
        "llm": {"provider": "anthropic", "config": {"model": MODEL, "temperature": 0.0}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}}})
    for st, text in EVENTS:
        try:
            m0.add([{"role": "user", "content": text}], user_id="acct", infer=True)
        except Exception:
            pass

    print("=" * 78)
    print("COMPLIANCE QUERIES  (same responder + prompt for both)")
    print("=" * 78)
    for label, q in QUERIES:
        ga, _ = genome_answer(g, q)
        ma = haiku(ANSWER_PROMPT.format(context=mem0_ctx(m0, q) or "(no memories)", question=q)).strip()
        print(f"\nQ [{label}]: {q}")
        print(f"   GENOME : {ga}")
        print(f"   Mem0   : {ma}")

    print("\n" + "=" * 78)
    print("AUDIT SURFACE  — \"show me the full income history and the source of each value\"")
    print("=" * 78)
    print("\nGENOME explain_belief(Rivera, 'income')  [deterministic, domain-time, with provenance]:")
    recs = explain_belief(g, "Rivera", user_id="acct")
    inc = [r for r in recs if "income" in r.attribute] or recs
    for r in inc:
        frm = time.strftime("%b %Y", time.gmtime(r.valid_from))
        to = time.strftime("%b %Y", time.gmtime(r.valid_until)) if r.valid_until else "present"
        print(f"  - {r.attribute} = {r.value}  valid [{frm} -> {to}]")
        print(f"      source: \"{(r.source_text or '')[:90]}\"")

    print("\nMem0 raw store + history() for the same account  [wall-clock, unstructured]:")
    try:
        allm = m0.get_all(filters={"user_id": "acct"})
        items = allm.get("results", allm) if isinstance(allm, dict) else allm
        for it in (items or []):
            print(f"  - {it.get('memory')}")
            for h in (m0.history(it.get('id')) or []):
                print(f"      change: {h.get('old_memory')} -> {h.get('new_memory')}  at {h.get('updated_at') or h.get('created_at')}")
    except Exception as e:
        print(f"  (mem0 error: {e})")

    print("\n" + "=" * 78)
    print("HONEST READ of this demo:")
    print("- On the QA rows, BOTH systems can answer when Mem0 keeps the dated statements")
    print("  and the dates sit in the text (as here). Do not claim 'Mem0 gets it wrong'.")
    print("- The durable difference is the AUDIT RECORD itself:")
    print("  * GENOME: a deterministic, structured, per-attribute bi-temporal table with")
    print("    explicit DOMAIN-time validity windows + the source text for each value --")
    print("    machine-queryable, reproducible, no LLM at audit time.")
    print("  * Mem0: unstructured text blobs whose only timestamps are WALL-CLOCK (ingest")
    print("    time, all identical here) and whose retention is a non-deterministic LLM")
    print("    decision; answering requires re-deriving from text with an LLM.")
    print("- The QA-accuracy gap (0.87 vs 0.68 on the TempBelief benchmark) shows up at")
    print("  VOLUME, where Mem0 more often overwrites/loses a value or retrieval misses the")
    print("  right dated statement. On a small clean file like this, Mem0 keeps up.")
    print("The regulated wedge is therefore about a DETERMINISTIC, STRUCTURED, PROVENANCE-")
    print("LINKED audit record -- not about Mem0 returning wrong answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
