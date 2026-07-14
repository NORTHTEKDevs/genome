"""Adversarial verification of the TempBelief belief-state result.

Kill-criterion #4 (fairness auditor): if extraction precision/recall against the
ground-truth event log is poor yet QA scores well, the as-of win is judge leniency
masking a broken pipeline -- do NOT trust it. This script re-ingests conversations
through the real belief pipeline, dumps the resulting bi-temporal KG, and scores it
against gold FactEvents reconstructed from the (independently authored) history
questions. It also prints sample as-of contexts so the mechanism is auditable.

Run: .venv/Scripts/python.exe benchmarks/tempbelief_verify.py [--convs 2]
"""
from __future__ import annotations

import argparse
import os
import re
import time

from anthropic import Anthropic

from genome.embeddings import EmbeddingProvider
from genome.evals.tempbelief import load
from genome.memory.belief import (
    answer_belief_context, ingest_belief_turn, parse_when,
)
from genome.memory.entities import _norm, list_entities
from genome.memory.facade import Memory
from genome.memory.temporal import entity_timeline

MODEL = "claude-haiku-4-5-20251001"
client = Anthropic()


def haiku(p, mt=300):
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


def gold_events(conv):
    """Reconstruct (entity, attr, value, from_epoch) from history questions."""
    ev = []
    for q in conv.questions:
        if q.category != "history":
            continue
        for seg in q.answer.split(";"):
            m = re.match(r"\s*(.+?)\s*\(from ([A-Z][a-z]+ \d{4})\)", seg)
            if m:
                ev.append((q.entity, q.attribute, m.group(1).strip(),
                           parse_when(m.group(2), 0)))
    return ev


def core(v):
    return re.sub(r"^(a|an|the) ", "", v.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=2)
    ap.add_argument("--data", default="benchmarks/data/tempbelief.json")
    args = ap.parse_args()
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need keys"); return 1
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    convs = load(args.data)[: args.convs]

    tp = fp = gold_n = matched = 0
    date_ok = 0
    for conv in convs:
        m = Memory(storage=":memory:", embedding_provider=embed, llm_call=lambda p: haiku(p))
        for t in conv.turns:
            st = parse_when(t.session_datetime, time.time())
            ingest_belief_turn(m, t.text, session_time=st, user_id="tb", llm=lambda p: haiku(p))
        gold = gold_events(conv)
        gold_n += len(gold)
        # recorded facts per entity
        recorded = []  # (entity_name, attr, value, valid_from)
        for e in list_entities(m, user_id="tb"):
            en = str(e.metadata.get("entity_name", ""))
            for f in entity_timeline(m, e.id, user_id="tb"):
                recorded.append((en, f.fact_type, f.value, f.valid_from))
        # precision: each recorded fact should match some gold event (value+attr+entity)
        for (en, attr, val, vf) in recorded:
            hit = None
            for (ge, ga, gv, gf) in gold:
                if _norm(en) == _norm(ge) and (core(val) in core(gv) or core(gv) in core(val)):
                    hit = (ge, ga, gv, gf); break
            if hit:
                tp += 1
                if abs(vf - hit[3]) <= 45 * 86400:   # domain-time within ~1.5 months
                    date_ok += 1
            else:
                fp += 1
        # recall: each gold event should be captured by some recorded fact
        for (ge, ga, gv, gf) in gold:
            if any(_norm(en) == _norm(ge) and (core(v) in core(gv) or core(gv) in core(v))
                   for (en, at, v, vf) in recorded):
                matched += 1

        # print a couple of as-of audit examples from this conv
        asofs = [q for q in conv.questions if q.category == "as-of"][:2]
        for q in asofs:
            ctx = answer_belief_context(m, q.question, user_id="tb", llm=lambda p: haiku(p))
            print(f"\n[{conv.conversation_id}] AS-OF AUDIT: {q.question}")
            print(f"  gold: {q.answer}")
            print(f"  belief context:\n    " + (ctx.replace(chr(10), chr(10) + '    ') if ctx else "(empty)"))
        m.close()

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = matched / gold_n if gold_n else 0
    dprec = date_ok / tp if tp else 0
    print(f"\n=== EXTRACTION QUALITY vs gold ({args.convs} convs) ===")
    print(f"  recorded facts: {tp+fp}  (true {tp}, false/spurious {fp})")
    print(f"  precision (recorded facts that are real):      {prec:.3f}")
    print(f"  recall    (gold events captured):              {rec:.3f}  ({matched}/{gold_n})")
    print(f"  domain-time accuracy (valid_from within ~45d): {dprec:.3f}  ({date_ok}/{tp})")
    verdict = ("HEALTHY - as-of win is backed by a correct KG" if prec > 0.7 and rec > 0.7 and dprec > 0.7
               else "SUSPECT - extraction weak; as-of QA may be judge leniency")
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
