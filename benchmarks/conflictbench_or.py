"""ConflictBench: does resolve_conflicts=True produce the correct FINAL STORE?

Deterministic contradiction scenarios per (entity, attribute) slot:
    1. assert v1                ("Maya moved to Boston.")
    2. duplicate, rephrased     ("Like I said, Maya is living in Boston these days.")
    3. correction to v2         ("Actually I had that wrong: Maya moved to Denver, not Boston.")
    4. pure negation (half the slots)  ("Maya doesn't have a car anymore.")

Both arms ingest the SAME turns: resolve_conflicts=True vs the naive INSERT
control. The audit is 100% mechanical (value-substring presence in the final
store + top-1 retrieval check), so the only LLM cost is GENOME's own
extraction/resolution, via OpenRouter. No judge model.

Metrics per arm: current-value present, stale (superseded) value still present,
negated fact still present, mean records per slot, top-1 retrieval correct.

Env: OPENROUTER_API_KEY required.
Run: .venv/Scripts/python.exe benchmarks/conflictbench_or.py [--slots 16]
"""
from __future__ import annotations

import argparse
import os
import time

from openai import OpenAI

from genome.embeddings import EmbeddingProvider
from genome.evals.tempbelief import _ATTRS, _NAMES
from genome.memory.facade import Memory

OR_BASE = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("OR_MODEL", "anthropic/claude-haiku-4.5")
_client: OpenAI | None = None


def llm(p, mt=300):
    global _client
    if _client is None:
        _client = OpenAI(base_url=OR_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    for a in range(5):
        try:
            r = _client.chat.completions.create(
                model=MODEL, max_tokens=mt, temperature=0.0,
                messages=[{"role": "user", "content": p}])
            return (r.choices[0].message.content or "") if r.choices else ""
        except Exception:
            if a == 4:
                raise
            time.sleep(2 ** a)


_ASSERT = {
    "location": "{e} moved to {v}.",
    "employer": "{e} started working at {v}.",
    "occupation": "{e} became a {v}.",
    "relationship_status": "{e} is {v} now.",
    "car": "{e} got {v}.",
    "favorite_food": "{e}'s favorite food is {v}.",
    "gym": "{e} joined {v}.",
}
_DUP = {
    "location": "Like I said, {e} is living in {v} these days.",
    "employer": "As I mentioned, {e} works at {v}.",
    "occupation": "Like I said before, {e} works as a {v}.",
    "relationship_status": "As I mentioned, {e} is {v}.",
    "car": "Like I mentioned, {e} drives {v}.",
    "favorite_food": "As I said, {e} loves {v}.",
    "gym": "Like I said, {e} goes to {v}.",
}
_CORRECT = {
    "location": "Actually I had that wrong: {e} moved to {v2}, not {v1}.",
    "employer": "Correction: {e} actually works at {v2}, not {v1}.",
    "occupation": "I got that wrong before: {e} is a {v2}, not a {v1}.",
    "relationship_status": "Actually, {e} is {v2} now, not {v1}.",
    "car": "My mistake earlier: {e} drives {v2}, not {v1}.",
    "favorite_food": "Actually {e}'s favorite food is {v2}, not {v1}.",
    "gym": "I misspoke before: {e} goes to {v2}, not {v1}.",
}
_NEGATE = {
    "location": "{e} doesn't live in {v} anymore.",
    "employer": "{e} no longer works at {v}.",
    "occupation": "{e} isn't a {v} anymore.",
    "relationship_status": "{e} is no longer {v}.",
    "car": "{e} doesn't have {v} anymore.",
    "favorite_food": "{e} has gone off {v} completely.",
    "gym": "{e} quit {v}.",
}
_QUESTION_NOUN = {a: noun for a, (noun, _pool) in _ATTRS.items()}


def build_slots(n_slots):
    attrs = list(_ATTRS)
    slots = []
    for i in range(n_slots):
        e = _NAMES[i % len(_NAMES)]
        a = attrs[i % len(attrs)]
        _noun, pool = _ATTRS[a]
        v1 = pool[i % len(pool)]
        v2 = pool[(i + 3) % len(pool)]
        if v2 == v1:
            v2 = pool[(i + 4) % len(pool)]
        negate = (i % 2 == 1)
        slots.append({"entity": e, "attr": a, "v1": v1, "v2": v2, "negate": negate})
    return slots


def turns_for(s):
    a = s["attr"]
    out = [
        _ASSERT[a].format(e=s["entity"], v=s["v1"]),
        _DUP[a].format(e=s["entity"], v=s["v1"]),
        _CORRECT[a].format(e=s["entity"], v1=s["v1"], v2=s["v2"]),
    ]
    if s["negate"]:
        out.append(_NEGATE[a].format(e=s["entity"], v=s["v2"]))
    return out


_NEG_MARKERS = ("n't", " not ", "no longer", "anymore", "quit", "gone off",
                "used to", "My mistake", "misspoke", "had that wrong",
                "got that wrong", "Correction")


def _asserts(text, value, other):
    """True iff `text` positively asserts `value`: the value appears, the
    competing value does not (correction turns mention both), and no negation
    marker is present (a stored negation is not an assertion). Conservative
    and mechanical; markers cover this benchmark's templates plus common
    extractor phrasings."""
    if value not in text or other in text:
        return False
    low = text.lower()
    return not any(m.lower() in low for m in _NEG_MARKERS)


def audit(mem, slots, dump_path=None):
    recs = mem.list_all(user_id="cb")
    texts = [r.content for r in recs]
    if dump_path:
        import json
        from pathlib import Path
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({"id": r.id, "content": r.content}) + "\n")
    res = {"current_present": 0, "stale_present": 0, "negated_present": 0,
           "top1_correct": 0, "top1_total": 0, "records": len(recs)}
    for s in slots:
        ent_texts = [t for t in texts if s["entity"] in t]
        asserts_v1 = any(_asserts(t, s["v1"], s["v2"]) for t in ent_texts)
        asserts_v2 = any(_asserts(t, s["v2"], s["v1"]) for t in ent_texts)
        if s["negate"]:
            # final truth: v2 was negated with no replacement -> no positive
            # assertion of either value should survive
            if asserts_v2:
                res["negated_present"] += 1
            if asserts_v1:
                res["stale_present"] += 1
        else:
            if asserts_v2:
                res["current_present"] += 1
            if asserts_v1:
                res["stale_present"] += 1
            # retrieval poisoning check: does the top hit positively assert
            # the CURRENT value?
            noun = _QUESTION_NOUN[s["attr"]]
            hits = mem.search(f"What is {s['entity']}'s {noun}?", user_id="cb", limit=3)
            res["top1_total"] += 1
            if hits and _asserts(hits[0].content, s["v2"], s["v1"]):
                res["top1_correct"] += 1
    return res


def run_arm(name, slots, resolve):
    embed = EmbeddingProvider()  # local default
    mem = Memory(storage=":memory:", embedding_provider=embed,
                 llm_call=llm, resolve_conflicts=resolve)
    t0 = time.time()
    for s in slots:
        for turn in turns_for(s):
            mem.add(turn, user_id="cb")
    dt = time.time() - t0
    res = audit(mem, slots, dump_path=f"results/conflictbench/store_{name}.jsonl")
    mem.close()
    res["ingest_s"] = round(dt, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=16)
    args = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("need OPENROUTER_API_KEY"); return 1
    print(f"[cfg] model={MODEL} via OpenRouter  embedder=local all-MiniLM  slots={args.slots}", flush=True)
    slots = build_slots(args.slots)
    n_neg = sum(s["negate"] for s in slots)
    n_pos = len(slots) - n_neg
    print(f"slots: {len(slots)} ({n_pos} correction-only, {n_neg} correction+negation)", flush=True)

    results = {}
    for name, resolve in (("naive-insert", False), ("resolve-conflicts", True)):
        print(f"running arm: {name} ...", flush=True)
        results[name] = run_arm(name, slots, resolve)

    print("\n=== ConflictBench: final-store audit (mechanical, no judge) ===")
    hdr = f"{'metric':26}" + "".join(f"{n:>20}" for n in results)
    print(hdr)
    rows = [
        ("current value present", "current_present", n_pos),
        ("stale value REMAINS", "stale_present", len(slots)),
        ("negated fact REMAINS", "negated_present", n_neg),
        ("top-1 retrieval correct", "top1_correct", n_pos),
        ("total records in store", "records", None),
        ("ingest seconds", "ingest_s", None),
    ]
    for label, key, denom in rows:
        line = f"{label:26}"
        for n in results:
            v = results[n][key]
            line += f"{(f'{v}/{denom}' if denom else str(v)):>20}"
        print(line)
    print("\nREAD: resolve-conflicts should push 'stale value REMAINS' and "
          "'negated fact REMAINS' toward 0 without losing current values; "
          "naive-insert shows the pile-up baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
