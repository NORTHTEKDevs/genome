# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Walkthrough demo: a fact changes over two years, and GENOME answers
'what was true back then' from structure rather than from guesswork.

Runs entirely locally. No API key, no LLM call in the write path, no network.
This is the script behind docs/demo.gif; regenerate the GIF with
`python tools/render_demo_gif.py` after changing it.

Run: python examples/demo_timeline.py
"""
from __future__ import annotations

import os
import socket
import time

from genome import Memory
from genome.memory.belief import answer_belief_context, ingest_belief_turn, parse_when


def say(line: str = "") -> None:
    print(line, flush=True)


def main() -> int:
    say("GENOME: local-first memory with a point-in-time record")
    say("=" * 58)
    say()

    # ---- 1. write path: no LLM, no network -------------------------------
    say("[1] Storing conversation turns. Local embedding only, zero LLM calls.")
    mem = Memory(storage=":memory:")

    turns = [
        ("March 2023", "Priya moved to Boston for the new job."),
        ("August 2023", "Priya adopted a dog, a beagle named Cooper."),
        ("January 2024", "Priya relocated to Seattle. She loves the coffee."),
        ("June 2024", "Priya is thinking about maybe moving to Denver, nothing decided."),
        ("February 2025", "Priya moved again, this time to Austin."),
    ]
    t0 = time.perf_counter()
    for when, text in turns:
        mem.add(f"[{when}] {text}", user_id="demo")
        say(f"    stored  {when:14} {text}")
    dt_ms = (time.perf_counter() - t0) * 1000 / len(turns)
    say()
    say(f"    {len(turns)} turns, {dt_ms:.1f} ms per write, 0 LLM calls, 0 network calls")
    say()

    # ---- 2. plain recall --------------------------------------------------
    say("[2] Semantic recall over the local store.")
    for hit in mem.search("where does Priya live", user_id="demo", limit=2):
        say(f"    {hit.score:.2f}  {hit.content}")
    say()

    # ---- 3. the part other memory systems cannot do -----------------------
    say("[3] Point-in-time: what was true THEN, not what is true now.")
    belief = Memory(storage=":memory:", llm_call=_local_extractor)
    for when, text in turns:
        ingest_belief_turn(belief, f"[{when}] {text}",
                           session_time=parse_when(when, time.time()),
                           user_id="demo", llm=_local_extractor)

    for question in ("What was Priya's city in May 2023?",
                     "What was Priya's city in March 2024?",
                     "What is Priya's city now?"):
        ctx = answer_belief_context(belief, question, user_id="demo", llm=_local_extractor)
        # The context starts with a header, then one line per resolved fact.
        # Stop at the raw-memory fallback block; the structured answer is above it.
        facts = []
        for ln in (ctx or "").splitlines()[1:]:
            ln = ln.strip()
            if ln.lower().startswith("recent related memories"):
                break
            if ln.startswith("- "):
                facts.append(ln[2:])
        say(f"    Q: {question}")
        say(f"    A: {facts[0] if facts else 'no record'}")
    say()
    say("    The June 2024 'thinking about Denver' turn is a plan, not a fact.")
    say("    It never becomes the answer.")
    say()

    # ---- 4. prove the write path is offline -------------------------------
    say("[4] Same write path with every outbound socket blocked.")
    real_socket = socket.socket

    class Blocked(socket.socket):
        def __init__(self, *a, **k):
            raise OSError("network disabled for this demo")

    socket.socket = Blocked
    try:
        mem.add("[April 2025] Priya started at a new lab in Austin.", user_id="demo")
        say("    write succeeded with networking disabled")
    finally:
        socket.socket = real_socket
    say()
    say("Reproduce all of it:  python -m genome.verify")
    return 0


def _local_extractor(prompt: str) -> str:
    """Deterministic stand-in for the belief layer's LLM, so this demo runs with
    no API key. It speaks the same two protocols a real model would.

    Set GENOME_DEMO_MODEL and OPENROUTER_API_KEY to run it against a real model
    instead. The core write path in steps 1, 2 and 4 never calls an LLM at all.
    """
    import re

    model = os.environ.get("GENOME_DEMO_MODEL")
    if model and os.environ.get("OPENROUTER_API_KEY"):
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
        r = client.chat.completions.create(
            model=model, max_tokens=300, temperature=0.0,
            messages=[{"role": "user", "content": prompt}])
        return (r.choices[0].message.content or "") if r.choices else ""

    # Query protocol: which belief is being asked about, and as of when.
    if "<question>" in prompt:
        q = prompt.split("<question>", 1)[1].split("</question>", 1)[0].strip()
        asof = re.search(r"in\s+([A-Z][a-z]+\s+\d{4})", q)
        mode = "as_of" if asof else "current"
        return (f"SUBJECT: Priya\nATTRIBUTE: location\nMODE: {mode}\n"
                f"ASOF: {asof.group(1) if asof else 'NONE'}")

    # Ingest protocol: durable facts only, dated to when they became true.
    turn = prompt.split("<turn>", 1)[1].split("</turn>", 1)[0].strip()
    when = re.search(r"\[([A-Z][a-z]+\s+\d{4})\]", turn)
    move = re.search(r"(?:moved to|relocated to|this time to)\s+([A-Z][a-z]+)", turn)
    tentative = re.search(r"thinking about|maybe|nothing decided", turn, re.I)
    if move and not tentative:
        return (f"FACT | Priya | location | {move.group(1)} | "
                f"{when.group(1) if when else 'NOW'} | FIRM")
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
