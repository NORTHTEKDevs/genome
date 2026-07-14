"""Controlled ingestion-cost benchmark: GENOME (dense) vs Mem0, MEASURED.

Ingests the SAME N turns through both, with Mem0's internal LLM instrumented to
count real calls + tokens. Converts the earlier '$53 estimated' into a measured
per-message ingestion cost, then extrapolates to the full corpus.

GENOME dense config = IdentityExtractor + no auto-extract => 0 LLM calls at
ingest (embeddings only). Mem0 = LLM fact-extraction per message.

Run: .venv/Scripts/python.exe benchmarks/ingest_cost.py [--n 100]
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import tiktoken

warnings.filterwarnings("ignore")
ENC = tiktoken.get_encoding("cl100k_base")
# Haiku pricing $/1M tokens (from genome cost table)
HAIKU_IN, HAIKU_OUT = 1.00, 5.00
EMBED_PER_M = 0.02  # text-embedding-3-small


def ntok(s): return len(ENC.encode(str(s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        print("need both keys"); return 1

    from genome.evals.locomo import load_locomo
    conv = load_locomo("benchmarks/data/locomo10.json")[0]
    turns = conv.turns[: args.n]
    spk_a = conv.speaker_a or (conv.speakers[0] if conv.speakers else "")
    contents = [
        (f"[{t.session_datetime}] {t.speaker}: {t.text}"
         if t.session_datetime else f"{t.speaker}: {t.text}")
        for t in turns
    ]
    print(f"corpus for test: {len(turns)} turns, "
          f"~{sum(ntok(c) for c in contents):,} content tokens", flush=True)

    # ---- Mem0, instrumented ----
    from mem0 import Memory as Mem0
    m = Mem0.from_config({
        "llm": {"provider": "anthropic",
                "config": {"model": "claude-haiku-4-5-20251001", "temperature": 0.0}},
        "embedder": {"provider": "openai",
                     "config": {"model": "text-embedding-3-small"}}})
    stat = {"calls": 0, "in": 0, "out": 0}
    _orig = m.llm.generate_response
    def _wrapped(messages, *a, **k):
        stat["calls"] += 1
        for msg in (messages or []):
            stat["in"] += ntok(msg.get("content", "") if isinstance(msg, dict) else msg)
        r = _orig(messages, *a, **k)
        stat["out"] += ntok(r)
        return r
    m.llm.generate_response = _wrapped

    t0 = time.time()
    for t, content in zip(turns, contents):
        role = "user" if t.speaker == spk_a else "assistant"
        try:
            m.add([{"role": role, "content": content}], user_id="ic",
                  metadata={"dia_id": t.dia_id})
        except Exception as e:
            print(f"  mem0 add error (continuing): {repr(e)[:100]}", flush=True)
    mem0_time = time.time() - t0

    # ---- GENOME dense (0 LLM calls) ----
    from genome.embeddings import EmbeddingProvider
    from genome.memory.facade import Memory as GMemory
    embed = EmbeddingProvider(model_name="openai:text-embedding-3-small")
    g = GMemory(storage=":memory:", embedding_provider=embed)  # IdentityExtractor, no LLM
    t0 = time.time()
    for content in contents:
        g.add(content, user_id="ic")
    gen_time = time.time() - t0
    gen_embed_tokens = sum(ntok(c) for c in contents)

    N = len(turns)
    mem0_cost = stat["in"] / 1e6 * HAIKU_IN + stat["out"] / 1e6 * HAIKU_OUT
    gen_cost = gen_embed_tokens / 1e6 * EMBED_PER_M
    FULL = 5882

    print(f"\n=== INGESTION COST, MEASURED over N={N} turns ===")
    print(f"MEM0 : {stat['calls']} LLM calls ({stat['calls']/N:.2f}/msg), "
          f"{stat['in']:,} in + {stat['out']:,} out tokens, {mem0_time:.1f}s, "
          f"${mem0_cost:.4f}")
    print(f"GENOME(dense): 0 LLM calls, {N} embeds, {gen_embed_tokens:,} embed "
          f"tokens, {gen_time:.1f}s, ${gen_cost:.5f}")
    print(f"\n--- ratios (per message) ---")
    print(f"  LLM calls:  mem0 {stat['calls']/N:.2f}  vs  GENOME 0")
    print(f"  cost:       mem0 ${mem0_cost/N:.5f}/msg  vs  GENOME ${gen_cost/N:.7f}/msg "
          f"= {mem0_cost/max(gen_cost,1e-9):.0f}x cheaper (GENOME)")
    print(f"  wall-clock: mem0 {mem0_time/N*1000:.0f} ms/msg  vs  GENOME "
          f"{gen_time/N*1000:.0f} ms/msg = {mem0_time/max(gen_time,1e-9):.0f}x faster")
    print(f"\n--- extrapolated to full {FULL}-turn corpus ---")
    print(f"  MEM0 : ~{int(stat['calls']/N*FULL):,} LLM calls, "
          f"~${mem0_cost/N*FULL:.2f}, ~{mem0_time/N*FULL/60:.0f} min")
    print(f"  GENOME: 0 LLM calls, ~${gen_cost/N*FULL:.4f}, "
          f"~{gen_time/N*FULL:.0f}s (per-message; batched is far faster)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
