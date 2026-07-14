# GENOME vs Mem0 — Total Cost of Ownership (memory ingest)

**The one slide.** GENOME matches Mem0 on answer accuracy while ingesting memory at
**0 LLM calls per message** vs Mem0's **1.00** — an architectural fact, not a benchmark
score. Projected to a 10,000-user deployment that is a **$159k–$1.6M/year** difference in
memory-ingest LLM spend. On its default local embedder GENOME writes at **~10 ms/message
(~203× faster than Mem0's 2,055 ms) and runs air-gapped** — no LLM, no API, no network in
the write path (privacy, offline capability, no rate limits).

This number is only meaningful because of the parity result below it. If Mem0 bought
better memories with that spend, the cost would be justified. It doesn't — we measured it.

---

## Measured (not assumed)

Instrumented 80-turn slice, identical data, Mem0 in its default `infer=True` mode with the
same Haiku 4.5 model (`benchmarks/ingest_cost.py` → `results/ingest_cost.log`, Finding 03):

| System | LLM calls / msg | Write latency / msg | Cost (80 turns) |
|---|---|---|---|
| Mem0 | **1.00** | 2,055 ms | $0.7119 |
| GENOME (dense) | **0** | 240 ms* | $0.00008 |

\* GENOME's 240 ms includes the OpenAI embedding round-trip and is already counted as a
cost below; with a local embedder the write path is single-digit milliseconds.

## Projected to deployment (10,000 users × 50 msgs/day = 15M messages/month)

Reproduce: `.venv/Scripts/python.exe benchmarks/tco_project.py`

| Extraction model | Mem0 $/month | Mem0 $/year | GENOME $/year | Ratio |
|---|---|---|---|---|
| Haiku 4.5 | $133,480 | $1,601,757 | $190 | **8,433×** |
| gpt-4o-mini | $19,883 | $238,596 | $190 | 1,256× |
| gpt-4.1-nano (cheapest hosted) | $13,255 | $159,064 | $190 | **837×** |

- **LLM calls/month:** Mem0 15,000,000 · GENOME **0**.
- The ratio is **deployment-size-independent** and survives the cheapest extraction model.
- The gap **widens** in production: Mem0 ships existing memories back to the LLM on each
  write, so per-message cost rises as the store fills. This linear projection under-states it.

## The categorical difference: GENOME's default write path is fully local

Measured (`benchmarks/local_writepath.py`, no API keys): with the **default** embedder
(`all-MiniLM-L6-v2`, local), GENOME writes at **~10 ms/message** with **0 LLM calls, 0
embedding-API calls, 0 external dependencies**. The test *blocks all outbound network during
the writes* and they still succeed (0 connection attempts) — GENOME runs **air-gapped**.

Mem0 architecturally cannot do this: its write path requires an LLM API call to extract
memories. This is the on-prem / regulated / offline capability that isn't on a price curve —
it's a yes/no, and only one system can say yes.

| | Write latency | LLM calls | Runs air-gapped? |
|---|---|---|---|
| Mem0 | 2,055 ms/msg | 1 / msg | **No** (needs LLM API) |
| GENOME (default, local) | **~10 ms/msg** | **0** | **Yes** (network blocked, writes still pass) |

*Scope: this is GENOME's default **dense** memory path. The optional belief-state temporal
layer does use an LLM at ingest and is priced separately; it is not required for core memory.*

## Why the cost gap is a *win*, not a *tradeoff* — the parity result

The cost win would be hollow if GENOME were less accurate. It isn't:

- **LoCoMo:** statistical tie (~0.85 both; same responder, judge, embedder).
- **LongMemEval-S (harder, 90 Q, Sonnet responder, gold+4 distractors):** GENOME
  **0.700** / GENOME+rerank vs Mem0 **0.622** — a directional lead (McNemar p=0.19, not yet
  significant), *not* a deficit.
- **Retrieval hit-rate:** GENOME+rerank ≥ Mem0 on every question type measured.

So Mem0 spends 837–8,433× more at ingest and buys **no measured accuracy advantage**. That
is the honest, defensible claim: *same-or-better answers, at a fraction of the cost, with no
LLM in the write path.*

## What this is NOT (bounds a critic can't exploit if you state them first)

- This is **memory-ingest cost only.** Both systems pay the same answer-LLM cost at query
  time; this table does not claim to cover total inference spend.
- Measured under Mem0's default `infer=True`. Batched/deferred extraction would lower Mem0's
  cost and latency at the expense of memory freshness — the 0-vs-1 call structure is unchanged.
- Linear extrapolation from an 80-message slice, disclosed; real Mem0 cost grows super-linearly.
