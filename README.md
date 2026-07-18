# GENOME

**Open memory for AI agents. Same answer accuracy as Mem0 — but ~1,000× cheaper to store, runs fully offline, and keeps an auditable record.**

[![tests](https://github.com/NORTHTEKDevs/genome/actions/workflows/tests.yml/badge.svg)](https://github.com/NORTHTEKDevs/genome/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/genome-memory)](https://pypi.org/project/genome-memory/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

Most agent-memory tools (like Mem0) call an LLM on **every message** to decide what to
remember. That's the slow, expensive part — and GENOME's bet is that you don't need it.
GENOME just embeds each message locally: no LLM, no API, no network in the write path.

Benchmarked honestly on public datasets (LoCoMo, LongMemEval), GENOME **answers just as
accurately as Mem0** — while storing memories for a tiny fraction of the cost and running
completely offline.

> **Honest up front:** on answer accuracy, GENOME *ties* Mem0 — we do **not** claim to beat
> it there (two independent benchmark runs confirm parity). The advantage is cost, speed,
> offline operation, and a temporal/auditable record Mem0 can't produce.

## Don't believe it? Prove it yourself

The **cost, speed, and offline** claims need no API key — measure them on *your* machine in 60 seconds:

```bash
git clone https://github.com/NORTHTEKDevs/genome && cd genome
pip install -e . && python -m genome.verify
```

It writes memories with your **outbound network physically blocked** and prints a live
pass/fail receipt — 0 network calls, 0 LLM calls, single-digit-ms writes, retrieval that works:

```
  [PASS] Air-gapped write path: wrote 200 memories with every outbound socket blocked -> 0 network attempts, 0 LLM calls
  [PASS] Write latency: 7.1 ms/message  (Mem0's measured write path: ~2,055 ms + 1 LLM call/message)
  [PASS] Retrieval works: top hit score 0.598
```

That receipt covers the cost/speed/offline story only. The **accuracy-parity with Mem0** claim
is a separate, larger check that needs an LLM key — reproduce it head-to-head on the same
questions with your own key via `python benchmarks/head_to_head.py` (one OpenRouter key works;
see [`benchmarks/RESULTS.md`](./benchmarks/RESULTS.md) for the n=90 / n=205 runs, the paired
significance tests, and the published nulls). The full test suite runs in public CI (badge
above). The pitch isn't "trust me" — it's "run it."

## Add persistent memory to your agent in one line (MCP)

GENOME ships a **fully-local MCP server** — cross-session memory for Claude Desktop, Claude
Code, or Cursor with **no API key and no data leaving your machine**:

```bash
pip install "genome-memory[mcp]"
```

```json
{ "mcpServers": { "genome": { "command": "genome-mcp" } } }
```

Or zero-install via uv: `{ "command": "uvx", "args": ["--from", "genome-memory[mcp]", "genome-mcp"] }`

Tools the agent gets: **`remember`**, **`recall`**, **`forget`**, **`reset_memories`**.
Memories persist locally in `~/.genome/memories.db`. [Full MCP details ↓](#use-it-as-an-mcp-server-fully-local-memory-for-any-agent)

## GENOME vs Mem0 at a glance

| | GENOME | Mem0 |
|---|---|---|
| **Answer accuracy** (LoCoMo, LongMemEval) | tied | tied |
| **LLM calls to store one message** | **0** | 1+ |
| **Write speed** | **~10 ms** | ~2,000 ms |
| **Runs offline / air-gapped** | **yes** | no (needs an LLM API) |
| **Ingest cost** (10k-user deployment) | **~$190 / yr** | $159k–$1.6M / yr |
| **"What was true in March?"** (point-in-time) | **yes** | no |
| **Deterministic, auditable memory** | **yes** | no |

Every number is measured within one harness — same responder, judge, embedder, and top-k;
only the memory layer changes — with paired significance tests. Full detail and per-number
provenance: [`benchmarks/RESULTS.md`](./benchmarks/RESULTS.md). Formatted report:
[`benchmarks/GENOME-LoCoMo-Report.pdf`](./benchmarks/GENOME-LoCoMo-Report.pdf).

## Why it's ~1,000× cheaper: it never calls an LLM to remember

Storing one message costs **one LLM call in Mem0, zero in GENOME** (just a local embedding).
That's not a benchmark you can argue with — it's arithmetic, and it holds no matter which
LLM you price it against. At 10,000 users × 50 messages/day (15M messages/month):

| Model Mem0 uses to extract | Mem0's yearly ingest bill | GENOME |
|---|---|---|
| Claude Haiku | $1,601,757 | **$190** |
| gpt-4o-mini | $238,596 | **$190** |
| cheapest hosted model | $159,064 | **$190** |

The gap survives the cheapest model and *grows* in production (Mem0 re-sends stored memories
to the LLM as the store fills). Reproduce: `python benchmarks/tco_project.py` (no API key).

## It runs air-gapped

GENOME's default embedder is local. We proved the write path is genuinely offline by
**blocking all network during writes** — they still succeed:

- **~10 ms/message, 0 network calls, 0 LLM calls** (`python benchmarks/local_writepath.py`)
- Mem0 can't do this — it needs an LLM API call to ingest.

That makes GENOME usable on-prem, in regulated environments, or fully offline. It's a yes/no
capability, not a price point.

## How it works

- **Write:** embed the message locally and store it. No LLM, no network. (~10 ms)
- **Read:** vector search over your memories, with an optional local cross-encoder reranker
  for harder queries.
- **Optional bi-temporal layer:** track how facts change over time and answer "what was true
  at time T" — see below.

## Install

```bash
pip install genome-memory
```

The default embedder is local (`sentence-transformers/all-MiniLM-L6-v2`) — no API key,
works offline; the first run downloads the ~90 MB model once. OpenAI embeddings are
optional for higher-dimensional retrieval.

**Dependency footprint, honestly:** the core install is `numpy`, `sentence-transformers`,
`scikit-learn`, and `rank-bm25`. Local embeddings run on PyTorch (pulled in by
sentence-transformers), so it isn't a tiny install — that's the deliberate tradeoff for
offline, zero-cost embedding. Plotting/benchmark-chart deps live in an optional `[viz]`
extra, not the core. Migrating from Mem0? See
**[docs/migrating_from_mem0.md](docs/migrating_from_mem0.md)**.

## Quickstart (fully local, no API key)

```python
from genome import Memory

mem = Memory(storage="genome.db")   # local embedder by default; ":memory:" for ephemeral

# Store a message -- embedded locally, no LLM call, no network
mem.add("Ada met Lin at the robotics summit in Berlin.", user_id="u1")
mem.add("They are collaborating on an open-source planning library.", user_id="u1")

# Retrieve the most relevant memories
for hit in mem.search("Where did Ada meet Lin?", user_id="u1", limit=5):
    print(f"{hit.score:.3f}  {hit.content}")
```

`Memory` mirrors Mem0's API (`add` / `search` / `get` / `delete` / `reset`) — a near
drop-in swap. To use OpenAI embeddings instead (set `OPENAI_API_KEY`):

```python
from genome import Memory, EmbeddingProvider
mem = Memory(storage="genome.db",
             embedding_provider=EmbeddingProvider(model_name="openai:text-embedding-3-small"))
```

## Use it as an MCP server (fully-local memory for any agent)

GENOME ships an MCP server, so any MCP client (Claude Desktop, Claude Code, Cursor, …) gets
persistent cross-session memory that runs **entirely on the local machine** — no LLM calls,
no API keys, no data leaves the box. Most memory MCPs can't say that.

Install with the `mcp` extra, then add it to your client's config:

```bash
pip install "genome-memory[mcp]"
```

```json
{
  "mcpServers": {
    "genome": { "command": "genome-mcp" }
  }
}
```

Tools the agent gets: **`remember`** (store a fact/preference, local + 0 LLM), **`recall`**
(semantic search), **`forget`** (delete the memory matching a query), **`reset_memories`**
(clear a user's memories). Memories persist in `~/.genome/memories.db` (override with the
`GENOME_MCP_DB` env var). Run standalone with `genome-mcp` or `python -m genome.mcp.server`.

## Run it as an HTTP API

Prefer HTTP? GENOME ships a FastAPI server that mirrors the library 1:1 (`add` / `search` /
`get` / `update` / `delete` / `reset` / `synthesize`), with an auto-generated OpenAPI spec at
`/docs`.

```bash
pip install "genome-memory[fastapi]"
```

**Try it locally** (keyless, loopback only — one flag makes the "no auth" intent explicit):

```bash
GENOME_ALLOW_NO_AUTH=1 python -m genome.server        # serves on 127.0.0.1:8080
```

```bash
curl -X POST localhost:8080/v1/memories \
  -H 'Content-Type: application/json' \
  -d '{"text": "Ada met Lin at the robotics summit in Berlin.", "user_id": "u1"}'

curl -X POST localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "Where did Ada meet Lin?", "user_id": "u1", "limit": 5}'
```

**Safe by default.** The server refuses to serve unauthenticated unless you opt in as
above, and it will not bind a non-loopback interface without a key. To expose it, set an
API key (sent as `X-API-Key`) — required to bind beyond localhost:

```bash
GENOME_API_KEY=$(openssl rand -hex 32) GENOME_HOST=0.0.0.0 python -m genome.server
# then add:  -H "X-API-Key: $GENOME_API_KEY"  to every request
```

For multi-tenant deployments, set `GENOME_REQUIRE_SCOPE=1` to require `user_id`/`agent_id` on
every call and disable the global reset. Docker: `docker-compose up` (needs `GENOME_API_KEY`
and `POSTGRES_PASSWORD`; Postgres is published on loopback only). Full guide, including the
Postgres backend and every env var: [`docs/tutorial_quickstart.md`](./docs/tutorial_quickstart.md).

## The honest results

Same responder + judge + embedder for every system; only the memory layer changes.

| What we measured | Result | Verdict |
|---|---|---|
| Answer accuracy, in-window (LoCoMo) | GENOME 0.851 vs Mem0 0.855 (p > 0.23) | **Tied** |
| Answer accuracy, harder bench (LongMemEval, n=90 & n=205) | directionally ahead, not significant (p = 0.14–0.19) | **Tied** |
| Accuracy when history overflows the context window | **+0.409** at 80× less context (p = 8e-10) | **Win** |
| Cost to store a message | 0 LLM calls vs 1+; **837–8,433× cheaper** | **Win** |
| Write path | **~10 ms, air-gapped**, 0 network calls | **Win** |
| Point-in-time ("what was true at T") | belief-state **0.870** vs Mem0 0.676 (synthetic data) | **Win, with caveat** |
| Retrieval hit-rate with reranking | improves hit@10 (up to 0.943); local + free | **Win** |

### What we tested that *didn't* help (so you don't have to)

We publish our nulls — it's how you know the wins are real:
- **Synthesis / consolidation:** accuracy-neutral at equal token budget (p = 0.86).
- **Hybrid (BM25 + dense) and graph retrieval:** hybrid underperformed plain dense on LoCoMo;
  graph was not validated here.
- **Reranking's accuracy gain is embedder-dependent:** it reliably improves *retrieval
  hit-rate*, but its effect on final *answer accuracy* depends on the embedder — treat it as a
  retrieval-quality tool, not a guaranteed accuracy win.

## Bi-temporal memory: "what was true at time T"

GENOME can track how facts change over time and answer point-in-time questions — something
overwrite-based memory structurally can't do (it only keeps the latest value):

```python
from genome.memory.belief import ingest_belief_turn, answer_belief_context

mem = Memory(storage="genome.db", llm_call=my_llm_fn)

# facts land at their DOMAIN time (parsed from the text), not wall-clock ingest time
ingest_belief_turn(mem, "In March 2024, Jordan moved to Seattle.", session_time=t0, user_id="u")
ingest_belief_turn(mem, "Jordan just moved to Austin.", session_time=t2, user_id="u")

answer_belief_context(mem, "Where does Jordan live now?", user_id="u")            # -> Austin
answer_belief_context(mem, "Where did Jordan live in early 2024?", user_id="u")   # -> Seattle
answer_belief_context(mem, "List every city Jordan has lived in.", user_id="u")   # -> Seattle; Austin
```

On the TempBelief benchmark it answers as-of queries at **0.870** vs Mem0's 0.676, with the
knowledge graph audited at 0.97 precision / 0.96 recall. **Caveat:** TempBelief is synthetic
text with explicit dates; the edge shrinks on natural speech. Real capability, bounded proof.

## Optional features

Opt-in; the default path stays LLM-free and local at ingest.

```python
mem = Memory(
    storage="genome.db",
    llm_call=my_llm_fn,             # LLM-based fact extraction on add()
    resolve_conflicts=True,         # ADD/UPDATE/DELETE vs existing memories
    auto_extract_entities=True,     # entity graph for graph retrieval
    auto_consolidate_threshold=200, # summarize-or-prune when a scope grows past N
)
mem.search("...", user_id="u1", mode="hybrid")   # modes: "dense" (default), "hybrid", "graph"
```

Reranking (local, free, no API):

```python
from genome.memory.rerank import CrossEncoderReranker
mem = Memory(storage="genome.db", reranker=CrossEncoderReranker())   # lazy-loaded
mem.search("Where did the user go on vacation?", user_id="u1", limit=5)  # reranked
```

## Reproduce the benchmarks

The LoCoMo and LongMemEval datasets are **not bundled** (they carry their own licenses —
LoCoMo is CC BY-NC 4.0). See [`benchmarks/data/README.md`](./benchmarks/data/README.md) to
download them. The first two lines need no dataset and no API keys:

```bash
python benchmarks/local_writepath.py        # local write path: ~10ms/msg, 0 network
python benchmarks/tco_project.py            # deployment cost projection
python benchmarks/verdict.py                # in-window accuracy + McNemar
python benchmarks/haystack_report.py        # overflow / context-window crossover
python benchmarks/ingest_cost.py --n 80     # measured ingestion cost vs Mem0
python benchmarks/lme_qa.py --n 90          # LongMemEval head-to-head vs Mem0
python benchmarks/tempbelief_run.py --convs 6   # bi-temporal point-in-time vs baselines
```

## License

**Apache License 2.0** — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

GENOME is free and open source: read it, modify it, self-host it, and embed it in your own
applications — commercial use included — under the terms of Apache 2.0. Questions:
info@northtek.io.

Copyright 2026 Northtek (FrostByte Digital LLC).
