# GENOME: the memory layer that knows what was true *when* — benchmarked honestly

*Draft announcement for the repo. Every number is measured and reproducible; the
null results are included on purpose.*

---

Most memory-layer benchmarks you'll read are cherry-picked: one flattering number,
no significance test, no baseline you'd actually deploy. We benchmarked GENOME the
opposite way — same responder, same judge, same embedder, same top-k for every
system, paired significance tests, baselines run at their *best*, and we publish the
results that *didn't* go our way alongside the ones that did.

## The headline: point-in-time memory

Ask a normal memory layer "where did the user live in March 2024?" and it fails —
it either returns the *latest* value (overwrite-based memory like Mem0) or can't tell
which of several dated mentions is the right one (plain retrieval). GENOME records
every fact at its **domain-time** validity, so it answers "what was true at time T"
directly. On our TempBelief benchmark (facts that change over time, some revealed out
of order), point-in-time accuracy:

- **GENOME belief-state: 0.870**
- Mem0 (run at its best, with history traversal): 0.676  — GENOME wins, p=0.0007
- dense retrieval: 0.407
- full-context (the whole transcript in the prompt): 0.139

It beats every baseline with significance, and a full six-conversation audit finds
the underlying knowledge graph correct (0.97 precision, 0.96 recall, and every
captured fact dated correctly) — this isn't the judge being lenient. Honest tradeoff: on trivial "what's the *current*
value" lookups GENOME is slightly *behind* (0.833 vs ~0.97), because there raw
retrieval just reads the last mention.

**The honest caveat:** this is measured on synthetic conversations where every fact
states an explicit date ("in March 2024, ..."), which our date parser reads directly.
Real dialogue rarely states dates that cleanly ("last year", "when I started this
job"), so on real text the advantage would shrink. The mechanism and the win are real;
proving it survives natural, relative-dated speech is the next experiment. We're
telling you that up front.

## And on LoCoMo (1,540 questions), here's the rest.

## Ingestion is essentially free

GENOME writes a memory by embedding it — no LLM in the write path. LLM-extraction
layers like Mem0 call a model on every message. We instrumented Mem0's own LLM
client and counted:

| | LLM calls / message | Cost (80 turns) | Full corpus (est.) |
|---|---|---|---|
| Mem0 | 1.00 | $0.71 | ~$52, ~3.3 hrs |
| **GENOME** | **0** | **$0.00008** | **~$0.006** |

**Zero LLM calls at ingest vs 1.00/message — a deterministic, model-independent
gap.** At Haiku pricing that's ~8,400x cheaper per message and 9x faster; a cheaper
extraction model shrinks the dollar ratio but GENOME's ingest LLM cost stays
structurally zero. (Dollar figures are a single 80-turn run at Haiku pricing; we'd
also expect Mem0's per-message cost to climb as its store fills, though we didn't
measure that curve here.)

## When history overflows the window, retrieval wins big

A memory layer earns its keep when the conversation no longer fits the context
window. On a 284,000-token history (we truncated full-context from *both* ends so
the baseline isn't a strawman):

- **GENOME: 0.817 accuracy at 1,602 tokens/query**
- best full-context (recency window, last 128k): 0.408
- prefix full-context (first 128k): 0.300
- **+0.409 accuracy at 80x less context, vs the stronger baseline** — and GENOME
  beats both truncation directions in every category.

## And here's what we *don't* claim

Because an honest benchmark is the whole point:

- **In-window accuracy: no confirmed difference.** GENOME 0.851 vs Mem0 0.855 vs
  full-context 0.863. Paired McNemar p > 0.23 — not significant, and GENOME's point
  estimate is nominally the lowest. LoCoMo's in-window ceiling is ~0.85 and everyone
  hits it. We match state of the art; we don't beat it, and we won't pretend to.
- **Our synthesis feature is a wash at equal budget.** Summarizing old memories
  instead of forgetting them helps cross-session questions and hurts exact-fact
  lookup — net zero. It's a tunable tradeoff, not a free win.
- **Hybrid retrieval is worse than plain dense; graph retrieval we couldn't fairly
  test** here (the offline harness didn't build the entity graph it needs). Dense is
  the validated default; we make no claim for graph.

## Why publish the nulls?

Because a memory layer that only reports its wins can't be trusted on them. The
efficiency and overflow results above are real *because* we told you where GENOME
is merely average. Every figure regenerates from `benchmarks/` — run them yourself.

**Use GENOME when** ingestion cost matters, when history exceeds the window, or
when you want a memory layer whose behavior you can reason about. **Don't expect** a
higher in-window LoCoMo score than Mem0 — that number is saturated, and we won't
pretend otherwise.
