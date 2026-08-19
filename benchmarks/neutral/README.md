# Neutral multi-system memory benchmark

Agent-memory benchmarks have a trust problem. Every vendor publishes numbers from
a harness they designed, and independent reruns of vendor-reported scores have
differed by tens of points on the same public dataset. Nobody's self-reported
table settles anything, including ours.

This harness is the opposite bet:

- Every system answers the **same questions** through the **same responder
  model**, is scored by the **same judge**, and retrieves with the **same
  embedder tier**. Only the memory architecture differs.
- Results print with a **pairwise McNemar matrix** (paired, continuity
  corrected), not a bar chart, and a **full-disclosure block**: dataset slice,
  models, embedder, top-k.
- GENOME is **one row in the table, not the house**. When it loses a pairing,
  the harness prints that in the same font. Our published claim on LoCoMo is
  parity on accuracy, not a win; GENOME's edges are ingest cost, determinism,
  and local operation, and those are measured by other artifacts in
  `benchmarks/`.

## Run it

```bash
# offline plumbing test - no key, no network
python benchmarks/neutral/run.py --smoke

# real run, one OpenRouter key, local embedders everywhere
export OPENROUTER_API_KEY=sk-or-...
curl -L -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python benchmarks/neutral/run.py --provider openrouter --n 2 --q 10 \
  --systems genome,mem0,fullcontext
```

LoCoMo is CC BY-NC 4.0 (Snap Inc.); this repo does not redistribute it.

## Fairness rules (also binding on us)

1. One responder, one judge, shared by every system in a run.
2. Same embedder tier everywhere: either everyone local MiniLM, or everyone
   `text-embedding-3-small`. Never mixed.
3. Same top-k, same question slice, no per-system prompt tuning.
4. Report the McNemar pairing, not just headline accuracy - and print the
   small-sample warning when n is small.
5. Any adapter change that touches another system's path lands with a rerun.

## Adding your system

`adapters.py` defines a three-method protocol (`ingest`, `answer`, `close`);
an adapter is ~40 lines. See `ADAPTERS.md` for wiring notes on Zep and Letta.
If you think this harness treats your system unfairly, the fix is a pull
request, not a rebuttal thread - that is the point of the harness.
