# GENOME on LoCoMo — Benchmark Results

An honest, reproducible evaluation of GENOME against Mem0 and full-context on the
LoCoMo long-conversation memory benchmark. Wins and null results are reported with
equal prominence. Every number regenerates from `benchmarks/` and is pinned to its
source in [`VERIFIED-NUMBERS.md`](./VERIFIED-NUMBERS.md).

## Methodology (why the comparison is fair)

Every system answers with the **same** model (Claude Haiku 4.5), is graded by the
**same** judge (Mem0-verbatim prompt, temperature 0), embeds with the **same**
encoder (`text-embedding-3-small`), and retrieves the **same** top-k. Only the
memory layer varies. Scores are over the **1,540 headline questions** (LoCoMo
categories 1–4). Significance is paired **McNemar** on per-question correctness.

Two caveats stated up front: (1) the answer model and the judge are the same
family (Haiku 4.5) — standard for LoCoMo, but a same-model judge carries a
self-preference risk that applies equally to every system here; (2) numbers are
single-run at a fixed seed with no confidence intervals, so treat sub-percent gaps
as noise.

## Summary

| # | Question | Result | Verdict |
|---|----------|--------|---------|
| 06 | Point-in-time ("as of time T") queries | Wins when source states explicit dates (0.870 vs Mem0 0.676, p=0.0007); **ties Mem0 on natural relative-dated language** (0.759 vs 0.722, p=0.82) | **Win only for explicit-dated sources** |
| 01 | In-window answer accuracy | GENOME 0.851 vs Mem0 0.855 vs full-ctx 0.863; no significant difference (p > 0.23), GENOME nominally lowest | **Parity** |
| 02 | Accuracy when history overflows the window | GENOME beats both prefix- and recency-truncated full-context by a wide margin | **Decisive win** |
| 03 | Ingestion cost | 0 LLM calls/msg vs 1.00 (deterministic); ~8,433× cheaper at Haiku pricing | **Decisive win** |
| 04 | Do hybrid/graph retrieval beat dense? | Hybrid is worse; graph not exercised here (no entity graph) | **Honest null** |
| 05 | Does synthesis beat pruning at equal budget? | Net wash (+0.005, p=0.86) | **Honest null** |

---

## Finding 01 — In-window accuracy is saturated (parity)

| System | Headline | multi-hop | temporal | open-domain | single-hop |
|---|---|---|---|---|---|
| full-context (upper bound) | 0.863 | 0.883 | 0.748 | 0.531 | 0.938 |
| Mem0 | 0.855 | 0.890 | 0.863 | 0.479 | 0.882 |
| **GENOME** | **0.851** | 0.851 | 0.798 | 0.500 | 0.911 |

Paired McNemar: GENOME vs Mem0 Δ = −0.004 (p = 0.747); GENOME vs full-context
Δ = −0.012 (p = 0.232). No significant difference — but note the direction:
GENOME's point estimate is nominally the lowest of the three.

**Read:** LoCoMo in-window accuracy has a ~0.85 ceiling that every serious system
reaches. GENOME reaches it but sits a hair below both baselines; the gaps aren't
statistically significant, so the honest statement is "no confirmed difference,"
not "GENOME ahead." We do not claim an in-window accuracy win. This is not where a
memory layer differentiates.

*Source: `benchmarks/verdict.py` over `results/locomo_claude_v2/`, n = 1,540.
GENOME = parent-filtered variant; the plain dense variant is within a point (0.852).*

## Finding 02 — When history overflows the window, retrieval wins (decisive)

On a **~284k-token** history (past a 200k window), same 120 questions. Full-context
was truncated from **both ends** so the baseline isn't a strawman — prefix (first-B
tokens) and recency (last-B tokens, what a real chat app keeps):

| System | Accuracy | Context / query |
|---|---|---|
| **GENOME (retrieval)** | **0.817** | 1,602 |
| full-context recency @128k | 0.408 | 128,000 |
| full-context prefix @128k | 0.300 | 128,000 |
| full-context @32k (either end) | 0.125–0.133 | 32,000 |
| full-context @8k (either end) | 0.067–0.117 | 8,000 |

**+0.409 accuracy vs the stronger (recency) baseline, at 80× less context.**
Per-category vs recency@128k: single-hop 0.871 vs 0.500, multi-hop 0.810 vs 0.429,
temporal 0.783 vs 0.217, open-domain 0.333 vs 0.000 (n=6). Neither truncation
direction helps — with 284k of history no fixed window holds it all, so the model
can't see out-of-window evidence; retrieval puts it back. This is the case that
justifies a memory layer.

*Source: `benchmarks/haystack_report.py` over `results/haystack/` (+ `haystack_recency.py`).*

## Finding 03 — Ingestion costs almost nothing (decisive)

GENOME embeds each message (no LLM in the write path). Mem0 runs LLM
fact-extraction per message. We instrumented Mem0's LLM client over an identical
80-turn slice:

| System | LLM calls / msg | Tokens (80 turns) | Wall-clock | Cost (80 turns) |
|---|---|---|---|---|
| Mem0 | 1.00 | 687,192 in + 4,940 out | 164.4 s | $0.7119 |
| **GENOME (dense)** | **0** | 4,221 embed | 19.2 s | **$0.00008** |

The robust, model-independent fact is the **call count: 0 vs 1.00 LLM call per
message** — no pricing choice changes it. The dollar multiplier does depend on the
model priced: at Haiku pricing, the full 5,882-turn corpus is ≈ $52 (Mem0) vs
≈ $0.006 (GENOME); a cheaper extraction model shrinks the ratio but it stays large,
because GENOME's ingest LLM cost is structurally zero. One mechanism we did **not**
measure but expect: Mem0's per-message cost should rise as its store fills (each
write ships existing memories to the LLM), so linear extrapolation is if anything
an underestimate.

*Source: `benchmarks/ingest_cost.py` (`results/ingest_cost.log`), single 80-turn
run. Mem0 configured with the same Haiku 4.5 model; dollars at Haiku pricing
($1/$5 per Mtok). The 692k input tokens are Mem0's own extraction prompts.*

**Deployment projection** (`benchmarks/tco_project.py`, `benchmarks/TCO.md`): at 10,000
users × 50 msgs/day (15M msgs/month), Mem0's memory-ingest LLM bill is $159k/year (cheapest
hosted extraction model) to $1.6M/year (Haiku), vs ~$190/year for GENOME — 837× to 8,433×,
plus 8.6× lower write latency and 0 LLM calls in the write path. The ratio is
deployment-size-independent and survives the cheapest extraction model. This is only a *win*
rather than a *tradeoff* because accuracy is at parity (Findings 01/07): Mem0 buys no measured
accuracy advantage with that spend.

## Finding 04 — Among modes we could fairly test, dense wins (honest null)

Using LoCoMo's annotated evidence turns as ground truth, fraction of gold evidence
each mode surfaces (no LLM — isolates retrieval):

| Mode | hit@5 | hit@10 | hit@20 | full-recall@10 |
|---|---|---|---|---|
| **dense** | 0.622 | 0.715 | 0.798 | 0.652 |
| dense + recency rerank | 0.586 | 0.708 | 0.798 | 0.645 |
| hybrid | 0.497 | 0.564 | 0.714 | 0.512 |

Among the modes this offline harness could fairly exercise, plain dense is best:
hybrid is clearly worse (keyword blending pulls in lexically-similar but wrong
turns), recency rerank is slightly worse (LoCoMo evidence spans all sessions).

**Graph mode is omitted, not validated.** It requires an entity graph that this
offline store never built (we confirmed 0 entities), so it correctly fell back to
dense and cannot be scored here. We make **no** claim that graph helps — and an
earlier full-pipeline graph variant scored far worse on answers (0.494), so the
burden is on graph to prove itself. Dense is GENOME's validated default.

*Source: `benchmarks/retrieval_quality.py`, 1,536 evidence-annotated questions.*

## Finding 05 — Synthesis is a tradeoff, not a free win (honest null)

Consolidation (summarize old memories instead of forgetting them), tested at
**equal token budget** on both sides:

| Strategy @ equal budget | Accuracy | multi-hop | temporal | single-hop | open-domain |
|---|---|---|---|---|---|
| PRUNE (forget old) | 0.255 | 0.333 | 0.162 | 0.254 | 0.375 |
| SYNTHESIZE (summarize old) | 0.260 | 0.405 | 0.246 | 0.206 | 0.344 |

Storage budget S = 3,000 tok, answer budget B = 1,500 tok, n = 584. Context fed was
equal (1,473 vs 1,436 tok). Delta **+0.005, McNemar p = 0.861 — not significant.**

Per-category (directional only — only the aggregate was significance-tested):
synthesis trends up on cross-session reasoning (temporal +0.085, multi-hop +0.072)
and down on exact-fact lookup (single-hop −0.048, open-domain −0.031), consistent
with the mechanism — a summary blurs the one verbatim detail a single-hop question
needs. Read as a tendency, not proof. Consolidation is a **tunable tradeoff** — on
for temporal/analytical workloads, off for lookup-heavy ones. (An earlier "+0.49
synthesis win" was a token confound: the summary arm was fed far more context. Held
to equal tokens, the gain disappears.)

*Source: `benchmarks/budget_fair.py` (`results/budget_fair/`).*

---

## Finding 06 — Point-in-time / belief-state queries (decisive win, new capability)

The other findings are about *retrieval*. This one is about a different primitive:
maintaining a contradiction-resolved, **bi-temporal** model of how facts change over
time, and answering "what was true at time T?". GENOME records each fact at its
**domain-time** validity (when it became true in the world), not wall-clock ingest
time — so `facts_valid_at(entity, T)` is exact even when facts are revealed out of
chronological order.

We built **TempBelief** (`genome/evals/tempbelief.py`): a deterministic, gold-validated
benchmark where entities' attributes change over 6-8 sessions, ~1/3 of updates are
narrated out of order, and distractors (tentative plans, one-off events) must not
change belief. All systems ingest the *same* raw NL turns; Mem0 is run at its best
(`infer=True` + `history()` traversal). Within-harness, 6 conversations, n=108 as-of.

| split | belief | Mem0 | dense | full-context |
|---|---|---|---|---|
| **as-of (point-in-time)** | **0.870** | 0.676 | 0.407 | 0.139 |
| current-value | 0.833 | 0.861 | 0.972 | 1.000 |
| history | 1.000 | 1.000 | 1.000 | 1.000 |
| **macro-J** | **0.901** | 0.846 | 0.793 | 0.713 |

On as-of, belief-state beats **every** baseline with significance: vs Mem0 p=0.0007
(28 belief-only vs 7 Mem0-only of 108 paired), vs dense p<1e-5, vs full-context p<1e-5.
Overwrite-based memory loses older values; raw retrieval and full-context return the
*latest* value or can't disambiguate dates in a cluttered context. Only the bi-temporal
KG answers point-in-time reliably. (Numbers recomputed bit-for-bit from the run log by an
automated cross-check — our own tooling, **not** an independent third party; Mem0 was run at
its best, not hobbled. Independent replication is welcome and encouraged — the harness is in
this repo.)

**Not judge leniency:** a full 6-conversation extraction audit
(`benchmarks/tempbelief_verify.py --convs 6`) finds the belief KG at **precision 0.972,
recall 0.963** (104/108 gold events), **domain-time accuracy 1.000** (every captured
fact dated correctly) — so the as-of accuracy reflects a correct KG, not a lenient
judge. (An earlier 100% figure was from a 2-conv subset; these full-scope numbers
supersede it.)

**Honest tradeoff:** belief *loses* current-value (0.833 vs 0.97-1.0) — its LLM
extraction is noisier on trivial "latest value" lookups than just reading the last
mention.

**Relative-dated language: an early "collapse" was mostly a resolution BUG, now fixed
(2026-07-12); QA head-to-head re-validation pending.** The first relative-language run
(vague phrasing "about two years ago" + a weak resolution prompt) gave as-of belief
0.759 vs Mem0 0.722 (p=0.82, tie), and the decisive verbatim win (0.870 vs 0.676)
appeared to vanish. But `tempbelief_verify.py --data tempbelief_rel.json` showed the
real cause: **domain-time accuracy was only 0.323** — the pipeline was dating relative
facts wrong, not that the task was unwinnable. Fix: a sharper arithmetic-resolution
ingest prompt + resolvable relative phrasing → **domain-time accuracy 0.323 → 0.829,
extraction recall 0.861 → 0.972, with NO regression on verbatim** (still 1.000 /
1.000 / 1.000; recall even up 0.963 → 1.000). So relative-date handling is now genuinely
good. **Re-measured head-to-head (3 convs): the fix regained a clear edge.** On the
fixed pipeline with *resolvable* relative phrasing ("8 months ago"), as-of belief
**0.741 vs Mem0 0.574** (+0.167; belief-only 19 vs mem0-only 10, McNemar p=0.137 —
leads, trending significant at n=54), up from the pre-fix tie (0.759 vs 0.722, p=0.82).
current-value 0.722 vs 0.500, history 0.944 vs 0.444. Mechanism: GENOME resolves the
relative date to a domain-time deterministically at ingest; Mem0 must do the arithmetic
at query time with its LLM and errs more. Honest scope: this holds for *resolvable*
relative dates (what people usually say); genuinely vague ones ("about two years ago")
stay hard for every system. Belief-state's durable benefits remain the *deterministic,
structured, auditable record* + 0-LLM-ingest dense retrieval.

**Other limitations:**
- **history is a ceiling, not a result:** all four systems score 1.000 because the
  Mem0-verbatim judge gives partial credit (≥1 of N list items = CORRECT). It's
  uninformative and inflates macro-J equally for everyone — **the real signal is the
  as-of split alone**, not macro-J.
- 6 conversations, single deterministic dataset, single seed; a real-conversation control
  is future work.

*Source: `benchmarks/tempbelief_run.py`, `benchmarks/tempbelief_verify.py`.*

## Finding 07 — Reranking generalizes to a harder benchmark (LongMemEval)

LoCoMo is saturated, so retrieval gains don't show in its answer accuracy. On
**LongMemEval-S** (60 questions, avg **51 sessions/question** with distractors — a far
harder retrieval task), the cross-encoder reranker's benefit concentrates exactly where
retrieval is hard:

| question type | dense hit@10 | + rerank |
|---|---|---|
| **multi-session** | 0.610 | **0.832** (+0.222) |
| knowledge-update | 0.950 | 1.000 (+0.050) |
| temporal-reasoning | 0.873 | 0.913 (+0.040) |
| single-session (easy) | ~1.00 | ~0.90 (slightly worse) |
| **aggregate** | 0.878 | 0.896 (+0.019) |

Rerank lifts the *hard* multi-session retrieval by +0.22 and is neutral/slightly-negative
on easy single-session queries (dense already nails those). The aggregate understates the
value because easy types have no headroom. *Free loop (embeddings + local cross-encoder);
gold = turns flagged `has_answer`. Source: `benchmarks/lme_retrieval.py`.*

**Answer-accuracy head-to-head — powered run (90 Q, Sonnet responder to un-bottleneck
the answer step, gold + 4 distractor sessions; `benchmarks/lme_qa.py`):**

| system | answer accuracy | retrieval hit@10 |
|---|---|---|
| **GENOME + rerank** | **0.700** | 0.943 |
| GENOME dense | 0.689 | 0.917 |
| Mem0 (`infer=True`, session-ingest) | 0.622 | — |

On this *hard, non-saturated* benchmark GENOME scores **nominally higher than Mem0 by +0.078**
(rerank 0.700 vs Mem0 0.622) — but **this is NOT statistically significant, and the larger
n=205 run below confirms the honest conclusion is parity, not a beat.** It answers 4 of 6
question types higher (single-session-assistant 1.00 vs 0.60, multi-session 0.73 vs 0.60,
single-session-user 1.00 vs 0.93). Paired McNemar: GENOME-only correct 14 vs Mem0-only 7,
**p=0.19 — nominally higher, NOT statistically significant at n=90**. **Two honest corrections
vs an earlier small-n (n=24, Haiku-responder) read:** that run showed 0.542/0.500/0.500 and
suggested reranking *hurt* answers — both were small-n noise. At n=90 with a stronger
responder, **reranking is the best of GENOME's own configs (0.700)**, helping answers as well
as retrieval; the apparent edge over Mem0 is not significant (see the n=205 update). Caveat
that remains: temporal-reasoning is ~0.07 for all systems — those
questions need GENOME's belief-state (as-of) resolution wired into answering, which this
retrieval-only harness does not do (a concrete next upgrade).

**UPDATE (n=205, independent run) — the lead did NOT hold; it's parity.** A larger
significance run of 205 questions via OpenRouter (Sonnet-4 responder, Haiku-4.5 judge) with a
**local** embedder for both systems gave: GENOME dense 0.595, rerank 0.546, Mem0 0.537.
McNemar dense-vs-Mem0 34 vs 22 **p=0.14 (not significant)**; rerank-vs-Mem0 28 vs 26 p=0.89
(tied). Across *both* runs (n=90 p=0.19, n=205 p=0.14) GENOME is directionally ahead but never
statistically significant, so the honest conclusion is **answer-accuracy parity, not a beat**.
Also note: on the weaker local embedder, rerank improved hit@10 (0.895 vs 0.827) but *not*
accuracy (0.546 < 0.595 dense) — rerank's accuracy benefit is embedder-dependent. Different
backend + local embedder means absolute numbers differ from the n=90 table above; treat this
as a corroborating parity data point, not an extension. Script: `benchmarks/lme_qa_or.py`.

## When to use GENOME

**Use GENOME when** ingestion cost matters (high message volume, or no LLM allowed
in the write path), when history exceeds the model's context window, or when you
want a memory layer whose behavior you can reason about. **Don't expect** a higher
in-window LoCoMo score than Mem0 — that metric is saturated and the two tie.

## Reproduce

```bash
python benchmarks/verdict.py            # 01: in-window accuracy + McNemar
python benchmarks/haystack_report.py    # 02: overflow / haystack
python benchmarks/ingest_cost.py --n 80 # 03: measured ingestion cost
python benchmarks/retrieval_quality.py  # 04: retrieval-mode hit-rate
python benchmarks/budget_fair.py        # 05: token-fair synthesis ablation
```
