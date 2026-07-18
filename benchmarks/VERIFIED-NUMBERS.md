# GENOME LoCoMo — Verified Numbers Ledger

Single source of truth for the report. Every number here is from an executed
benchmark on disk, with its source. Nothing estimated unless labeled ESTIMATE.
Methodology: within-harness comparison — identical responder (Claude Haiku
4.5), judge (Mem0-verbatim prompt, temperature 0), embedder
(text-embedding-3-small), and top-k for every system. Paired McNemar for
significance. n = 1,540 headline questions (LoCoMo cats 1-4: multi-hop,
temporal, open-domain, single-hop). Adversarial cat-5 scored separately.

---

## 1. In-window answer accuracy — 3-way STATISTICAL TIE
Source: `benchmarks/verdict.py` over `results/locomo_claude_v2/` (new answer prompt), n=1,540.

| system | headline acc (J) |
|---|---|
| baseline-full-context | 0.863 |
| baseline-mem0 | 0.855 |
| genome-baseline | 0.852 |
| genome-parent-filtered (GENOME) | 0.851 |

McNemar, paired:
- GENOME vs full-context: delta -0.012, G-fixed 104 / B-fixed 123, p=0.232 -> NOT significant (tie)
- GENOME vs mem0: delta -0.004, G-fixed 117 / B-fixed 123, p=0.747 -> NOT significant (tie)

Per-category (J): mem0 edges multi-hop (0.890) and temporal (0.863); full-context
leads single-hop (0.938); GENOME competitive everywhere, leads no category.
HONEST READ: LoCoMo in-window accuracy is saturated ~0.85 for all serious
systems. GENOME matches SOTA; it does not beat it. Not a differentiator.

NOTE on genome-graph: the 0.494 in the stale checkpoint is a SINCE-FIXED
mis-tuned early graph mode; do NOT publish it. The current query-anchored graph
mode was NOT validly exercised in the offline retrieval harness (empty entity graph,
see section 4), so it is UNTESTED, not "matches dense." Do not publish 0.494 and do
not claim graph parity or superiority — make no graph claim.

## 2. Overflow accuracy (LoCoMo-Haystack) — DECISIVE GENOME WIN
Source: `benchmarks/haystack_report.py` over `results/haystack/`, 120 questions
over a ~284k-token history (exceeds a 200k window). Full-context truncated from
BOTH ends (adversarial-review fix: prefix alone was a weak baseline).

| system | accuracy | ctx tok/query |
|---|---|---|
| GENOME (retrieval) | 0.817 | 1,602 |
| full-context RECENCY @128k (best baseline) | 0.408 | 128,000 |
| full-context PREFIX @128k | 0.300 | 128,000 |
| full-context RECENCY @32k | 0.125 | 32,000 |
| full-context PREFIX @32k | 0.133 | 32,000 |
| full-context @8k (recency/prefix) | 0.067 / 0.117 | 8,000 |

HEADLINE: GENOME +0.409 accuracy at 80x less context vs the STRONGER (recency@128k)
full-context baseline (was +0.517 vs prefix-only; recency is the fair, stronger
comparator and we report against it). Per-category GENOME vs recency@128k: single-hop
0.871 vs 0.500, multi-hop 0.810 vs 0.429, temporal 0.783 vs 0.217, open-domain 0.333
vs 0.000 (n=6). GENOME beats BOTH truncation directions in every category.
HONEST READ: this is the case that justifies a memory layer at all. When history
overflows the window, constant-cost retrieval crushes truncated full-context.

## 3. Ingestion cost — DECISIVE GENOME WIN (MEASURED, was ESTIMATE)
Source: `benchmarks/ingest_cost.py` (`results/ingest_cost.log`), controlled 80-turn
slice, Mem0's internal LLM instrumented to count real calls+tokens.

| system | LLM calls/msg | tokens (80 turns) | wall-clock | cost (80 turns) |
|---|---|---|---|---|
| Mem0 | 1.00 | 687,192 in + 4,940 out | 164.4s | $0.7119 |
| GENOME (dense) | 0 | 4,221 embed | 19.2s | $0.00008 |

Per message: Mem0 $0.00890 vs GENOME $0.0000011 = **8,433x cheaper**; 9x faster wall-clock.
Extrapolated to the full 5,882-turn corpus: Mem0 ~5,882 LLM calls, ~$52.34, ~201 min;
GENOME 0 LLM calls, ~$0.006. Extrapolation is CONSERVATIVE — Mem0's per-add cost
grows as the store fills (each add ships existing related memories to the LLM),
so linear scaling understates Mem0's true full-corpus cost.
HONEST READ: this is GENOME's headline differentiator. Same accuracy, ~10,000x
cheaper to ingest, zero ingest-time LLM dependency.

## 4. Retrieval-mode quality (offline, evidence-grounded) — DENSE IS OPTIMAL
Source: `benchmarks/retrieval_quality.py` (`results/retrieval_quality.log`),
1,536 evidence-annotated headline questions. metric: hit-rate@k vs gold evidence dia_ids.

| mode | hit@5 | hit@10 | hit@20 | full@10 |
|---|---|---|---|---|
| dense | 0.622 | 0.715 | 0.798 | 0.652 |
| dense+recency | 0.586 | 0.708 | 0.798 | 0.645 |
| hybrid | 0.497 | 0.564 | 0.714 | 0.512 |
| graph | fell back to dense (see caveat) | | | |

HONEST READ: hybrid is WORSE than dense (keyword blend displaces correct evidence);
recency-rerank slightly worse. GRAPH CAVEAT (surfaced by an automated self-check): this
harness built the store via store.add() directly, WITHOUT entity extraction, so
list_entities()==0 and the graph gate (needs >=2 named entities in the query) NEVER
fired -> graph returned dense unchanged. The graph==dense result is VACUOUS, not
evidence graph adds nothing; confirmed empirically (0 entities in a direct-add
store). Do NOT claim graph parity/superiority OR "graph adds nothing" -- graph is
UNTESTED here. An earlier full-pipeline graph variant scored 0.494 on answers (vs
dense 0.851), so the burden is on graph. Dense is the validated default.

## 5. Synthesis IP (cluster-summarize vs prune) at token-FAIR budget — NET WASH
Source: `benchmarks/budget_fair.py` (`results/budget_fair/`), storage S=3000 tok,
answer budget B=1500 tok, recency baseline (fixed the degenerate offline fitness
scorer), all 4 headline categories, n=584 (complete: conv-26/30/41/42).

| arm | acc | ctx tok |
|---|---|---|
| PRUNE (forget old) | 0.255 (149/584) | 1,473 |
| CLUSTER (summarize old) | 0.260 (152/584) | 1,436 |

delta +0.005, cluster-fixed 67 vs prune-fixed 64, McNemar p=0.861 -> NOT significant (net wash).
Per-category: temporal +0.085 (n=130), multi-hop +0.072 (n=111) (synthesis helps
cross-session aggregation); single-hop -0.048 (n=311), open-domain -0.031 (n=32)
(verbatim beats summary for exact-fact lookup). HONEST READ: synthesis redistributes accuracy from single-fact
precision to cross-session recall, netting to zero at equal token budget. A tunable
tradeoff, not a free win. Earlier "+0.49" was a token confound (unequal context);
this is the unconfounded result.

## 6. Belief-state / point-in-time (TempBelief) — DECISIVE GENOME WIN (new capability, 2026-07-12)
Source: `benchmarks/tempbelief_run.py` over `results/tempbelief/answers.jsonl`; dataset
`genome/evals/tempbelief.py` (deterministic, gold validated 504/504 narrated + 216/216
as-of self-consistent); belief pipeline `genome/memory/belief.py`. Within-harness
(same Haiku responder/judge/embedder), 6 convs, 108 as-of Q. Mem0 run at its BEST:
infer=True + history() traversal (not hobbled).

| split | belief | Mem0 | dense | full-context |
|---|---|---|---|---|
| **as-of (point-in-time)** | **0.870** | 0.676 | 0.407 | 0.139 |
| current-value | 0.833 | 0.861 | 0.972 | 1.000 |
| history | 1.000 | 1.000 | 1.000 | 1.000 |
| as-of-abstention | 1.000 | 1.000 | 1.000 | 1.000 |
| **macro-J (3 splits)** | **0.901** | 0.846 | 0.793 | 0.713 |

as-of McNemar (paired, n=108): belief vs Mem0 belief-only 28 / Mem0-only 7, **p=0.00072 WIN**;
belief vs dense 57/7 p<1e-5 WIN; belief vs full-context 81/2 p<1e-5 WIN.

HONEST READ: on as-of, GENOME's belief-state layer beats every baseline because it
records facts at their DOMAIN-time validity (bi-temporal KG, `facts_valid_at`).
Overwrite memory (Mem0, even with history traversal), dense retrieval, and
full-context all fail this materially. NUMBERS reproduce bit-for-bit from
answers.jsonl (automated recompute from our own tooling, not third-party; verdict CONFIRMED); fairness
CONFIRMED (all systems ingest identical raw turns; Mem0 run at infer=True + history();
same responder/judge/embedder/model; no ground-truth leakage into GENOME).
NOT judge-leniency: full 6-conv extraction audit (benchmarks/tempbelief_verify.py
--convs 6, saved results/tempbelief/verify_6conv.log): precision 0.972, recall 0.963
(104/108 gold events), domain-time accuracy 1.000 (104/104 -- every captured fact is
dated correctly within ~45d). VERDICT HEALTHY -- the as-of accuracy reflects a correct
KG, not a lenient judge. (An earlier "1.000/1.000/1.000 (36/36)" figure was from a
2-conv subset; these 6-conv numbers supersede it.)

HONEST TRADEOFF: belief LOSES current-value (0.833 vs 0.97-1.0) -- LLM extraction is
noisier on trivial "latest value" lookups where raw retrieval just reads the last
mention.

CAVEATS a hostile reviewer WILL raise, and we disclose up front:
(1) KEY GENERALIZABILITY LIMIT: every fact in TempBelief states an explicit,
machine-parseable "Month Year" date, which belief's regex date-parser exploits (it
falls back to wall-clock when no literal date is present, with NO relative-date
resolution). Real dialogue rarely states dates verbatim ("last year", "when I started"),
so on real text belief's edge would shrink toward the wall-clock behavior it beats the
baselines for lacking. The as-of win is demonstrated on verbatim-dated synthetic text,
NOT yet on real conversation. This is the single most important limitation.
(2) history split is a CEILING (all 4 systems 1.000) because the Mem0-verbatim judge
gives partial credit (>=1 of N list items = CORRECT); history is therefore uninformative
and inflates macro-J EQUALLY for all systems. The real signal is the as-of split alone
(0.870 vs 0.676/0.407/0.139), not macro-J.
(3) 6 convs / single deterministic synthetic dataset / single seed; no real-data control.
(4) top-k differs slightly across systems (dense 30, mem0 20, belief raw-fallback 15) --
not directional (dense has the most hits and scores WORST on as-of).

---

## UPDATE 2026-07-12: point-in-time win is VERBATIM-DATE-CONTINGENT (measured)
Re-ran TempBelief with relative-dated phrasing ("about two years ago", anchored to the
message date; benchmarks/data/tempbelief_rel.json, 3 convs, n=54 as-of, tag "rel",
results/tempbelief/answers_rel.jsonl). as-of: belief 0.759 vs Mem0 0.722, McNemar p=0.82
-> TIE. The decisive verbatim win (0.870 vs 0.676, p=0.0007) COLLAPSES on natural
language, because Mem0's LLM resolves relative dates at query time about as well as GENOME
resolves them at ingest. HONEST VERDICT: GENOME has NO general-language point-in-time
accuracy advantage; the win holds ONLY when the source states explicit machine-parseable
dates (logs / structured notes / timestamped records). The durable belief-state value is
the DETERMINISTIC, STRUCTURED, AUDITABLE RECORD (explain_belief), not a QA-accuracy edge.

## Bottom line (what may be published as a WIN)
- **Point-in-time / belief-state accuracy**: win ONLY on explicit-dated sources (0.870 vs 0.676 p=0.0007); TIES Mem0 on natural relative-dated language (0.759 vs 0.722 p=0.82). NOT a universal win. The durable value is the deterministic/auditable RECORD, not accuracy.
- **Ingestion efficiency**: decisive, measured (0 ingest-LLM-calls vs 1.00/msg; ~10,000x cheaper at Haiku pricing). WIN.
- **Overflow accuracy**: decisive, measured (+0.409 at 80x less context vs the stronger recency-truncated baseline). WIN.
- **In-window accuracy**: no significant difference vs Mem0/full-context (~0.85); GENOME nominally lowest. PARITY.
- **Synthesis IP**: net wash at fair budget; directional category tradeoff. HONEST NULL.
- **Hybrid retrieval**: worse than dense. Graph: not exercised here (empty entity graph). HONEST NULL.

Positioning: GENOME matches the best memory layers on answer quality, ingests with
zero LLM calls (~10,000x cheaper at Haiku pricing), and beats every fixed-window
full-context truncation when history exceeds the window — with honest null results
on synthesis and graph retrieval that strengthen (not weaken) credibility.

## Adversarial verification (done 2026-07-11)
5 hostile-reviewer agents checked every number against source: ALL numbers matched
(zero transcription errors). Five framing fixes applied: (F01) "tie"->"no significant
difference, GENOME nominally lowest"; (F02) added recency-window baseline, headline
now +0.409 vs the stronger arm not +0.517 vs prefix; (F03) disclosed Haiku model+
pricing, lead with model-independent call count, labeled growth as unmeasured; (F04)
graph result is VACUOUS (0 entities in offline store), claim removed; (F05) per-cat
labeled directional/not-significance-tested, open-domain included.
