# Feature-audit results index (2026-08-18)

Five opt-in features of GENOME, each measured under a controlled harness with
the cost of the feature reported next to its benefit. Wins and failures get
equal billing. Every table regenerates from the named script with one
OpenRouter key; local all-MiniLM embedder throughout.

| Feature | Verdict | Result file | Harness |
|---|---|---|---|
| Bi-temporal belief on NATURAL LANGUAGE | Decisive win: as-of 0.861 vs Mem0 0.593, p<0.0001 | [tempbelief_nl_result.txt](./tempbelief_nl_result.txt) | tempbelief_run_or.py |
| Conflict resolution (ADD/UPDATE/DELETE) | Works (15/16 stale removed) with 3 documented defects | [conflictbench_result.txt](./conflictbench_result.txt) | conflictbench_or.py |
| Graph retrieval | Honest null: +0.016 hit rate for ~1000x ingest cost | [ab_graph_or_result.txt](./ab_graph_or_result.txt) | ab_graph_retrieval_or.py |
| Auto-consolidation (shipped trigger) | Harmful at default target: 5x accuracy collapse | [consolidation_scale_result.txt](./consolidation_scale_result.txt) | consolidation_scale_or.py |
| Cross-encoder reranking | Retrieval feature: reliable hit@10 gain, answer effect small and inconsistent | [lme_embedder_sweep_result.txt](./lme_embedder_sweep_result.txt) | lme_qa_or.py --embedder ... |

Core-claim replication: GENOME vs Mem0 answer-accuracy parity has replicated
in four independent configurations (see the sweep file); a locked rerun of the
published head-to-head protocol lives at head_to_head_locked.txt when present.

Methodology notes that came out of this audit:
- Store audits must count POSITIVE assertions only; naive value-substring
  audits overstated conflict-resolution failures 5x before correction.
- TempBelief dataset regeneration requires PYTHONHASHSEED=0 (salted str hash
  in one offset); the committed JSONs in data/ are canonical.

The full evaluation behind the core system claims is RESULTS.md; the published
paper is https://doi.org/10.5281/zenodo.21987934.
