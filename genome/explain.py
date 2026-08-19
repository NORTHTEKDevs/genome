"""explain_search: the recall debugger.

Agent developers debug memory blind: a fact was stored, the agent did not recall
it, and the memory system offers no way to ask why. This module answers it, per
candidate: its dense score, its BM25 rank, its fused score, and - when it was NOT
returned - the exact reason (parent-filtered, quarantined, beyond the limit).

This is a dividend of determinism. The report re-derives the real pipeline, so
two runs agree byte for byte, and a failing recall can be committed as a test. A
memory system whose ingest rewrites content through an LLM cannot reproduce its
own state, let alone explain it.

The report never mutates anything: no cache writes, no access-count touches.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genome.firewall import is_quarantined


@dataclass(frozen=True)
class CandidateExplanation:
    id: str
    content: str
    dense_score: float | None
    dense_rank: int | None
    bm25_rank: int | None
    fused_score: float | None
    final_rank: int | None
    included: bool
    reason: str


@dataclass(frozen=True)
class ExplainReport:
    query: str
    mode: str
    user_id: str | None
    agent_id: str | None
    limit: int
    candidates: tuple[CandidateExplanation, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        lines = [
            f"explain_search({self.query!r}, mode={self.mode}, limit={self.limit})"
        ]
        for c in self.candidates:
            mark = f"#{c.final_rank}" if c.included else "--"
            score = (
                f"fused={c.fused_score:.4f}"
                if c.fused_score is not None
                else f"dense={c.dense_score:.4f}"
                if c.dense_score is not None
                else "unscored"
            )
            lines.append(f"  {mark:>3}  {score}  {c.reason}  | {c.content[:60]}")
        return "\n".join(lines)


def explain_search(
    memory: Any,
    query: str,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 10,
    mode: str = "dense",
) -> ExplainReport:
    """Re-derive a search and report every candidate's fate.

    Mirrors ``Memory.search`` (dense and hybrid modes) with instrumentation.
    Bypasses the response cache and does not touch access statistics, so
    explaining a query never changes subsequent behavior.
    """
    if mode not in {"dense", "hybrid"}:
        raise ValueError(f"explain_search supports dense/hybrid, got {mode!r}")

    in_scope = memory.store.list_by_scope(user_id=user_id, agent_id=agent_id)
    by_id = {rec.id: rec for rec in in_scope}

    # The two exclusion sets search() applies, kept separate so the reason is exact.
    parent_ids: set[str] = set()
    for rec in in_scope:
        parent_ids.update(rec.parents)
    policy = getattr(memory, "_trust_policy", None)
    quarantined = {rec.id for rec in in_scope if is_quarantined(rec, policy)}

    # Dense scores for EVERYTHING in scope - candidates are diagnosed, not dropped.
    q = memory.embed.encode(query)
    dense_all = memory.store.search(
        q, user_id=user_id, agent_id=agent_id, limit=max(1, len(in_scope))
    )
    dense_score: dict[str, float] = {r.id: float(r.score) for r in dense_all}
    dense_rank: dict[str, int] = {
        r.id: rank for rank, r in enumerate(dense_all, start=1)
    }

    bm25_rank: dict[str, int] = {}
    fused_score: dict[str, float] = {}
    if mode == "hybrid":
        from rank_bm25 import BM25Okapi

        from genome.memory.hybrid import HybridScorer, _tokenize

        eligible = [
            rec for rec in in_scope
            if rec.id not in parent_ids and rec.id not in quarantined
        ]
        corpus = {rec.id: rec.content for rec in eligible}
        if corpus:
            ids = list(corpus)
            bm25 = BM25Okapi([_tokenize(corpus[i]) for i in ids])
            scores = bm25.get_scores(_tokenize(query))
            ranked = sorted(
                zip(ids, scores, strict=True), key=lambda p: p[1], reverse=True
            )
            for rank, (rid, score) in enumerate(ranked, start=1):
                if score > 0:
                    bm25_rank[rid] = rank
            dense_pairs = [
                (r.id, float(r.score))
                for r in dense_all
                if r.id in corpus
            ]
            fused = HybridScorer().fuse(
                query=query, dense_results=dense_pairs, corpus=corpus
            )
            fused_score = {rid: float(s) for rid, s in fused}

    # Final ordering, derived exactly as search() derives it (exclusions, then
    # the mode's ranking, then the limit) - but without the store touches the
    # real path performs. test_report_matches_search_order pins the equivalence.
    excluded = parent_ids | quarantined
    if mode == "hybrid" and fused_score:
        eligible_order = sorted(fused_score, key=fused_score.__getitem__, reverse=True)
    else:
        eligible_order = [r.id for r in dense_all if r.id not in excluded]
    final_ids = [rid for rid in eligible_order if rid not in excluded][:limit]
    final_rank = {rid: rank for rank, rid in enumerate(final_ids, start=1)}

    ordering = sorted(
        by_id,
        key=lambda rid: (
            final_rank.get(rid, 10**9),
            -(fused_score.get(rid) or dense_score.get(rid) or 0.0),
        ),
    )

    candidates: list[CandidateExplanation] = []
    for rid in ordering:
        rec = by_id[rid]
        if rid in final_rank:
            included, reason = True, f"included at rank {final_rank[rid]}"
        elif rid in parent_ids:
            included, reason = False, "excluded: parent of a synthesized memory"
        elif rid in quarantined:
            trust = (rec.metadata.get("_provenance") or {}).get("trust")
            included, reason = False, (
                f"excluded: quarantined by trust policy (trust {trust} < "
                f"min {policy.recall_min_trust})"
            )
        else:
            included, reason = False, f"excluded: beyond the top-{limit} limit"
        candidates.append(
            CandidateExplanation(
                id=rid,
                content=rec.content,
                dense_score=dense_score.get(rid),
                dense_rank=dense_rank.get(rid),
                bm25_rank=bm25_rank.get(rid),
                fused_score=fused_score.get(rid),
                final_rank=final_rank.get(rid),
                included=included,
                reason=reason,
            )
        )
    return ExplainReport(
        query=query,
        mode=mode,
        user_id=user_id,
        agent_id=agent_id,
        limit=limit,
        candidates=tuple(candidates),
    )
