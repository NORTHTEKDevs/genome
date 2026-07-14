"""RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Based on Sarthi et al., Stanford 2024 (https://arxiv.org/abs/2401.18059).

The idea: memories form a multi-tier tree. Leaf memories are atomic facts.
At each higher level, similar memories are clustered (we use k-means on the
embedding space), and each cluster is summarized by an LLM into a single
parent memory. Repeat until the tree has a sensible height.

Retrieval can then query at any level -- atomic facts for specific questions,
summaries for big-picture questions.

This is the public, papered version of "hierarchical compression" -- implemented
from the RAPTOR paper directly, not lifted from any proprietary source. The
clustering step uses scikit-learn; the summarization step uses any LLMCallFn.

Key design choices:
- `RaptorTree` is a new concept alongside regular memories -- atomic memories
  stay as-is, and tree nodes reference the memories they summarize via `parents`.
- Each tree node is itself a MemoryRecord, so it plugs straight into existing
  search, filtering, consolidation. The `operator` field is set to "raptor_summary"
  and `metadata["raptor_level"]` is set to the tree level (0 = leaves).
- If no LLM is configured, we fall back to simple "k memories summarized as
  'summary of: a; b; c'" textual concatenation -- still useful for retrieval
  because the embedding is the centroid of the cluster.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from genome.memory.extraction import LLMCallFn
from genome.memory.facade import Memory
from genome.memory.graph import DERIVED_FROM
from genome.memory.schema import MemoryRecord

RAPTOR_OPERATOR = "raptor_summary"


@dataclass
class RaptorBuildResult:
    """Summary of a RAPTOR tree build pass."""

    user_id: str | None
    agent_id: str | None
    levels: int
    level_counts: list[int]  # len == levels + 1, [leaves, level1, level2, ...]
    summaries_created: int


SUMMARIZE_PROMPT = """\
You are summarizing a cluster of related memories into a single higher-level
memory that captures the common theme.

Memories:
{items}

Write ONE concise summary (<= 1 sentence, <= 25 words). Do not use bullet points.
Do not start with "This summary describes". Write the summary directly.

Summary:
"""


def _default_summarizer(texts: list[str], llm_call: LLMCallFn | None) -> str:
    """Summarize a cluster. Uses llm_call if provided, else a textual fallback."""
    if not texts:
        return "empty cluster"
    if llm_call is None:
        return "summary of: " + " ; ".join(t[:80] for t in texts[:6])
    items = "\n".join(f"- {t}" for t in texts)
    response = llm_call(SUMMARIZE_PROMPT.format(items=items))
    # First non-empty line
    for line in response.splitlines():
        line = line.strip()
        if line and not line.lower().startswith("summary:"):
            return line
    return texts[0][:80]


def _cluster_embeddings(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> np.ndarray:
    """K-means cluster assignment. Returns array of shape (n,) with cluster indices.

    Uses scikit-learn's MiniBatchKMeans for speed on medium corpora. Falls back
    gracefully if k >= n (every point its own cluster).
    """
    n = embeddings.shape[0]
    if k >= n:
        return np.arange(n)
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        raise ImportError(
            "RAPTOR requires scikit-learn. Install: pip install scikit-learn"
        ) from e
    km = MiniBatchKMeans(
        n_clusters=k, random_state=seed, n_init=10, batch_size=256
    )
    return km.fit_predict(embeddings)


def build_raptor_tree(
    memory: Memory,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    branching_factor: int = 4,
    max_levels: int = 3,
    llm_call: LLMCallFn | None = None,
    link_derivations: bool = True,
    summarizer: Callable[[list[str]], str] | None = None,
) -> RaptorBuildResult:
    """Build a hierarchical summary tree over the user's atomic memories.

    Parameters
    ----------
    memory : Memory facade
    user_id, agent_id : scope to build the tree over
    branching_factor : how many children per cluster (i.e., cluster_count = n // bf)
    max_levels : cap the tree height
    llm_call : optional LLM for summarization. If None, a textual fallback is used.
    link_derivations : if True, also create DERIVED_FROM edges from summary -> children
    summarizer : optional explicit summarizer; overrides llm_call

    Returns
    -------
    RaptorBuildResult with per-level counts.

    Raises
    ------
    ValueError if branching_factor < 2 or max_levels < 1.
    """
    if branching_factor < 2:
        raise ValueError(
            f"branching_factor must be >= 2 (got {branching_factor}); "
            "a factor of 1 produces no clustering and 0 would divide by zero."
        )
    if max_levels < 1:
        raise ValueError(
            f"max_levels must be >= 1 (got {max_levels})."
        )
    # Start from existing atomic (non-summary) memories in scope
    all_records = memory.list_all(user_id=user_id, agent_id=agent_id)
    atomic = [r for r in all_records if r.operator != RAPTOR_OPERATOR]
    level_counts = [len(atomic)]
    summaries_created = 0

    current = atomic
    for level in range(1, max_levels + 1):
        if len(current) < branching_factor * 2:
            # Too few to cluster meaningfully
            break
        embeddings = np.stack([r.embedding for r in current])
        k = max(2, len(current) // branching_factor)
        cluster_ids = _cluster_embeddings(embeddings, k=k)

        new_summaries: list[MemoryRecord] = []
        for cid in range(k):
            members = [r for r, c in zip(current, cluster_ids, strict=True) if c == cid]
            if len(members) < 2:
                continue  # skip singletons
            if summarizer is not None:
                summary_text = summarizer([m.content for m in members])
            else:
                summary_text = _default_summarizer(
                    [m.content for m in members], llm_call
                )

            # Embedding of the summary = centroid of members (cheap, good enough)
            centroid = np.stack([m.embedding for m in members]).mean(axis=0)
            centroid = centroid.astype(np.float32)

            summary_rec = MemoryRecord(
                content=summary_text,
                embedding=centroid,
                user_id=user_id,
                agent_id=agent_id,
                parents=[m.id for m in members],
                operator=RAPTOR_OPERATOR,
                metadata={
                    "raptor_level": level,
                    "cluster_size": len(members),
                    "built_at": time.time(),
                },
            )
            memory.store.add(summary_rec)
            new_summaries.append(summary_rec)
            summaries_created += 1

            if link_derivations:
                # Edge: summary DERIVED_FROM each member
                for m in members:
                    memory.link(
                        summary_rec.id, m.id,
                        relation=DERIVED_FROM, weight=1.0 / len(members),
                        metadata={"raptor_level": level},
                    )

        level_counts.append(len(new_summaries))
        if not new_summaries:
            break
        current = new_summaries

    return RaptorBuildResult(
        user_id=user_id,
        agent_id=agent_id,
        levels=len(level_counts) - 1,
        level_counts=level_counts,
        summaries_created=summaries_created,
    )


def search_raptor(
    memory: Memory,
    query: str,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    level: int | None = None,
    limit: int = 10,
) -> list:
    """Search only at a specific RAPTOR level (0 = leaves / atomic).

    level=None returns results from ALL levels (default Memory.search does this
    already, but here we can filter to one level for targeted retrieval).
    """
    results = memory.search(
        query, user_id=user_id, agent_id=agent_id,
        limit=limit * 3,  # retrieve more, then filter
        filter_parents=False,  # we WANT parents in tree nav
    )
    if level is None:
        return results[:limit]
    if level == 0:
        filtered = [r for r in results if r.record.operator != RAPTOR_OPERATOR]
    else:
        filtered = [
            r for r in results
            if r.record.operator == RAPTOR_OPERATOR
            and r.record.metadata.get("raptor_level") == level
        ]
    return filtered[:limit]


__all__ = [
    "RAPTOR_OPERATOR",
    "RaptorBuildResult",
    "build_raptor_tree",
    "search_raptor",
]
