"""Memory consolidation -- fitness-based pruning with optional synthesis.

Basic v0.3 version. The full GENOME v2 "evolution" mechanism (sleep cycles,
selection pressure, multi-generation) is out of scope here.

Fitness formula:
    fitness = log(access_count + 1) * exp(-age_days / half_life) + density_bonus

Where `density_bonus` rewards memories close to the centroid of the user's other
memories (clustered concepts survive; isolated outliers get pruned first).

When `synthesize_before_prune=True`, low-fitness memories that will be pruned
are first recombined in pairs to preserve their information in a hybrid. This
is the GENOME-specific bit -- we don't just forget, we compress forward.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from genome.memory.schema import MemoryRecord
from genome.memory.store import MemoryStore
from genome.observability import get_logger
from genome.synthesis import recombine

_log = get_logger("memory.consolidation")


@dataclass
class ConsolidationResult:
    user_id: str | None
    agent_id: str | None
    kept: int
    pruned: int
    synthesized: int  # hybrids created from low-fitness pairs before pruning
    before: int


def _fitness(
    rec: MemoryRecord,
    centroid: np.ndarray | None,
    now: float,
    half_life_days: float,
) -> float:
    age_days = max(0.0, (now - rec.created_at) / 86400.0)
    recency = math.exp(-age_days / half_life_days) if half_life_days > 0 else 1.0
    access = math.log(rec.access_count + 1)
    density = 0.0
    if centroid is not None:
        v = rec.embedding
        v_norm = np.linalg.norm(v) or 1.0
        c_norm = np.linalg.norm(centroid) or 1.0
        density = float((v @ centroid) / (v_norm * c_norm))
        density = max(0.0, density)  # clip negatives
    return access * recency + 0.1 * density


def score_memories(
    records: list[MemoryRecord],
    *,
    half_life_days: float = 30.0,
    now: float | None = None,
) -> list[tuple[MemoryRecord, float]]:
    """Return records paired with their fitness scores.

    Higher = more worth keeping. Synthesized memories get the same treatment as
    atomic ones (no special preservation yet).
    """
    if not records:
        return []
    now = now if now is not None else time.time()
    # Centroid for density bonus: mean of all embeddings in scope
    stacked = np.stack([r.embedding for r in records])
    centroid = stacked.mean(axis=0)
    return [(r, _fitness(r, centroid, now, half_life_days)) for r in records]


def consolidate(
    store: MemoryStore,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    max_memories: int = 500,
    half_life_days: float = 30.0,
    synthesize_before_prune: bool = False,
    synthesis_operator: str = "frequency_crossover",
) -> ConsolidationResult:
    """Prune memories in scope down to `max_memories` by fitness.

    If `synthesize_before_prune` is True, pairs of adjacent low-fitness memories
    are recombined into hybrids before the originals are deleted. This preserves
    the information in compressed form.
    """
    records = store.list_by_scope(user_id=user_id, agent_id=agent_id)
    before = len(records)
    if before <= max_memories:
        return ConsolidationResult(
            user_id=user_id,
            agent_id=agent_id,
            kept=before,
            pruned=0,
            synthesized=0,
            before=before,
        )

    scored = score_memories(records, half_life_days=half_life_days)
    scored.sort(key=lambda x: x[1], reverse=True)

    keepers = [r for r, _ in scored[:max_memories]]
    to_prune = [r for r, _ in scored[max_memories:]]

    synthesized_count = 0
    if synthesize_before_prune and len(to_prune) >= 2:
        # Pair up low-fitness memories and synthesize a hybrid from each pair
        for i in range(0, len(to_prune) - 1, 2):
            a = to_prune[i]
            b = to_prune[i + 1]
            try:
                hybrid_embedding = recombine(
                    [a.embedding, b.embedding], operator=synthesis_operator
                )
            except Exception as e:
                _log.warning(
                    "consolidation synthesis failed; skipping pair",
                    extra={
                        "a_id": a.id, "b_id": b.id,
                        "operator": synthesis_operator,
                        "error": repr(e),
                    },
                )
                continue
            hybrid_content = f"consolidated: {a.content[:60]} + {b.content[:60]}"
            hybrid = MemoryRecord(
                content=hybrid_content,
                embedding=hybrid_embedding,
                user_id=user_id,
                agent_id=agent_id,
                parents=[a.id, b.id],
                operator=synthesis_operator,
                metadata={"consolidation": True},
            )
            store.add(hybrid)
            synthesized_count += 1

    for r in to_prune:
        store.delete(r.id)

    return ConsolidationResult(
        user_id=user_id,
        agent_id=agent_id,
        kept=len(keepers),
        pruned=len(to_prune),
        synthesized=synthesized_count,
        before=before,
    )
