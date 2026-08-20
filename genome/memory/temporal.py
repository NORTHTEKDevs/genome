"""Temporal knowledge graph over entity facts.

Each entity (person, org, place, etc.) has a history of facts that were true
over specific time windows. Facts carry (valid_from, valid_until, fact_type,
value) and support:

- `record_fact`: assert a new fact about an entity as of a time
- `invalidate_fact`: mark a fact as no longer true (sets valid_until)
- `entity_timeline`: return all facts about an entity, newest first
- `facts_valid_at`: return facts that were true at a specific timestamp
- `current_facts`: return facts that are true right now (valid_until is None)

This is the Zep-parity feature. Built on top of the existing entity
infrastructure (entities are memories with `operator="entity"`), with facts
stored as memories tagged `operator="entity_fact"`.

Why store facts as memory records instead of a separate table? So they
inherit:
- scope isolation (user_id / agent_id)
- cascade delete (entity deleted -> facts deleted)
- embedding + search (query "where did Alice live?" finds relevant facts)
- parent-filtered retrieval (facts about an entity are parents of its timeline)
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from genome.errors import MemoryNotFoundError, ScopeError
from genome.memory.entities import ENTITY_OPERATOR
from genome.memory.schema import MemoryRecord

FACT_OPERATOR = "entity_fact"


@dataclass
class EntityFact:
    """A fact about an entity with a validity window."""
    id: str
    entity_id: str
    fact_type: str                  # e.g. "lives_in", "works_at", "age"
    value: str                      # e.g. "Tokyo", "OpenAI", "35"
    valid_from: float               # unix timestamp
    valid_until: float | None       # None = still currently true
    source_memory_id: str | None    # optional: which memory introduced this fact
    confidence: float               # [0, 1]
    metadata: dict[str, Any]
    believed_by: str | None = None  # attribution in a shared multi-agent store

    @property
    def is_current(self) -> bool:
        return self.valid_until is None

    @classmethod
    def from_record(cls, r: MemoryRecord) -> EntityFact:
        m = r.metadata
        return cls(
            id=r.id,
            entity_id=m["entity_id"],
            fact_type=m["fact_type"],
            value=m["value"],
            valid_from=m["valid_from"],
            valid_until=m.get("valid_until"),
            source_memory_id=m.get("source_memory_id"),
            confidence=m.get("confidence", 1.0),
            believed_by=m.get("believed_by"),
            metadata={k: v for k, v in m.items() if k not in {
                "entity_id", "fact_type", "value", "valid_from", "valid_until",
                "source_memory_id", "confidence", "believed_by",
            }},
        )


def _fact_trust(memory: Any, source_memory_id: str | None) -> int | None:
    """The trust of the memory that sourced a fact, or None when trust is off.

    Used to enforce origin-bound authority on the fact timeline: a fact sourced
    from a low-trust memory must not close one sourced from a high-trust memory.
    """
    policy = getattr(memory, "_trust_policy", None)
    if policy is None or not getattr(policy, "supersede_requires_geq", False):
        return None
    from genome.firewall import trust_of

    if source_memory_id is None:
        return policy.untagged_trust
    src = memory.store.get(source_memory_id)
    return trust_of(src, policy) if src is not None else policy.untagged_trust


def _refuse_close(memory: Any, incoming_trust: int | None, prior: MemoryRecord) -> bool:
    """True when the incoming fact is too low-trust to close ``prior``."""
    if incoming_trust is None:
        return False
    prior_trust = _fact_trust(memory, prior.metadata.get("source_memory_id"))
    return prior_trust is not None and incoming_trust < prior_trust


def _fact_records_for_entity(
    memory: Any, entity_id: str
) -> list[MemoryRecord]:
    """All fact records linked to an entity, ordered by valid_from descending."""
    ent = memory.store.get(entity_id)
    if ent is None:
        return []
    scope_records = memory.store.list_by_scope(
        user_id=ent.user_id, agent_id=ent.agent_id,
    )
    facts = [
        r for r in scope_records
        if r.operator == FACT_OPERATOR
        and r.metadata.get("entity_id") == entity_id
    ]
    facts.sort(key=lambda r: r.metadata.get("valid_from", 0), reverse=True)
    return facts


def record_fact(
    memory: Any,
    entity_id: str,
    fact_type: str,
    value: str,
    *,
    valid_from: float | None = None,
    source_memory_id: str | None = None,
    confidence: float = 1.0,
    invalidate_previous: bool = True,
    believed_by: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
) -> EntityFact:
    """Record a new fact about an entity as of `valid_from` (default: now).

    If `invalidate_previous=True` (default) and there's an existing current
    fact of the same `fact_type` for this entity, its `valid_until` is set to
    `valid_from` (so the timeline closes the old fact cleanly before opening
    the new one).

    `believed_by` attributes the fact to a believer (an agent id) in a shared
    store. Invalidation is believer-aware: a new fact only closes the previous
    current fact of the SAME believer (None matches None), so two agents that
    disagree hold their beliefs side by side instead of clobbering each other.
    Use `belief_conflicts` to surface the disagreements.
    """
    # Reject non-finite numbers at the boundary. A NaN confidence or valid_from is
    # accepted silently by SQLite but is not JSON-representable, so it detonates much
    # later - in an unrelated module, at report/receipt time - and permanently breaks
    # every downstream operation on that entity until someone finds the bad fact.
    if not math.isfinite(float(confidence)):
        raise ValueError(
            f"confidence must be a finite number, got {confidence!r}"
        )
    if valid_from is not None and not math.isfinite(float(valid_from)):
        raise ValueError(f"valid_from must be a finite epoch, got {valid_from!r}")

    entity = memory.store.get(entity_id)
    if entity is None:
        raise MemoryNotFoundError(entity_id)
    if entity.operator != ENTITY_OPERATOR:
        raise ValueError(
            f"record_fact expects an entity record, got "
            f"operator={entity.operator!r} for {entity_id}"
        )
    # Tenant ownership, mirroring link()'s cross-scope refusal: a caller that names
    # another tenant's entity_id must not be able to write facts onto their timeline.
    if user_id is not None and entity.user_id != user_id:
        raise ScopeError(
            f"entity {entity_id} is not in user_id={user_id!r}",
            hint=(
                "Pass the entity_id of an entity in your own scope. Facts are part "
                "of a tenant's record; writing across tenants is refused."
            ),
        )
    if agent_id is not None and entity.agent_id != agent_id:
        raise ScopeError(
            f"entity {entity_id} is not in agent_id={agent_id!r}",
            hint="Facts must be recorded within the entity's own scope.",
        )
    # A fact's provenance must belong to the same tenant as the fact.
    if source_memory_id is not None:
        src = memory.store.get(source_memory_id)
        if src is not None and (
            src.user_id != entity.user_id or src.agent_id != entity.agent_id
        ):
            raise ScopeError(
                f"source memory {source_memory_id} is in a different scope than "
                f"entity {entity_id}",
                hint=(
                    "A fact's source must be a memory in the same tenant; a "
                    "cross-tenant source would forge provenance."
                ),
            )

    now = valid_from if valid_from is not None else time.time()

    # Embed the fact's natural-language rendering so it's retrievable.
    # Done OUTSIDE the mutation lock because embeddings can be slow
    # (sentence-transformers, OpenAI API) and we don't want to serialize
    # all record_fact calls behind one another -- only the read-modify-
    # write needs to be atomic per (entity_id, fact_type) slot.
    rendered = f"{fact_type}: {value}"
    embedding = memory.embed.encode(rendered)

    # Mutation lock: prevent two concurrent record_fact() calls for the
    # same (entity_id, fact_type) from BOTH finding the same prior fact
    # (valid_until=None) and BOTH closing it then adding a new one --
    # which would leave two simultaneously-valid facts for the same slot
    # and corrupt the temporal timeline invariant.
    # Duck-type the lock so this works for any caller that doesn't
    # have a Memory facade (e.g. direct AgentMemory wrappers); the
    # uncoordinated path is documented as not safe for concurrent
    # record_fact in the same slot.
    fact_lock = getattr(memory, "_fact_mutation_lock", None)
    if fact_lock is not None:
        fact_lock.acquire()
    # If this fact is backdated ahead of an existing OPEN fact of the same
    # type (a future-dated fact valid_from > now), we must NOT leave two
    # open facts. Instead of closing the future fact, cap THIS fact's own
    # valid_until at the earliest such future valid_from, so the timeline
    # stays single-current: this fact covers [now, future_from), the future
    # fact remains open from future_from onward.
    incoming_trust = _fact_trust(memory, source_memory_id)
    new_valid_until: float | None = None
    try:
        # Close the existing current fact of the same type if requested
        if invalidate_previous:
            for prior in _fact_records_for_entity(memory, entity_id):
                if (
                    prior.metadata.get("fact_type") != fact_type
                    or prior.metadata.get("valid_until") is not None
                    # Believer-aware: never close another believer's fact.
                    or prior.metadata.get("believed_by") != believed_by
                ):
                    continue
                # Origin-bound authority: a lower-trust fact must not close a
                # higher-trust one. This is the temporal-layer counterpart of the
                # firewall's guarantee that web content cannot overwrite what the
                # user said - the two facts coexist as current instead.
                if _refuse_close(memory, incoming_trust, prior):
                    continue
                prior_from = prior.metadata.get("valid_from", 0)
                if prior_from <= now:
                    # Prior started at/before us: close it as we open.
                    new_meta = dict(prior.metadata)
                    new_meta["valid_until"] = now
                    memory.store.update(
                        prior.id, metadata=new_meta, embedding=None,
                    )
                else:
                    # Prior is a future-dated open fact: cap OUR window so we
                    # don't overlap it as a second current fact.
                    if new_valid_until is None or prior_from < new_valid_until:
                        new_valid_until = prior_from

        fact_meta: dict[str, Any] = {
            "entity_id": entity_id,
            "fact_type": fact_type,
            "value": value,
            "valid_from": now,
            "valid_until": new_valid_until,
            "source_memory_id": source_memory_id,
            "confidence": confidence,
        }
        if believed_by is not None:
            fact_meta["believed_by"] = believed_by
        fact_record = MemoryRecord(
            content=rendered,
            embedding=np.asarray(embedding, dtype=np.float32),
            user_id=entity.user_id,
            agent_id=entity.agent_id,
            operator=FACT_OPERATOR,
            metadata=fact_meta,
        )
        memory.store.add(fact_record)
    finally:
        if fact_lock is not None:
            fact_lock.release()

    # Bump scope epoch so caches invalidate (outside lock; cheap atomic ops)
    if hasattr(memory, "_scope_epochs"):
        memory._scope_epochs.bump(entity.user_id, entity.agent_id)

    return EntityFact.from_record(fact_record)


def invalidate_fact(
    memory: Any,
    fact_id: str,
    *,
    at: float | None = None,
) -> EntityFact | None:
    """Mark a fact as no longer true (sets valid_until). Returns the updated fact."""
    rec = memory.store.get(fact_id)
    if rec is None:
        return None
    if rec.operator != FACT_OPERATOR:
        raise ValueError(f"{fact_id} is not an entity_fact record")
    # Serialize the read-modify-write with record_fact so a concurrent
    # record_fact on the same slot can't lost-update this close (same lock
    # that protects the timeline invariant in record_fact).
    fact_lock = getattr(memory, "_fact_mutation_lock", None)
    if fact_lock is not None:
        fact_lock.acquire()
    try:
        new_meta = dict(rec.metadata)
        new_meta["valid_until"] = at if at is not None else time.time()
        memory.store.update(fact_id, metadata=new_meta, embedding=None)
    finally:
        if fact_lock is not None:
            fact_lock.release()
    if hasattr(memory, "_scope_epochs"):
        memory._scope_epochs.bump(rec.user_id, rec.agent_id)
    updated = memory.store.get(fact_id)
    return EntityFact.from_record(updated) if updated else None


def entity_timeline(
    memory: Any,
    entity_id: str,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
) -> list[EntityFact]:
    """Return all facts about an entity, ordered newest first.

    If `user_id`/`agent_id` are given, the entity must match that scope or an empty
    list returns. BOTH dimensions are checked: a deployment that tenants by
    agent_id gets the same isolation as one that tenants by user_id.
    """
    ent = memory.store.get(entity_id)
    if ent is None:
        return []
    if user_id is not None and ent.user_id != user_id:
        return []
    if agent_id is not None and ent.agent_id != agent_id:
        return []
    return [EntityFact.from_record(r) for r in _fact_records_for_entity(memory, entity_id)]


def current_facts(
    memory: Any,
    entity_id: str,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
) -> list[EntityFact]:
    """Return facts about an entity that are true *right now*.

    A fact is current iff `valid_from <= now < valid_until` (or valid_until is
    None). Filtering on `valid_until is None` alone is wrong: a future-dated
    open fact (valid_from in the future, valid_until None) would be reported as
    true now when it is not yet in effect.
    """
    return facts_valid_at(
        memory, entity_id, time.time(), user_id=user_id, agent_id=agent_id
    )


def facts_valid_at(
    memory: Any,
    entity_id: str,
    timestamp: float,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
) -> list[EntityFact]:
    """Return facts about an entity that were true at `timestamp`.

    Boundary semantics (SQL:2011 application-time period style):
        valid_from is INCLUSIVE  -- a fact becomes valid AT exactly valid_from
        valid_until is EXCLUSIVE -- a fact stops being valid AT exactly valid_until

    Concretely: a fact is valid at T iff `valid_from <= T < valid_until`
    (or `valid_until is None`, meaning currently true). Querying at exactly
    valid_until returns the SUCCESSOR fact, not this one. This matches the
    Zep / temporal-KG convention.
    """
    timeline = entity_timeline(
        memory, entity_id, user_id=user_id, agent_id=agent_id
    )
    return [
        f for f in timeline
        if f.valid_from <= timestamp
        and (f.valid_until is None or f.valid_until > timestamp)
    ]


def merge_entity_facts(
    memory: Any,
    from_entity_id: str,
    to_entity_id: str,
) -> int:
    """Move all facts from one entity to another (e.g. deduplication).

    Both entities must be in the same scope. Returns count of facts moved.
    Fails with ScopeError if scopes differ.
    """
    a = memory.store.get(from_entity_id)
    b = memory.store.get(to_entity_id)
    if a is None:
        raise MemoryNotFoundError(from_entity_id)
    if b is None:
        raise MemoryNotFoundError(to_entity_id)
    if a.user_id != b.user_id or a.agent_id != b.agent_id:
        raise ScopeError(
            f"cannot merge facts across scopes: "
            f"{from_entity_id}=({a.user_id},{a.agent_id}) vs "
            f"{to_entity_id}=({b.user_id},{b.agent_id})",
            hint="Entities in different user_id/agent_id scopes cannot be merged.",
        )
    # Serialize the fact re-parenting with concurrent record_fact/
    # invalidate_fact on the same slot to avoid lost updates.
    fact_lock = getattr(memory, "_fact_mutation_lock", None)
    if fact_lock is not None:
        fact_lock.acquire()
    try:
        moved = 0
        for fact in _fact_records_for_entity(memory, from_entity_id):
            new_meta = dict(fact.metadata)
            new_meta["entity_id"] = to_entity_id
            new_meta["merged_from"] = from_entity_id
            memory.store.update(fact.id, metadata=new_meta, embedding=None)
            moved += 1
    finally:
        if fact_lock is not None:
            fact_lock.release()
    if moved > 0 and hasattr(memory, "_scope_epochs"):
        memory._scope_epochs.bump(a.user_id, a.agent_id)
    return moved


__all__ = [
    "FACT_OPERATOR",
    "EntityFact",
    "record_fact",
    "invalidate_fact",
    "entity_timeline",
    "current_facts",
    "facts_valid_at",
    "merge_entity_facts",
]


# ---------------------------------------------------------------------------
# Multi-agent belief attribution
# ---------------------------------------------------------------------------


@dataclass
class BeliefPosition:
    """One believer's current stance in a conflict."""
    believed_by: str | None
    value: str
    valid_from: float
    source_memory_id: str | None
    fact_id: str


@dataclass
class BeliefConflict:
    """Believers currently disagreeing about one attribute of one entity."""
    entity_id: str
    fact_type: str
    positions: list[BeliefPosition]


def facts_believed_by(
    memory: Any,
    entity_id: str,
    believer: str | None,
    *,
    user_id: str | None = None,
) -> list[EntityFact]:
    """The timeline as one believer holds it, newest first.

    `believer=None` returns the unattributed (shared) timeline only -- an
    agent's attributed beliefs never leak into the shared view, and vice versa.
    """
    return [
        f for f in entity_timeline(memory, entity_id, user_id=user_id)
        if f.believed_by == believer
    ]


def belief_conflicts(
    memory: Any,
    entity_id: str,
    *,
    user_id: str | None = None,
) -> list[BeliefConflict]:
    """Surface attributes where believers currently hold different values.

    A conflict is >1 distinct current value for the same fact_type across
    believers (including the unattributed shared view). Agreement in different
    voices is not a conflict. Conflicts are surfaced for the caller to resolve
    deliberately -- silently picking a winner is how shared stores rot.
    """
    current = current_facts(memory, entity_id, user_id=user_id)
    by_type: dict[str, list[EntityFact]] = {}
    for fact in current:
        by_type.setdefault(fact.fact_type, []).append(fact)

    conflicts: list[BeliefConflict] = []
    for fact_type in sorted(by_type):
        facts = by_type[fact_type]
        if len({f.value for f in facts}) <= 1:
            continue
        conflicts.append(BeliefConflict(
            entity_id=entity_id,
            fact_type=fact_type,
            positions=[
                BeliefPosition(
                    believed_by=f.believed_by,
                    value=f.value,
                    valid_from=f.valid_from,
                    source_memory_id=f.source_memory_id,
                    fact_id=f.id,
                )
                for f in sorted(facts, key=lambda f: (f.believed_by or "", f.id))
            ],
        ))
    return conflicts
