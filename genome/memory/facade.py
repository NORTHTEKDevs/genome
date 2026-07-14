"""The Memory facade -- user-facing API.

Combines EmbeddingProvider + MemoryStore + FactExtractor + synthesis + consolidation
into a single clean class that mirrors Mem0's API surface.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from genome.embeddings import EmbeddingProvider
from genome.errors import MemoryNotFoundError, ScopeError, SynthesisError
from genome.memory.cache import ResponseCache, ScopeEpochs
from genome.memory.extraction import (
    FactExtractor,
    IdentityExtractor,
    LLMCallFn,
    LLMExtractor,
)
from genome.memory.graph import MemoryEdge
from genome.memory.schema import MemoryRecord, SearchResult, _now
from genome.memory.sqlite_store import SQLiteMemoryStore
from genome.memory.store import MemoryStore
from genome.observability import get_error_capture, get_logger, get_metrics
from genome.synthesis import recombine as synthesize_embeddings


class Memory:
    """A persistent, searchable memory layer with recombination support.

    Quick start::

        from genome import Memory

        m = Memory()  # in-memory SQLite by default
        m.add("I love pour-over coffee", user_id="alice")
        results = m.search("what drinks does Alice like?", user_id="alice")
        for r in results:
            print(r.content, r.score)

    With an LLM for fact extraction::

        def claude(prompt: str) -> str:
            # your Anthropic client call here
            ...

        m = Memory(llm_call=claude)
        # Extracts "user likes pour-over coffee" and "user lives in Tokyo" as separate memories
        m.add("I love pour-over coffee and just moved to Tokyo", user_id="alice")

    With synthesis (the recombination differentiator)::

        ids = [r.id for r in m.search("...", user_id="alice", limit=3)]
        hybrid = m.synthesize(memory_ids=ids, user_id="alice",
                              operator="uniform_crossover")
    """

    def __init__(
        self,
        *,
        storage: str | Path | MemoryStore = ":memory:",
        embedding_provider: EmbeddingProvider | None = None,
        llm_call: LLMCallFn | None = None,
        extractor: FactExtractor | None = None,
        cache_size: int = 1024,
        enable_cache: bool = True,
        resolve_conflicts: bool = False,
        conflict_llm: LLMCallFn | None = None,
        conflict_topk: int = 3,
        conflict_skip_unrelated: bool = False,
        auto_extract_entities: bool = False,
        auto_fact_confidence_threshold: float = 0.7,
        auto_consolidate_threshold: int | None = None,
        auto_consolidate_target: int = 150,
        auto_consolidate_synthesize: bool = True,
        auto_consolidate_operator: str = "frequency_crossover",
        close_drain_timeout_seconds: float = 30.0,
        reranker: Any = None,
    ) -> None:
        # Store
        if isinstance(storage, MemoryStore):
            self.store = storage
        else:
            self.store = SQLiteMemoryStore(path=storage)

        # Embedding provider
        self.embed = embedding_provider or EmbeddingProvider()

        # Optional reranker (genome.memory.rerank.Reranker); reorders a wider
        # retrieval pool to the top-k. None = no reranking (default).
        self._reranker = reranker

        # Extractor priority: explicit > llm_call > identity
        if extractor is not None:
            self.extractor = extractor
        elif llm_call is not None:
            self.extractor = LLMExtractor(llm_call)
        else:
            self.extractor = IdentityExtractor()

        # Conflict resolution (opt-in). Requires an LLM. When True, every
        # extracted fact is checked against top-k existing memories in the
        # same scope and may ADD / UPDATE existing / DELETE existing / NONE.
        self._resolve_conflicts = resolve_conflicts
        self._conflict_topk = conflict_topk
        self._conflict_skip_unrelated = conflict_skip_unrelated
        self._conflict_llm: LLMCallFn | None = conflict_llm or llm_call
        if resolve_conflicts and self._conflict_llm is None:
            raise ValueError(
                "resolve_conflicts=True requires conflict_llm or llm_call"
            )

        # Auto entity + temporal-fact extraction on add() (opt-in). Requires
        # an LLM. When True, every successfully inserted memory triggers
        # entity extraction; for each discovered entity, a fact-detection
        # prompt produces (fact_type, value, confidence). High-confidence
        # detections become EntityFact records via record_fact().
        self._auto_extract_entities = auto_extract_entities
        self._auto_fact_threshold = auto_fact_confidence_threshold
        self._llm_for_auto: LLMCallFn | None = llm_call
        if auto_extract_entities and self._llm_for_auto is None:
            raise ValueError(
                "auto_extract_entities=True requires llm_call (or pass an LLM "
                "via llm_call)"
            )

        # Auto-consolidation trigger (opt-in). When the per-scope memory count
        # exceeds threshold, run consolidate() down to target, optionally
        # synthesizing hybrids of pruned pairs first. This is what makes
        # GENOME's recombination operators actually fire on a vanilla
        # add()+search() benchmark protocol -- without it, synthesize() and
        # consolidate() require explicit user calls.
        if auto_consolidate_threshold is not None and auto_consolidate_threshold <= 0:
            raise ValueError(
                f"auto_consolidate_threshold must be positive or None, "
                f"got {auto_consolidate_threshold}"
            )
        if auto_consolidate_target <= 0:
            raise ValueError(
                f"auto_consolidate_target must be positive, "
                f"got {auto_consolidate_target}"
            )
        if (
            auto_consolidate_threshold is not None
            and auto_consolidate_target >= auto_consolidate_threshold
        ):
            raise ValueError(
                f"auto_consolidate_target ({auto_consolidate_target}) must be "
                f"strictly less than threshold ({auto_consolidate_threshold}); "
                f"otherwise consolidation cannot make progress."
            )
        self._auto_consolidate_threshold = auto_consolidate_threshold
        # Per-scope busy flags + a lock so two concurrent add() calls in the
        # same scope can't both fire consolidate. Without this, AsyncMemory
        # workloads that drive two threads through to_thread can double-prune
        # on the count==threshold boundary. The same lock guards the
        # auto-extract counter so close() can drain BOTH paths atomically.
        import threading as _t
        self._auto_consolidate_lock = _t.Lock()
        self._auto_consolidate_busy: set[tuple[str | None, str | None]] = set()
        # Counter (not set) because multiple records in the same scope can
        # be auto-extracting concurrently; track in-flight count, not just
        # "is anybody running".
        self._auto_extract_inflight = 0
        # Same pattern for explicit Memory.consolidate() calls so close()
        # can drain those too. The auto- vs explicit distinction matters
        # only because auto is gated by per-scope busy flag, while explicit
        # can run multiple times concurrently across scopes.
        self._explicit_consolidate_inflight = 0
        # Lock for record_fact()'s invalidate-previous-then-add sequence.
        # Without it, two concurrent record_fact() calls for the same
        # (entity_id, fact_type) can both find the same `prior` fact
        # (valid_until=None), both close it, both add a new fact -- result
        # is TWO simultaneously-valid facts for the same slot, which breaks
        # current_facts() and the SQL:2011 half-open interval invariant.
        self._fact_mutation_lock = _t.Lock()
        # Serializes the entity check-existing-then-create sequence so two
        # concurrent same-scope adds extracting the same new entity don't both
        # miss it and create duplicate entity records.
        self._entity_mutation_lock = _t.Lock()
        # Drain budget for close(). Default 30s covers slow LLM calls
        # (auto-extract) on real APIs without hanging tests forever.
        self._close_drain_timeout_seconds = float(close_drain_timeout_seconds)
        self._auto_consolidate_target = auto_consolidate_target
        self._auto_consolidate_synthesize = auto_consolidate_synthesize
        self._auto_consolidate_operator = auto_consolidate_operator

        # Response cache with O(1) epoch-based invalidation
        self._cache: ResponseCache | None = (
            ResponseCache(capacity=cache_size) if enable_cache else None
        )
        self._scope_epochs = ScopeEpochs()

        self._log = get_logger("memory")
        self._metrics = get_metrics()
        self._error_capture = get_error_capture()

    # ---------- add / get / update / delete ----------

    def add(
        self,
        text: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict | None = None,
    ) -> list[MemoryRecord]:
        """Extract facts from `text` and store each as a separate memory.

        When `resolve_conflicts=True` was passed to the constructor, each
        extracted fact is first compared against the top-K existing memories
        in the same scope; the LLM decides ADD / UPDATE / DELETE / NONE and
        only ADD-decided facts produce a new INSERT. Returns the list of
        newly-stored records.
        """
        with self._metrics.histogram(
            "memory.add.duration", tags={"user_id": user_id or ""}
        ).time():
            facts = self.extractor.extract(text)
            if not facts:
                return []
            vecs = self.embed.encode_batch(facts)
            records: list[MemoryRecord] = []
            mutated_scope = False
            for fact, vec in zip(facts, vecs, strict=True):
                if self._resolve_conflicts:
                    decision, candidate_ids = self._resolve_one(
                        fact=fact, user_id=user_id, agent_id=agent_id
                    )
                    if decision.kind == "NONE":
                        continue
                    # SECURITY: target_id is parsed from LLM output. It MUST be
                    # one of the candidates we actually showed the resolver --
                    # which were scope-filtered by search() -- else a
                    # hallucinated or injected id could mutate/delete another
                    # memory (across tenants). Membership in candidate_ids IS
                    # the scope guarantee; never pass an unvetted id to the
                    # store's raw update/delete.
                    if (
                        decision.kind in ("UPDATE", "DELETE")
                        and decision.target_id not in candidate_ids
                    ):
                        self._log.warning(
                            "conflict-resolution target_id not in candidate "
                            "set; ignoring the %s and adding the fact instead",
                            decision.kind,
                            extra={
                                "target_id": decision.target_id,
                                "user_id": user_id or "",
                                "agent_id": agent_id or "",
                            },
                        )
                        # Fall through to ADD (safe: never touch an id the
                        # resolver was never shown).
                    elif decision.kind == "UPDATE" and decision.target_id:
                        updated = self.store.update(
                            decision.target_id,
                            content=fact,
                            embedding=vec,
                            metadata=None,
                        )
                        if updated is not None:
                            mutated_scope = True
                        continue
                    elif decision.kind == "DELETE" and decision.target_id:
                        if self.store.delete(decision.target_id):
                            mutated_scope = True
                        continue
                    # else: ADD -> fall through to insert
                # Per-record copy: caller-side mutation of the metadata dict
                # must not bleed into stored records, and multi-fact add()
                # must not have all records share one reference.
                rec = MemoryRecord(
                    content=fact,
                    embedding=vec,
                    user_id=user_id,
                    agent_id=agent_id,
                    metadata=dict(metadata) if metadata else {},
                )
                self.store.add(rec)
                records.append(rec)
                mutated_scope = True
                if self._auto_extract_entities:
                    self._auto_extract_for_record(rec)
            if mutated_scope:
                self._scope_epochs.bump(user_id, agent_id)
            self._maybe_auto_consolidate(user_id=user_id, agent_id=agent_id)
            self._metrics.counter(
                "memory.add.count", tags={"user_id": user_id or ""}
            ).inc(len(records))
            self._log.debug(
                "added memories",
                extra={"user_id": user_id, "agent_id": agent_id, "count": len(records)},
            )
        return records

    def _resolve_one(
        self,
        *,
        fact: str,
        user_id: str | None,
        agent_id: str | None,
    ):
        """Consult the conflict resolver for one fact.

        Returns (ConflictDecision, candidate_ids) where candidate_ids is the
        set of memory ids actually offered to the resolver. The caller MUST
        validate any UPDATE/DELETE target_id against this set before mutating
        -- the ids are scope-filtered here, so membership enforces tenant
        isolation on the LLM-supplied target.

        Searches dense top-k existing memories in scope, ignoring parent_filter
        and the response cache (we want the freshest view, including hybrids).
        """
        from genome.memory.conflict import ConflictResolver

        existing_results = self.search(
            fact,
            user_id=user_id,
            agent_id=agent_id,
            limit=self._conflict_topk,
            filter_parents=False,
            use_cache=False,
            mode="dense",
        )
        existing = [(r.id, r.content) for r in existing_results]
        candidate_ids = {r.id for r in existing_results}
        resolver = ConflictResolver(self._conflict_llm)
        if self._conflict_skip_unrelated:
            decision = resolver.decide_with_skip(new_fact=fact, existing=existing)
        else:
            decision = resolver.decide(new_fact=fact, existing=existing)
        return decision, candidate_ids

    AUTO_FACT_DETECTION_PROMPT = """\
You analyze a single atomic fact about a user and extract one structured
attribute if present.

Fact: {fact}

Output exactly:
FACT_TYPE: <one of: location, employer, occupation, relationship, preference, age, none>
VALUE: <the attribute value, e.g. "Tokyo" or "Google">
CONFIDENCE: <a float 0.0-1.0>

If no clean attribute can be extracted, output FACT_TYPE: none and CONFIDENCE: 0.0.
"""

    def _auto_extract_for_record(self, rec: MemoryRecord) -> None:
        """If auto_extract_entities is on, run entity extraction + fact detection.

        Increments _auto_extract_inflight for the lifetime of this call so
        Memory.close() can drain in-flight LLM calls before tearing down the
        store. Without this, AsyncMemory + close() can race and the
        post-LLM `self.related()` / `self.record_fact()` hits a closed store.
        """
        if self._llm_for_auto is None:
            return
        with self._auto_consolidate_lock:
            self._auto_extract_inflight += 1
        try:
            self._auto_extract_for_record_impl(rec)
        finally:
            with self._auto_consolidate_lock:
                self._auto_extract_inflight -= 1

    def _auto_extract_for_record_impl(self, rec: MemoryRecord) -> None:
        """Inner implementation -- the in-flight counter wraps this."""
        from genome.memory.entities import (
            LLMEntityExtractor,
            persist_entities_for_memory,
        )

        try:
            persist_entities_for_memory(
                self, rec, LLMEntityExtractor(self._llm_for_auto)
            )
        except Exception as e:  # noqa: BLE001 - log + continue
            self._log.debug(
                "auto-entity-extract failed; skipping",
                extra={"record_id": rec.id, "error": repr(e)},
            )
            return

        # Find entities just linked from this record via MENTIONS edges.
        # Relation constant lives in genome.memory.entities as "mentions".
        from genome.memory.entities import MENTIONS as _MENTIONS

        entities = self.related(
            rec.id,
            relation=_MENTIONS,
            direction="out",
            user_id=rec.user_id,
            agent_id=rec.agent_id,
        )
        if not entities:
            return

        prompt = self.AUTO_FACT_DETECTION_PROMPT.format(fact=rec.content)
        try:
            response = self._llm_for_auto(prompt)
        except Exception as e:  # noqa: BLE001 - log + continue
            self._log.debug(
                "auto-fact-detection LLM call failed; skipping",
                extra={"record_id": rec.id, "error": repr(e)},
            )
            return
        fact_type, value, confidence = self._parse_fact_detection(response)
        if not fact_type or fact_type == "none":
            return
        if not value:
            # Empty/missing VALUE -- LLM omitted the field. Don't record an
            # empty fact; that would silently pollute the temporal KG.
            return
        if confidence < self._auto_fact_threshold:
            return

        for entity in entities:
            try:
                self.record_fact(
                    entity.id,
                    fact_type=fact_type,
                    value=value,
                    source_memory_id=rec.id,
                    confidence=confidence,
                )
            except Exception as e:  # noqa: BLE001 - log + continue
                self._log.debug(
                    "auto-record-fact failed; skipping",
                    extra={"entity_id": entity.id, "error": repr(e)},
                )

    def _maybe_auto_consolidate(
        self, *, user_id: str | None, agent_id: str | None
    ) -> None:
        """Trigger consolidation if scope size exceeds threshold."""
        if self._auto_consolidate_threshold is None:
            return
        scope_key = (user_id, agent_id)
        # Serialize concurrent triggers per-scope: if a consolidation is
        # already running for this scope, skip. Two threads hitting count
        # == threshold simultaneously must NOT both call consolidate.
        with self._auto_consolidate_lock:
            if scope_key in self._auto_consolidate_busy:
                return
            scope_count = self.store.count(user_id=user_id, agent_id=agent_id)
            if scope_count <= self._auto_consolidate_threshold:
                return
            self._auto_consolidate_busy.add(scope_key)
        from genome.memory.consolidation import consolidate as _consolidate

        try:
            try:
                result = _consolidate(
                    self.store,
                    user_id=user_id,
                    agent_id=agent_id,
                    max_memories=self._auto_consolidate_target,
                    synthesize_before_prune=self._auto_consolidate_synthesize,
                    synthesis_operator=self._auto_consolidate_operator,
                )
            except Exception as e:  # noqa: BLE001 - log + continue
                self._log.warning(
                    "auto-consolidation failed; scope left at current size",
                    extra={
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "before": scope_count,
                        "error": repr(e),
                    },
                )
                return
            self._scope_epochs.bump(user_id, agent_id)
            self._log.info(
                "auto-consolidation fired",
                extra={
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "before": scope_count,
                    "kept": result.kept,
                    "pruned": result.pruned,
                    "synthesized": result.synthesized,
                },
            )
        finally:
            # Always release the busy flag, even on raise/return paths above.
            with self._auto_consolidate_lock:
                self._auto_consolidate_busy.discard(scope_key)

    @staticmethod
    def _parse_fact_detection(response: str) -> tuple[str | None, str | None, float]:
        ftype: str | None = None
        value: str | None = None
        conf = 0.0
        for line in response.splitlines():
            line = line.strip()
            if line.upper().startswith("FACT_TYPE:"):
                raw = line.split(":", 1)[1].strip().lower()
                ftype = raw or None
            elif line.upper().startswith("VALUE:"):
                # Reject empty values so we don't store ("location", "")
                # facts when the LLM omits the field.
                raw = line.split(":", 1)[1].strip()
                value = raw or None
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    conf = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return ftype, value, conf

    def get(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> MemoryRecord | None:
        """Get a single memory by id. Touches accessed_at (post-touch snapshot).

        If `user_id` and/or `agent_id` are given, the returned record must also
        match that scope -- returns None if the id exists under a different
        scope. Use this to prevent cross-tenant reads in a multi-tenant
        deployment.
        """
        probe = self.store.get(memory_id)
        if probe is None:
            return None
        if user_id is not None and probe.user_id != user_id:
            return None
        if agent_id is not None and probe.agent_id != agent_id:
            return None
        self.store.touch(memory_id)
        # Patch the in-memory copy to reflect the touch -- saves a round-trip
        # and closes the TOCTOU window between touch() and a second get().
        probe.access_count += 1
        probe.accessed_at = _now()
        return probe

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict | None = None,
        re_embed: bool = True,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> MemoryRecord | None:
        """Update a memory's content and/or metadata. If content changes and
        `re_embed` is True (default), the embedding is recomputed.

        If `user_id`/`agent_id` are given, the target must match that scope or
        the update is refused (returns None). Prevents cross-tenant writes.
        """
        if user_id is not None or agent_id is not None:
            existing = self.store.get(memory_id)
            if existing is None:
                return None
            if user_id is not None and existing.user_id != user_id:
                return None
            if agent_id is not None and existing.agent_id != agent_id:
                return None
        new_embedding: np.ndarray | None = None
        if content is not None and re_embed:
            new_embedding = self.embed.encode(content)
        result = self.store.update(
            memory_id,
            content=content,
            embedding=new_embedding,
            metadata=metadata,
        )
        if result is not None:
            self._scope_epochs.bump(result.user_id, result.agent_id)
        return result

    def delete(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> bool:
        """Delete a memory. Returns True if something was deleted.

        If `user_id`/`agent_id` are given, the target must match scope; returns
        False without deleting if the record is out-of-scope. Prevents
        cross-tenant deletion.
        """
        existing = self.store.get(memory_id)
        if existing is None:
            return False
        if user_id is not None and existing.user_id != user_id:
            return False
        if agent_id is not None and existing.agent_id != agent_id:
            return False
        result = self.store.delete(memory_id)
        if result:
            self._scope_epochs.bump(existing.user_id, existing.agent_id)
        return result

    # ---------- search ----------

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        filter_parents: bool = True,
        exclude_ids: set[str] | None = None,
        use_cache: bool = True,
        mode: str = "dense",
        reranker: Any = None,
        rerank_pool: int = 50,
    ) -> list[SearchResult]:
        """Cosine search for the top-k most similar memories within scope.

        If a `reranker` is provided (or one was set on the Memory), the base retrieval
        pulls a wider pool of `rerank_pool` candidates and the reranker reorders them to
        the top `limit` -- a large retrieval-quality gain (LoCoMo gold-evidence hit@10
        0.732 -> 0.792). See genome.memory.rerank.

        `mode` controls retrieval strategy:
            - "dense" (default): pure cosine similarity on embeddings (legacy)
            - "hybrid": BM25 + dense via Reciprocal Rank Fusion

        `filter_parents` (default True) removes any memory whose id is in another
        memory's `parents` list -- preventing parents from crowding out hybrids,
        per the v0.2 validation finding.

        `use_cache` (default True) enables the response cache. Identical queries
        against unchanged scopes return cached results without re-computing the
        embedding or re-scoring. The cache invalidates automatically on
        add/update/delete/reset. The cache key includes `mode` so dense and
        hybrid results for the same query do not collide.
        """
        if mode not in {"dense", "hybrid", "graph"}:
            raise ValueError(
                f"mode must be 'dense', 'hybrid' or 'graph', got {mode!r}"
            )
        if mode == "graph":
            return self._graph_search(
                query, user_id=user_id, agent_id=agent_id, limit=limit,
                filter_parents=filter_parents, exclude_ids=exclude_ids,
                use_cache=use_cache,
            )
        with self._metrics.histogram(
            "memory.search.duration", tags={"user_id": user_id or ""}
        ).time():
            _rr = reranker if reranker is not None else getattr(self, "_reranker", None)
            fetch = max(limit, rerank_pool) if _rr else limit
            cache_usable = (
                use_cache and self._cache is not None and exclude_ids is None
                and _rr is None
            )
            if cache_usable:
                epoch = self._scope_epochs.current(user_id, agent_id)
                cached = self._cache.get(
                    query, user_id, agent_id, limit, filter_parents, epoch,
                    mode=mode,
                )
                if cached is not None:
                    self._metrics.counter(
                        "memory.search.cache_hit", tags={"user_id": user_id or ""}
                    ).inc()
                    return cached

            q = self.embed.encode(query)
            exclude = set(exclude_ids or set())
            if filter_parents:
                in_scope = self.store.list_by_scope(user_id=user_id, agent_id=agent_id)
                parent_ids: set[str] = set()
                for rec in in_scope:
                    parent_ids.update(rec.parents)
                exclude |= parent_ids

            if mode == "dense":
                results = self.store.search(
                    q,
                    user_id=user_id,
                    agent_id=agent_id,
                    limit=fetch,
                    exclude_ids=exclude or None,
                )
            else:  # hybrid
                from genome.memory.hybrid import HybridScorer

                # Pull a wider dense window so RRF has room to re-rank
                dense_window = max(50, fetch * 5)
                dense_raw = self.store.search(
                    q,
                    user_id=user_id,
                    agent_id=agent_id,
                    limit=dense_window,
                    exclude_ids=exclude or None,
                )
                # Build BM25 corpus from same scope, excluding filtered ids
                in_scope_recs = self.store.list_by_scope(
                    user_id=user_id, agent_id=agent_id
                )
                corpus = {
                    r.id: r.content
                    for r in in_scope_recs
                    if r.id not in exclude
                }
                dense_pairs = [(r.id, r.score) for r in dense_raw]
                fused = HybridScorer().fuse(
                    query=query, dense_results=dense_pairs, corpus=corpus
                )
                # Re-hydrate SearchResult in fused order, top `limit`.
                # SearchResult is (record: MemoryRecord, score: float).
                by_id = {r.id: r for r in dense_raw}
                results = []
                for fid, fscore in fused[:fetch]:
                    if fid in by_id:
                        sr = by_id[fid]
                        sr.score = float(fscore)
                        results.append(sr)
                    else:
                        rec = self.store.get(fid)
                        if rec is not None:
                            results.append(SearchResult(record=rec, score=float(fscore)))

            if _rr is not None:                       # rerank the wider pool -> top `limit`
                results = _rr.rerank(query, results, top_k=limit)
            for r in results:
                self.store.touch(r.id)

            if cache_usable:
                self._cache.put(
                    query, user_id, agent_id, limit, filter_parents, epoch,
                    results, mode=mode,
                )
            self._metrics.counter(
                "memory.search.count", tags={"user_id": user_id or ""}
            ).inc()
        return results

    # ---------- synthesize (the differentiator) ----------

    def synthesize(
        self,
        memory_ids: list[str],
        *,
        operator: str = "uniform_crossover",
        user_id: str | None = None,
        agent_id: str | None = None,
        content: str | None = None,
        metadata: dict | None = None,
        **operator_kwargs: Any,
    ) -> MemoryRecord:
        """Create a new memory whose embedding is a recombination of the given
        parent memories.

        Parameters
        ----------
        memory_ids : list of existing memory ids (>=2)
        operator : name of a registered N-parent operator
            (see `genome.synthesis.N_PARENT_OPERATORS`)
        user_id, agent_id : scope for the new hybrid memory
        content : optional explicit content string. If None, defaults to a
            compact summary like "hybrid of <A> + <B>".
        metadata : optional extra metadata; provenance is added automatically.

        Raises
        ------
        ValueError if fewer than 2 ids given or any id is not found.
        """
        if len(memory_ids) < 2:
            raise SynthesisError(
                f"synthesize needs at least 2 parent memory ids, got {len(memory_ids)}",
                hint="Pass at least 2 ids from Memory.search() or .list_all().",
            )
        parents: list[MemoryRecord] = []
        for mid in memory_ids:
            p = self.store.get(mid)
            if p is None:
                raise MemoryNotFoundError(mid)
            parents.append(p)

        # Tenant isolation: if caller specifies a scope, every parent MUST
        # live in that same scope. This prevents an attacker from passing
        # another tenant's memory_ids to leak their content into a hybrid
        # created in the attacker's scope (cross-tenant data leak).
        if user_id is not None:
            mismatched = [p.id for p in parents if p.user_id != user_id]
            if mismatched:
                raise ScopeError(
                    f"parent memories {mismatched} not in user_id={user_id!r}",
                    hint=(
                        "All parents must belong to the same user_id as the "
                        "hybrid. Pass memory_ids obtained from Memory.search("
                        "user_id=...) to avoid cross-tenant leaks."
                    ),
                )
        if agent_id is not None:
            mismatched = [p.id for p in parents if p.agent_id != agent_id]
            if mismatched:
                raise ScopeError(
                    f"parent memories {mismatched} not in agent_id={agent_id!r}",
                    hint="All parents must match the specified agent_id.",
                )
        # If no scope given, enforce that all parents share the same scope --
        # refuse silent cross-scope synthesis.
        parent_scopes = {(p.user_id, p.agent_id) for p in parents}
        if len(parent_scopes) > 1:
            raise ScopeError(
                f"parent memories span multiple scopes: {parent_scopes}",
                hint=(
                    "All parents must share the same (user_id, agent_id). "
                    "Filter by scope before passing ids to synthesize()."
                ),
            )

        hybrid_embedding = synthesize_embeddings(
            [p.embedding for p in parents], operator=operator, **operator_kwargs
        )

        if content is None:
            short = [p.content[:40] for p in parents]
            content = f"hybrid of: {' + '.join(short)}"
        meta = dict(metadata or {})
        meta.setdefault("parent_contents", [p.content for p in parents])

        rec = MemoryRecord(
            content=content,
            embedding=hybrid_embedding,
            user_id=user_id,
            agent_id=agent_id,
            parents=[p.id for p in parents],
            operator=operator,
            metadata=meta,
        )
        self.store.add(rec)
        self._scope_epochs.bump(user_id, agent_id)
        return rec

    # ---------- graph relations (v0.4) ----------

    def link(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        *,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> MemoryEdge:
        """Create a typed directed edge between two memories.

        Endpoints must be in the same (user_id, agent_id) scope. Cross-scope
        links are refused to prevent tenant-boundary pollution (which would then
        leak via `related()`).
        """
        a = self.store.get(from_id)
        if a is None:
            raise MemoryNotFoundError(from_id)
        b = self.store.get(to_id)
        if b is None:
            raise MemoryNotFoundError(to_id)
        if a.user_id != b.user_id or a.agent_id != b.agent_id:
            raise ScopeError(
                f"link endpoints are in different scopes: "
                f"{from_id}=({a.user_id},{a.agent_id}) vs "
                f"{to_id}=({b.user_id},{b.agent_id})",
                hint=(
                    "Edges must connect memories in the same (user_id, "
                    "agent_id) scope. Cross-tenant links would leak memories "
                    "via related()."
                ),
            )
        edge = MemoryEdge(
            from_id=from_id,
            to_id=to_id,
            relation=relation,
            weight=weight,
            metadata=dict(metadata) if metadata else {},
        )
        stored = self.store.add_edge(edge)
        # Edges affect related() results; bump the endpoint scope so cached
        # graph-based queries invalidate.
        self._scope_epochs.bump(a.user_id, a.agent_id)
        return stored

    def unlink(
        self,
        edge_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> bool:
        """Delete an edge by id. Returns True if deleted.

        If `user_id`/`agent_id` are given, the edge's `from` endpoint must match
        that scope or the call is refused (returns False). Prevents cross-tenant
        edge deletion via guessed UUIDs.
        """
        edge = self.store.get_edge(edge_id)
        if edge is None:
            return False
        if user_id is not None or agent_id is not None:
            a = self.store.get(edge.from_id)
            if a is None:
                return False
            if user_id is not None and a.user_id != user_id:
                return False
            if agent_id is not None and a.agent_id != agent_id:
                return False
        result = self.store.delete_edge(edge_id)
        if result:
            a = self.store.get(edge.from_id)
            if a is not None:
                self._scope_epochs.bump(a.user_id, a.agent_id)
        return result

    def _graph_search(
        self,
        query: str,
        *,
        user_id: str | None,
        agent_id: str | None,
        limit: int,
        filter_parents: bool,
        exclude_ids: set[str] | None,
        use_cache: bool,
        seed_k: int | None = None,
        expand_per_entity: int = 12,
    ) -> list[SearchResult]:
        """Multi-hop graph retrieval.

        Dense-seed, then expand along the entity graph: for each seed memory,
        follow its MENTIONS edges to entities, and pull other memories that
        mention those same entities (co-mention siblings). This surfaces
        evidence scattered across non-adjacent turns that a single-shot cosine
        search misses -- the multi-hop aggregation case ("what do BOTH X and Y
        like?"). Candidates are re-ranked by cosine to the query plus a bonus
        proportional to how many seed-entities point at them (co-mention
        strength = a cheap stand-in for personalized-PageRank centrality).

        Falls back to the seed ranking when no entity graph exists in scope, so
        it is safe on data ingested without entity extraction.
        """
        from genome.memory.entities import (
            ENTITY_OPERATOR,
            MENTIONS,
            list_entities,
        )

        # Strong base: the best-performing retrieval (dense + parent filter, the
        # top config on LoCoMo). Returned UNCHANGED unless the question is
        # genuinely multi-hop -- so genome-graph is never-worse-than-baseline by
        # construction, and only diverges on the questions the graph targets.
        base = self.search(
            query, user_id=user_id, agent_id=agent_id, limit=limit,
            filter_parents=filter_parents, exclude_ids=exclude_ids,
            use_cache=use_cache, mode="dense",
        )
        if not base:
            return []

        # Query-anchor: which known entities are NAMED in the question? Multi-hop
        # questions reference the specific entities whose scattered facts must be
        # joined. Expanding along ONLY those (not along every entity in every
        # seed) is what keeps precision -- the v1 failure expanded blindly.
        qlow = query.casefold()
        named: list[MemoryRecord] = []
        for e in list_entities(self, user_id=user_id, agent_id=agent_id):
            name = str(e.metadata.get("entity_name", "")).strip().casefold()
            if len(name) >= 3 and name in qlow:
                named.append(e)
        # Gate: need >= 2 distinct named entities to call it multi-hop.
        if len(named) < 2:
            return base

        q = self.embed.encode(query)
        qn = float(np.linalg.norm(q)) or 1.0

        def cos(rec: MemoryRecord) -> float:
            e = rec.embedding
            en = float(np.linalg.norm(e)) or 1.0
            return float(e @ q) / (en * qn)

        base_ids = {r.record.id for r in base}
        # Relevance floor: candidates must be near the retrieval boundary, i.e.
        # at least (weakest kept hit - margin) -- admits near-miss evidence, not
        # deep off-topic noise. This is the guard the v1 design lacked.
        floor = min(cos(r.record) for r in base) - 0.05

        cand: dict[str, tuple[MemoryRecord, int]] = {}
        for e in named:
            for m in self.related(
                e.id, relation=MENTIONS, direction="in",
                user_id=user_id, agent_id=agent_id,
            )[:expand_per_entity]:
                if m.id in base_ids or m.operator == ENTITY_OPERATOR:
                    continue
                if cos(m) < floor:
                    continue  # relevance gate
                rec, n = cand.get(m.id, (m, 0))
                cand[m.id] = (rec, n + 1)  # n = # named entities co-mentioned
        if not cand:
            return base

        # Add a small number of the best query-anchored candidates, replacing
        # only the weakest base hits (fixed top-k, additive-in-spirit).
        m_slots = min(max(1, limit // 6), len(cand))
        cand_ranked = sorted(
            cand.values(), key=lambda t: (t[1], cos(t[0])), reverse=True,
        )[:m_slots]
        base_ranked = sorted(base, key=lambda r: cos(r.record), reverse=True)
        out: list[SearchResult] = [
            SearchResult(record=rec, score=cos(rec) + 0.02 * n)
            for rec, n in cand_ranked
        ]
        for r in base_ranked[: limit - len(out)]:
            out.append(SearchResult(record=r.record, score=cos(r.record)))
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:limit]

    def related(
        self,
        memory_id: str,
        relation: str | None = None,
        *,
        direction: str = "out",
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryRecord]:
        """Get memories linked from/to this one.

        direction='out' returns targets of outgoing edges (default).
        direction='in'  returns sources of incoming edges.
        direction='both' returns the union.

        If `user_id`/`agent_id` are given, only returns memories that match
        that scope. Given `link()` refuses cross-scope edges this is a defense
        in depth -- enforces scope on historical data from before the link()
        fix, and on records created via direct store manipulation.
        """
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"direction must be out/in/both, got {direction!r}")
        ids: set[str] = set()
        if direction in {"out", "both"}:
            for e in self.store.edges_from(memory_id, relation=relation):
                ids.add(e.to_id)
        if direction in {"in", "both"}:
            for e in self.store.edges_to(memory_id, relation=relation):
                ids.add(e.from_id)
        out: list[MemoryRecord] = []
        for i in ids:
            rec = self.store.get(i)
            if rec is None:
                continue
            if user_id is not None and rec.user_id != user_id:
                continue
            if agent_id is not None and rec.agent_id != agent_id:
                continue
            out.append(rec)
        return out

    def edges_of(
        self,
        memory_id: str,
        relation: str | None = None,
        direction: str = "out",
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryEdge]:
        """Return the edges themselves (not the linked memories). Useful for introspection.

        If `user_id`/`agent_id` are given, the anchor `memory_id` must match
        that scope or an empty list is returned. Prevents cross-tenant edge
        enumeration via guessed UUIDs.
        """
        if user_id is not None or agent_id is not None:
            anchor = self.store.get(memory_id)
            if anchor is None:
                return []
            if user_id is not None and anchor.user_id != user_id:
                return []
            if agent_id is not None and anchor.agent_id != agent_id:
                return []
        if direction == "out":
            return self.store.edges_from(memory_id, relation=relation)
        if direction == "in":
            return self.store.edges_to(memory_id, relation=relation)
        if direction == "both":
            return self.store.edges_from(memory_id, relation=relation) + \
                self.store.edges_to(memory_id, relation=relation)
        raise ValueError(f"direction must be out/in/both, got {direction!r}")

    # ---------- housekeeping ----------

    def count(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> int:
        """Count memories in the given scope."""
        return self.store.count(user_id=user_id, agent_id=agent_id)

    def list_all(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> list[MemoryRecord]:
        """List all memories in the given scope. No ordering guarantee."""
        return self.store.list_by_scope(user_id=user_id, agent_id=agent_id)

    def reset(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> int:
        """Delete all memories in scope. Returns the count deleted.

        WARNING: if both user_id and agent_id are None, deletes EVERYTHING.
        """
        records = self.store.list_by_scope(user_id=user_id, agent_id=agent_id)
        for r in records:
            self.store.delete(r.id)
        if records:
            self._scope_epochs.bump(user_id, agent_id)
            # If caller omitted both filters, this wiped everything --
            # invalidate all cached entries, not just this (None, None) epoch.
            if user_id is None and agent_id is None and self._cache is not None:
                self._cache.clear()
        return len(records)

    @property
    def cache_stats(self):
        """Return a frozen snapshot of ResponseCache stats, or None if disabled.

        Snapshot semantics: each call returns a fresh `CacheStats` copy so that
        a captured value reflects the cache state at the time of the call. The
        underlying counters keep mutating, but your local reference does not.
        """
        if self._cache is None:
            return None
        import dataclasses
        return dataclasses.replace(self._cache.stats)

    def clear_cache(self) -> None:
        """Manually clear the response cache."""
        if self._cache is not None:
            self._cache.clear()

    def consolidate(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        max_memories: int = 500,
        half_life_days: float = 30.0,
        synthesize_before_prune: bool = False,
        synthesis_operator: str = "frequency_crossover",
    ):
        """Prune to `max_memories` by fitness (access count + recency + density).

        If `synthesize_before_prune`, pairs of about-to-be-pruned memories are
        first recombined into hybrids so their information lives on in compressed
        form. This is genome's "sleep cycle" in miniature.

        Returns a ConsolidationResult with before/kept/pruned/synthesized counts.
        """
        # Track in-flight so close() can drain explicit consolidate calls
        # before tearing down the store. Without this, m.consolidate() in
        # one thread + m.close() in another races on the SQLite connection.
        with self._auto_consolidate_lock:
            self._explicit_consolidate_inflight += 1
        try:
            from genome.memory.consolidation import consolidate as _consolidate
            result = _consolidate(
                self.store,
                user_id=user_id,
                agent_id=agent_id,
                max_memories=max_memories,
                half_life_days=half_life_days,
                synthesize_before_prune=synthesize_before_prune,
                synthesis_operator=synthesis_operator,
            )
            # Invalidate cached search results for this scope: consolidation
            # prunes (and may synthesize) records, so a stale cache would
            # otherwise serve just-deleted memories and hide new hybrids. The
            # auto-consolidate path bumps too; the explicit API must match.
            self._scope_epochs.bump(user_id, agent_id)
            return result
        finally:
            with self._auto_consolidate_lock:
                self._explicit_consolidate_inflight -= 1

    # ---------- Entity graph (v0.5) ----------

    def extract_entities(
        self,
        memory_id: str,
        *,
        extractor: Any = None,
        llm_call: LLMCallFn | None = None,
    ):
        """Extract entities from a memory's content, persist them as entity
        records, and link via MENTIONS edges. Entity relations also link between
        entities.

        If `extractor` is provided, it's used directly. Else if `llm_call` is
        provided, an `LLMEntityExtractor` is built. Else a regex-only fallback.

        Returns an EntityPersistResult.
        """
        from genome.memory.entities import (
            LLMEntityExtractor,
            RegexEntityExtractor,
            persist_entities_for_memory,
        )
        rec = self.store.get(memory_id)
        if rec is None:
            raise MemoryNotFoundError(memory_id)
        if extractor is None:
            extractor = (
                LLMEntityExtractor(llm_call) if llm_call is not None
                else RegexEntityExtractor()
            )
        return persist_entities_for_memory(self, rec, extractor)

    def list_entities(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        entity_type: str | None = None,
    ) -> list[MemoryRecord]:
        """List entity records in scope, optionally filtered by type."""
        from genome.memory.entities import list_entities as _le
        return _le(
            self, user_id=user_id, agent_id=agent_id, entity_type=entity_type
        )

    def memories_mentioning(self, entity_id: str) -> list[MemoryRecord]:
        """Return memories that mention the given entity."""
        from genome.memory.entities import memories_mentioning as _mm
        return _mm(self, entity_id)

    # ---------- Temporal knowledge graph (v1.1) ----------

    def record_fact(
        self,
        entity_id: str,
        fact_type: str,
        value: str,
        *,
        valid_from: float | None = None,
        source_memory_id: str | None = None,
        confidence: float = 1.0,
        invalidate_previous: bool = True,
    ):
        """Record a new fact about an entity. Closes prior current fact of
        the same type (unless invalidate_previous=False).

        Returns an EntityFact. See `genome.memory.temporal` for details.
        """
        from genome.memory.temporal import record_fact as _rf
        return _rf(
            self, entity_id, fact_type, value,
            valid_from=valid_from,
            source_memory_id=source_memory_id,
            confidence=confidence,
            invalidate_previous=invalidate_previous,
        )

    def invalidate_fact(self, fact_id: str, *, at: float | None = None):
        """Mark a fact as no longer true (sets valid_until)."""
        from genome.memory.temporal import invalidate_fact as _if
        return _if(self, fact_id, at=at)

    def entity_timeline(
        self, entity_id: str, *, user_id: str | None = None,
    ):
        """Return all facts about an entity, newest first."""
        from genome.memory.temporal import entity_timeline as _et
        return _et(self, entity_id, user_id=user_id)

    def current_facts(
        self, entity_id: str, *, user_id: str | None = None,
    ):
        """Return currently-true facts about an entity."""
        from genome.memory.temporal import current_facts as _cf
        return _cf(self, entity_id, user_id=user_id)

    def facts_valid_at(
        self, entity_id: str, timestamp: float,
        *, user_id: str | None = None,
    ):
        """Return facts about an entity that were true at a specific timestamp."""
        from genome.memory.temporal import facts_valid_at as _fva
        return _fva(self, entity_id, timestamp, user_id=user_id)

    def merge_entity_facts(self, from_entity_id: str, to_entity_id: str) -> int:
        """Move all facts from one entity to another (dedup). Same-scope only."""
        from genome.memory.temporal import merge_entity_facts as _merge
        return _merge(self, from_entity_id, to_entity_id)

    # ---------- RAPTOR hierarchical summaries (v0.5) ----------

    def build_raptor_tree(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        branching_factor: int = 4,
        max_levels: int = 3,
        llm_call: LLMCallFn | None = None,
    ):
        """Build a RAPTOR summary tree over the user's memories.

        Clusters similar memories, summarizes each cluster into a higher-level
        memory, and repeats up to `max_levels`. Summaries are added as regular
        memories with `operator='raptor_summary'` and stored alongside atomics.

        If `llm_call` is provided, it's used for summarization. Otherwise the
        fallback is textual concatenation (still useful since the embedding is
        the cluster centroid).

        Returns a RaptorBuildResult with per-level counts.
        """
        from genome.memory.raptor import build_raptor_tree as _build
        return _build(
            self,
            user_id=user_id,
            agent_id=agent_id,
            branching_factor=branching_factor,
            max_levels=max_levels,
            llm_call=llm_call,
        )

    def search_at_level(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        level: int | None = None,
        limit: int = 10,
    ):
        """Search at a specific RAPTOR level (0 = atomic, 1+ = summaries)."""
        from genome.memory.raptor import search_raptor
        return search_raptor(
            self, query,
            user_id=user_id, agent_id=agent_id,
            level=level, limit=limit,
        )

    def close(self) -> None:
        """Release resources held by the store.

        Drains any in-flight auto-consolidation, in-flight auto-extract
        LLM calls, AND in-flight explicit consolidate() calls so the store
        isn't closed out from under them. Without this, AsyncMemory +
        concurrent add() racing close() can hit OperationalError
        mid-LLM-call (since auto-extract runs LLM then immediately calls
        self.related() / self.record_fact() against a store that may now
        be closed). Same race exists for any caller that fires
        consolidate() in a thread and then calls close() before it
        finishes. Bounded wait so close() never hangs; budget is tunable
        via close_drain_timeout_seconds (default 30s, generous for slow
        LLM round-trips).
        """
        import time as _time
        deadline = _time.monotonic() + self._close_drain_timeout_seconds
        while _time.monotonic() < deadline:
            with self._auto_consolidate_lock:
                if (
                    not self._auto_consolidate_busy
                    and self._auto_extract_inflight == 0
                    and self._explicit_consolidate_inflight == 0
                ):
                    break
            _time.sleep(0.05)
        self.store.close()

    # Context manager sugar
    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
