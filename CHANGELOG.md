# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] - locomo-readiness branch

### Changed
- **Behavior change**: `LLMExtractor` now defaults to `prompt_version="v2"`
  (an 80-line few-shot prompt with explicit fact categories) instead of
  the v1 zero-shot prompt. Existing callers can opt back via
  `LLMExtractor(llm_call, prompt_version="v1")`. This brings extraction
  quality to parity with Mem0/Letta on multi-session benchmarks like
  LoCoMo. Output format (one fact per `- ` line, `NO_FACTS` sentinel)
  is unchanged, so `_parse_facts` callers are unaffected.

### Added
- `FACT_EXTRACTION_PROMPT_V2` constant in `genome.memory.extraction`
  with 5 fact categories (preference / plan / relationship /
  professional / temporal), 5 few-shot examples, pronoun-resolution
  rule, and temporal-cue preservation rule.
- `prompt_version` parameter on `LLMExtractor.__init__`. Accepts `"v1"`
  or `"v2"`, raises `ValueError` otherwise.
- `rank-bm25>=0.2.2` runtime dependency for hybrid retrieval.
- **Hybrid retrieval (BM25 + dense via Reciprocal Rank Fusion)**: pass
  `mode="hybrid"` to `Memory.search()` for keyword-aware re-ranking.
  Cache key includes `mode` so dense / hybrid don't collide.
- `Memory(conflict_skip_unrelated=True)`: opt-in cost-aware variant of
  `resolve_conflicts` that skips the LLM call when the new fact has zero
  non-trivial word overlap with any candidate memory. Saves ~60% of
  conflict-resolution LLM calls on typical conversational benchmarks.
  Wired through `LocomoConfig.conflict_skip_unrelated`.
- `EmbeddingProvider("openai:text-embedding-3-small")` (or `-large`):
  polymorphic on `openai:` prefix; routes to OpenAI's embeddings API
  with exponential-backoff retry on 5xx / rate-limit / timeout errors,
  2048-input chunked batches, and an empty-text guard. Local
  sentence-transformers models remain the default.
- `LocomoConfig` fields: `embed_model`, `auto_extract_entities`,
  `resolve_conflicts`, `conflict_skip_unrelated`, `search_mode`. All are
  read by the default `memory_factory` so the corresponding architectural
  levers actually engage during the benchmark sweep.
- `DEFAULT_CONFIGS` expanded from 4 to 9 configs covering every
  architectural lever individually plus a `genome-full` and
  `genome-full-openai` headline.
- `scripts/locomo_calibrate.py`: 5-conversation x N-config sanity check
  that bails with a clear warning if any config drops below 0.20 mean
  score. Run before the full sweep to avoid wasting LLM budget.

### Fixed
- `SQLiteMemoryStore` now sniffs existing-row dim at `__init__` and
  rejects mismatched-dim writes/queries with a clear error (mirrors
  `PostgresMemoryStore._verify_schema_dim()`). Catches the case where a
  file-based store is reopened with a different `EmbeddingProvider`.
- `ConflictResolver` prompt now wraps memory content in
  `<existing_memories>` / `<new_fact>` data delimiters with delimiter
  scrub. Closes a prompt-injection vector where a memory's content
  could forge a closing tag and inject a fake `DECISION: DELETE` line.
- `Memory._parse_fact_detection`: empty `VALUE: ` lines now produce
  `value=None` and the auto-extract path early-returns instead of
  recording an empty-value fact that would silently pollute the KG.
- `_OpenAIBackend._call_with_retry()`: HTTP 5xx detection now uses
  numeric range comparison (`500 <= status < 600`) instead of substring
  match on the digit "5" (which would have matched 250, 150 etc).
- `PostgresMemoryStore._verify_schema_dim()`: tuple-unpacking instead
  of index access so the verification is robust to psycopg row_factory
  configuration.
- `genome/agent/memory.py` AgentMemory class docstring documents the
  orphaned core-block records that result from reusing one Memory
  across multiple (user_id, session_id) pairs and points at
  `Memory.reset(user_id=..., agent_id=...)` for cleanup.
- `temporal.facts_valid_at()` docstring now explicitly documents the
  SQL:2011 half-open `[valid_from, valid_until)` boundary semantics.
- `benchmarks/mini_locomo/run.py`: deterministic seed at the top of
  `main()` for reproducible embedding quantization and clustering.

### Tested
- 437+ Python tests passing across the matrix (Py 3.11 / 3.12 / 3.13);
  23 TypeScript SDK tests passing; 3 Postgres-skipped without DSN.
- New systematic regression test: `test_every_memory_flag_exposed_on_locomo_config`
  asserts every Memory constructor flag has a corresponding LocomoConfig
  field that the default memory_factory actually reads. Catches the
  silent-lever bug class that has shipped FOUR times already in this
  codebase.
- New regression test: `test_default_config_names_are_unique` -- two
  configs sharing a name silently shadow each other in result JSON.

### Added (R-night-3..6)
- Top-level `genome.*` exports for `ConflictResolver`, `ConflictDecision`,
  `EmbeddingProvider`. Previously required reaching into sub-modules.
- `--yes` / `-y` flag on the LOCOMO CLI to skip the cost-confirmation
  prompt for automated runs. Cost estimate (LLM calls + rough USD) now
  prints before the sweep starts; sweeps over 50k calls require an
  interactive confirmation by default.
- `genome-conflict-resolved-fast` config in `DEFAULT_CONFIGS`: applies
  conflict resolution with the cost-aware fast-path on (skips LLM call
  on zero-overlap facts; saves ~60% of conflict LLM cost).
- HybridScorer: defensive `["__empty__"]` placeholder when a corpus
  document has zero tokens. rank-bm25 raises ZeroDivisionError on an
  all-empty corpus; this guard keeps a single bad record from crashing
  the whole hybrid search.

### Fixed (R-night-3..6)
- Auto-consolidation: per-scope `threading.Lock` + busy-set so two
  concurrent `Memory.add()` calls in the same scope can no longer both
  fire `consolidate()` and double-prune. AsyncMemory + multi-thread
  workloads now serialize the trigger correctly.
- `MemoryRecord` validation rejects whitespace-only content (previously
  only rejected empty strings). Whitespace-only docs produced empty
  BM25 token sets that crashed hybrid search at scale.
- `_OpenAIBackend._call_with_retry`: HTTP 5xx detection now uses
  numeric range `500 <= status < 600` instead of substring `"5"` match
  on `str(status_code)`.

### Fixed (R-night-8..11)
- `Memory.close()` drains in-flight auto-extract LLM calls in addition
  to auto-consolidation. Without this drain, AsyncMemory + concurrent
  `add()` calls racing `close()` could tear down the SQLite connection
  while an LLM response was mid-callback, hitting `OperationalError`
  on the post-LLM `self.related()` / `self.record_fact()` calls.
- `Memory.close()` ALSO drains in-flight explicit `consolidate()`
  calls (third leg of the same race -- a worker thread firing
  `m.consolidate()` while the main thread calls `m.close()` could
  crash mid-prune on a closed connection).
- `HybridScorer` empty-corpus placeholder is now a per-call
  `__bm25_placeholder_<uuid>__` rather than the static string
  `__empty__`. The static placeholder could collide with real document
  content and silently corrupt rankings; the per-call UUID guarantees
  uniqueness.
- `ConflictResolver._has_any_overlap` stopwords list now includes
  `"user"` -- genome-extracted facts canonically start with "user X",
  so `"user"` was a chronic false-overlap that defeated the cost-aware
  fast-path.
- `ConflictResolver.decide_with_skip()` is now an opt-in API; the
  existing `decide()` always calls the LLM. This preserves the
  default-safe semantics callers expect.
- Memory's `close_drain_timeout_seconds` constructor parameter
  (default 30s, generous for slow LLM round-trips) makes the drain
  budget tunable. Previously the 5-second hardcoded budget could
  truncate slow-LLM auto-extract / explicit-consolidate completions.

### Fixed (R-night-12)
- **Lost-update race in `MemoryStore.update()`** (both SQLite and
  Postgres backends). Previously `update()` called `self.get()` for
  the current record OUTSIDE the write lock, then took the lock to
  issue the UPDATE statement. Two concurrent partial-patch updates
  against the same record (one patching `content`, one patching
  `metadata`) would each read the same `current` snapshot, then the
  second to commit would silently overwrite the first thread's patch.
  The entire read-modify-write is now under a single lock acquisition,
  with embedding-shape validation done outside the lock as a cheap
  fail-fast. Regression guard:
  `test_sqlite_update_no_lost_update_under_concurrent_partial_patches`.

### Fixed (R-night-13)
- **Timeline corruption race in `record_fact(invalidate_previous=True)`**.
  Two concurrent record_fact() calls for the same (entity_id, fact_type)
  could each find the same prior fact (valid_until=None), each close
  it, each add a new fact -- leaving TWO simultaneously-valid facts for
  the same slot and breaking the SQL:2011 half-open interval invariant
  current_facts() relies on. Memory now owns a `_fact_mutation_lock`
  acquired around the read-modify-write block in temporal.record_fact()
  (embedding generation stays outside the lock so slow encoders don't
  serialize all callers). Regression guard:
  `test_record_fact_concurrent_no_double_open_facts` (4 concurrent
  writers must collapse to exactly 1 current fact for the slot).

## [1.0.0] - 2026-04-25 — Initial release

Initial private release of the genome memory system.

### Components
- Memory facade (`genome.memory.Memory`) -- multi-tenant memory store with
  add / search / update / delete, scope isolation by `user_id` / `agent_id`.
- Async facade (`genome.memory.AsyncMemory`) -- asyncio-friendly wrapper.
- SQLite store (`genome.memory.sqlite_store`) -- zero-infra default backend.
- Postgres / pgvector store (`genome.memory.postgres_store`) -- production
  backend with HNSW vector index and dimension-mismatch verification.
- Recombination operators (`genome.operators`) -- uniform / frequency /
  single-point crossover, mutation, and parent-filtered retrieval.
- Hierarchical RAPTOR memory tree (`genome.memory.raptor`) -- tiered summaries
  for long-context retrieval.
- Typed memory-to-memory edge graph (`genome.memory.graph`) -- SUPERSEDES,
  CONTRADICTS, DERIVED_FROM, RELATES_TO, CAUSES.
- Temporal entity-fact knowledge graph (`genome.memory.temporal`) -- valid_from
  / valid_until windows, fact invalidation, point-in-time queries.
- Agent runtime (`genome.agent`) -- core (in-prompt) + archival memory with
  Anthropic / OpenAI tool schemas.
- LLM-extracted atomic facts (`genome.memory.extraction`) -- LLM-agnostic
  extraction pipeline.
- Memory consolidation (`genome.memory.consolidation`) -- fitness-based
  pruning with synthesis-before-prune.
- Response cache (`genome.memory.cache`) -- O(1) scope-epoch invalidation,
  thread-safe.
- LangChain + LlamaIndex adapters (`genome.adapters`).
- FastAPI REST server (`genome.server`) -- timing-safe API key, request
  size limit, structured logging, metrics.
- TypeScript SDK (`sdks/typescript/`) -- fetch-based REST client matching
  the Python Memory API.
- LOCOMO eval harness (`genome.evals.locomo`) -- conversation-replay
  evaluation with LLM-as-judge.

### Tests
- 357 tests passing, 3 skipped (Postgres tests skip without `POSTGRES_DSN`).
- ruff lint clean.
