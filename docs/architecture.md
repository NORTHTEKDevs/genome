# Architecture

How genome is structured and why.

## System layers (top to bottom)

```
  +-------------------------------------------------+
  |  Framework adapters (LangChain, LlamaIndex)     |   soft deps
  +-------------------------------------------------+
  |  genome.server (FastAPI REST)                   |   soft deps
  +-------------------------------------------------+
  |  AsyncMemory      Memory                        |   public API
  +-----+-------------+-----------------------------+
        |             |
        v             v
  +---------+  +--------------+  +----------------+
  | cache   |  | extractor    |  | synthesizer    |
  +---------+  +--------------+  +----------------+
        |             |                  |
        v             v                  v
  +-------------------------------------------------+
  |  MemoryStore (abstract)                         |
  |    SQLiteMemoryStore   PostgresMemoryStore      |
  +-------------------------------------------------+
        |
        v
  +-------------------------------------------------+
  |  SQLite file / Postgres + pgvector              |
  +-------------------------------------------------+

  observability: structured logs + metrics registry
  errors: typed exceptions with hints
```

## Module map

| Module | Responsibility |
|---|---|
| `genome.memory.schema` | `MemoryRecord`, `SearchResult` dataclasses with input validation |
| `genome.memory.store` | `MemoryStore` abstract interface |
| `genome.memory.sqlite_store` | `SQLiteMemoryStore` reference implementation |
| `genome.memory.postgres_store` | `PostgresMemoryStore` production backend |
| `genome.memory.extraction` | `IdentityExtractor`, `LLMExtractor`, protocol |
| `genome.memory.graph` | `MemoryEdge`, relation constants |
| `genome.memory.entities` | GraphRAG-style entity extraction |
| `genome.memory.raptor` | RAPTOR hierarchical summarization |
| `genome.memory.cache` | `ResponseCache` LRU for search results |
| `genome.memory.consolidation` | Fitness-based pruning, synth-before-prune |
| `genome.memory.facade` | `Memory` -- public sync API |
| `genome.memory.async_facade` | `AsyncMemory` -- public async API |
| `genome.synthesis` | N-parent recombination operators |
| `genome.operators` | 2-parent recombination operators (v0.1 primitives) |
| `genome.embeddings` | `EmbeddingProvider` (sentence-transformers wrapper) |
| `genome.observability` | Structured logging + metrics registry |
| `genome.errors` | Typed exceptions with hints |
| `genome.adapters.langchain` | LangChain drop-in memory classes |
| `genome.adapters.llamaindex` | LlamaIndex drop-in memory classes |
| `genome.server.app` | FastAPI ASGI app factory |
| `genome.server.models` | Pydantic request/response schemas |
| `genome.memory_benchmark` | Head-to-head vs naive baseline |

## Data model

### Core: `MemoryRecord`

Every memory is a record with:

- `id` -- unique string (`mem_<12-char hex>`)
- `content` -- plain text (up to 100 KB)
- `embedding` -- 1-D float32 numpy array
- `user_id`, `agent_id` -- scoping (multi-tenant)
- `created_at`, `accessed_at`, `access_count` -- temporal + fitness signals
- `parents: list[str]` -- provenance for synthesized memories (empty for atomic)
- `operator: str | None` -- recombination operator used (e.g., `"uniform_crossover"`, `"raptor_summary"`, `"entity"`)
- `metadata: dict` -- extensible bag (role, source, raptor_level, entity_type, ...)

A record is atomic if `parents == []`, synthesized otherwise.

### Edges: `MemoryEdge`

```
MemoryEdge(
    from_id, to_id, relation,
    weight: float in [0, 1],
    created_at, metadata
)
```

Five common relation constants: `SUPERSEDES`, `CONTRADICTS`, `DERIVED_FROM`, `RELATES_TO`, `CAUSES`. Users can use arbitrary strings.

Edges cascade-delete when either endpoint is deleted.

### Special operator tags

- `operator=None` -- atomic fact
- `operator="simple_average"`, `"uniform_crossover"`, ... -- recombined hybrid (`parents` populated)
- `operator="raptor_summary"` -- RAPTOR cluster summary (`parents` = cluster members, `metadata["raptor_level"]` = tier)
- `operator="entity"` -- extracted entity (`metadata["entity_type"]` and `"entity_name"`)

## Key design decisions

### Why parent filtering is ON by default

When a hybrid of A and B is retrieved, A and B themselves score very high in cosine similarity. Without filtering, they saturate top-k and the actual hybrid gets pushed out. Our v0.2 benchmark showed this drops retrieval quality from 90% hit@3 to 28%.

Filtering: search auto-excludes any record `id` that appears in any other record's `parents` list in the same scope.

### Why the store is abstract

Real deployments need different backends: SQLite for dev, Postgres for prod, Qdrant for specialized workloads. `MemoryStore` is a thin ABC with ~10 methods; implementing a new backend is a few hundred lines.

### Why the cache is scope-fingerprinted

A per-query-string cache alone would return stale results after add/delete. Keying on `sha256(sorted(ids in scope))` auto-invalidates on mutation without explicit bookkeeping.

### Why LLMCallFn is sync OR async

Different LLM SDKs have different shapes. By accepting any `Callable[[str], str]` or `Callable[[str], Awaitable[str]]`, genome stays neutral. `AsyncMemory` auto-detects via `asyncio.iscoroutinefunction` and dispatches correctly.

### Why recombination is a first-class operation

Competitors (Mem0, Letta, Zep) merge memories via LLM summarization -- slow, expensive, non-deterministic. Recombination on embeddings is O(d) numpy ops, constant-time, deterministic-or-seeded. It enables:
- Compression during consolidation (`synthesize_before_prune`)
- Creative exploration (stochastic operators with different seeds)
- Agent self-reflection ("what's the common pattern across these?")

Whether synthesized hybrids carry useful semantic meaning is an empirical question. Our v0.2 benchmark validates it passes the design-doc criteria on 100 pairs across 3 encoders.

## Request flow: `Memory.add`

```
user calls m.add("I love coffee", user_id="alice")
    |
    v
FactExtractor.extract("I love coffee")          -- zero/one LLM call
    |  returns ["user likes coffee"]            -- one atomic fact
    v
EmbeddingProvider.encode_batch([fact])          -- one model inference
    |  returns np.array shape (1, 384)
    v
for fact, vec in zip(facts, vecs):
    MemoryRecord(content=fact, embedding=vec, user_id="alice")
    MemoryStore.add(record)                      -- one INSERT
cache.clear()                                    -- invalidate search cache
metrics: counter("memory.add.count", tags={"user_id": "alice"}).inc(1)
log: {"msg": "added memories", "user_id": "alice", "count": 1}
```

## Request flow: `Memory.search`

```
user calls m.search("drinks", user_id="alice", limit=5)
    |
    v
compute scope_fingerprint(store, user_id="alice")
    |
    v
cache.get(query, user_id, agent_id, limit, filter_parents, fingerprint)
    |--- hit -> return cached
    |--- miss -> continue
    v
EmbeddingProvider.encode("drinks")               -- one model inference
    |
    v
if filter_parents:
    collect all `parents` ids from records in scope
    exclude them from search
MemoryStore.search(q_vec, scope, limit, exclude_ids)
    |
    v
for each result: store.touch(id) -- update accessed_at / access_count
cache.put(...)
return results
```

## Async story

`AsyncMemory` delegates every write/read to `asyncio.to_thread(sync_method, ...)`. The underlying sqlite3 driver is synchronous, but SQLite is fast enough that off-loading to a thread pool doesn't bottleneck.

For Postgres we use `psycopg` in autocommit mode; same delegation pattern. A future `AsyncPostgresMemoryStore` can use `psycopg.AsyncConnection` directly for true async I/O.

`LLMCallFn`: if user passes `async def claude(prompt): ...`, `AsyncMemory` detects and bridges through a dedicated event loop inside the threadpool worker, so the full path stays non-blocking.

## Observability

- **Logs**: `genome` logger emits JSON to stderr (configurable). Every `add`/`search`/`synthesize` carries `user_id`, `agent_id`, and counts.
- **Metrics**: in-process counters + histograms in `genome.observability.get_metrics()`. Users attach a sink (lambda) to forward to OTel/Prometheus/Datadog.
- **Traces**: not built-in yet; v1.0 will add OTel span hooks.

## Security model

- All SQL uses parameter binding (no f-string interpolation of user input into queries).
- Input sizes validated in `MemoryRecord.__post_init__` (content 100KB, user_id 256 chars, metadata 100 keys).
- Framework adapters are soft-imported -- no transitive runtime deps force themselves on core users.
- REST server supports `X-API-Key` header authentication via `GENOME_API_KEY` env var.
- Multi-tenant isolation is enforced at the store layer -- no query ever crosses `user_id` unless the caller explicitly omits it.

Known limits:
- No row-level encryption at rest (use FDE on the host or Postgres TDE).
- No per-user rate limiting in the server (add via nginx/traefik middleware).
- No audit log (planned for v1.1).

## Extending genome

### Adding a new recombination operator

1. Write the function in `genome/operators.py` (2-parent) and `genome/synthesis.py` (N-parent).
2. Register in `OPERATORS` dict with default hyperparameters.
3. Register in `N_PARENT_OPERATORS` dict.
4. Add tests in `tests/test_operators_*.py` and `tests/test_synthesis.py`.
5. Benchmark: `python scripts/run_full_benchmark.py` -- ensure it passes Criteria 1-3.

### Adding a new storage backend

1. Subclass `genome.memory.store.MemoryStore`. Implement all abstract methods.
2. Cascade-delete edges in `delete()` for referential integrity.
3. Write integration tests (skip-if-infra-missing pattern -- see `test_postgres_store.py`).
4. Document connection string format in the class docstring.

### Adding a framework adapter

1. Create `genome/adapters/<framework>.py` with soft-imports for the framework.
2. Wrap `Memory` / `AsyncMemory` with the framework's interface shape.
3. Add tests gated on `pytest.importorskip(<framework>)`.
4. Update README's feature matrix.
