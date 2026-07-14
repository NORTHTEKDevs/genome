# API Reference

Complete reference for the public genome API.

## `genome.Memory`

The primary synchronous memory layer.

### Constructor

```python
Memory(
    *,
    storage: str | Path | MemoryStore = ":memory:",
    embedding_provider: EmbeddingProvider | None = None,
    llm_call: LLMCallFn | None = None,
    extractor: FactExtractor | None = None,
    cache_size: int = 1024,
    enable_cache: bool = True,
)
```

- `storage`: SQLite file path (`"memories.db"`), `":memory:"` for ephemeral, or a pre-built `MemoryStore` instance for Postgres / custom backends.
- `embedding_provider`: defaults to `EmbeddingProvider()` (loads `all-MiniLM-L6-v2`).
- `llm_call`: optional sync callable `(prompt: str) -> str` for auto fact-extraction. If provided, wraps into `LLMExtractor`.
- `extractor`: explicit `FactExtractor` overriding both default and `llm_call`.
- `cache_size`: response-cache LRU capacity.
- `enable_cache`: if False, no cache at all.

### Core methods

| Method | Description |
|---|---|
| `add(text, *, user_id, agent_id, metadata) -> list[MemoryRecord]` | Extract + store one or more atomic memories |
| `get(memory_id) -> MemoryRecord \| None` | Fetch by id; touches accessed_at |
| `update(memory_id, *, content, metadata, re_embed) -> MemoryRecord \| None` | Modify in place (re-embeds if content changes) |
| `delete(memory_id) -> bool` | Delete + cascade edges. True if deleted |
| `search(query, *, user_id, agent_id, limit, filter_parents, exclude_ids, use_cache) -> list[SearchResult]` | Cosine search; parents filtered by default |
| `count(*, user_id, agent_id) -> int` | Count within scope |
| `list_all(*, user_id, agent_id) -> list[MemoryRecord]` | Dump all in scope |
| `reset(*, user_id, agent_id) -> int` | Delete all in scope; returns count |

### Synthesis

| Method | Description |
|---|---|
| `synthesize(memory_ids, *, operator, user_id, agent_id, content, metadata, **operator_kwargs) -> MemoryRecord` | Recombine 2+ parents into a hybrid |

Operators available: `simple_average`, `weighted_sum`, `uniform_crossover`, `frequency_crossover`, `attention_weighted_crossover`, `uniform_crossover_with_mutation`, `single_point_crossover`, `multi_point_crossover`, `concat_project` (baseline).

### Graph

| Method | Description |
|---|---|
| `link(from_id, to_id, relation, *, weight, metadata) -> MemoryEdge` | Create typed directed edge |
| `unlink(edge_id) -> bool` | Delete edge |
| `related(memory_id, relation, *, direction) -> list[MemoryRecord]` | Fetch linked memories (`direction` in `"out"`, `"in"`, `"both"`) |
| `edges_of(memory_id, relation, direction) -> list[MemoryEdge]` | Fetch raw edges |

Relation constants: `SUPERSEDES`, `CONTRADICTS`, `DERIVED_FROM`, `RELATES_TO`, `CAUSES` (any free string also works).

### Consolidation

| Method | Description |
|---|---|
| `consolidate(*, user_id, agent_id, max_memories, half_life_days, synthesize_before_prune, synthesis_operator) -> ConsolidationResult` | LRU + fitness prune |

Returns `ConsolidationResult(user_id, agent_id, kept, pruned, synthesized, before)`.

### RAPTOR

| Method | Description |
|---|---|
| `build_raptor_tree(*, user_id, agent_id, branching_factor, max_levels, llm_call) -> RaptorBuildResult` | Cluster + summarize recursively |
| `search_at_level(query, *, user_id, agent_id, level, limit) -> list[SearchResult]` | Search at a specific tree level (0 = atomic) |

### Entity graph

| Method | Description |
|---|---|
| `extract_entities(memory_id, *, extractor, llm_call) -> EntityPersistResult` | Extract entities from a memory's content; create MENTIONS edges |
| `list_entities(*, user_id, agent_id, entity_type) -> list[MemoryRecord]` | List entity records |
| `memories_mentioning(entity_id) -> list[MemoryRecord]` | Inverse of MENTIONS |

### Introspection

| Method | Description |
|---|---|
| `cache_stats` (property) | `CacheStats(hits, misses, size, hit_rate)` or None if cache disabled |
| `clear_cache()` | Manually clear response cache |
| `close()` | Release DB connections; safe to call multiple times |

### Context manager

```python
with Memory(storage="m.db") as m:
    m.add("hello", user_id="alice")
# auto-closes on exit
```

---

## `genome.AsyncMemory`

Same API as `Memory` but every method is a coroutine. Accepts sync OR async `llm_call`. Concurrency-safe via `asyncio.to_thread`.

```python
async with AsyncMemory(storage="m.db") as m:
    await m.add(...)
    await m.search(...)
```

---

## `MemoryRecord`

```python
@dataclass
class MemoryRecord:
    content: str                       # up to 100 KB
    embedding: np.ndarray               # 1-D float32
    id: str                             # "mem_<hex>"
    user_id: str | None                 # up to 256 chars
    agent_id: str | None
    created_at: float                   # unix timestamp
    accessed_at: float
    access_count: int
    parents: list[str]                  # provenance ids
    operator: str | None                # recombination op name
    metadata: dict[str, Any]            # up to 100 keys
```

Helpers: `.is_synthesized` (bool), `.age_seconds` (float).

## `SearchResult`

```python
@dataclass
class SearchResult:
    record: MemoryRecord
    score: float            # cosine similarity [-1, 1]

    # shortcuts:
    content -> record.content
    id      -> record.id
```

## `MemoryEdge`

```python
@dataclass
class MemoryEdge:
    from_id: str
    to_id: str
    relation: str
    id: str                             # "edge_<hex>"
    weight: float                       # in [0, 1]
    created_at: float
    metadata: dict[str, Any]
```

---

## `genome.memory.store.MemoryStore` (ABC)

Implement all abstract methods to ship a new backend. See `SQLiteMemoryStore` and `PostgresMemoryStore` for references.

Methods: `add`, `get`, `update`, `delete`, `search`, `list_by_scope`, `count`, `touch`, `close`, `add_edge`, `get_edge`, `delete_edge`, `edges_from`, `edges_to`, `delete_edges_touching`.

---

## `genome.server.app.create_app`

Builds a FastAPI ASGI app. Call with a pre-built `Memory` for tests; without args to build from env vars.

Environment variables:
- `GENOME_STORAGE`: SQLite path or `postgresql://` DSN
- `GENOME_EMBED_MODEL`: sentence-transformers model name
- `GENOME_CACHE_SIZE`: response cache LRU capacity
- `GENOME_API_KEY`: if set, required in `X-API-Key` header

REST endpoints:
- `GET /health` -- readiness + cache stats
- `POST /v1/memories` -- add
- `GET /v1/memories/{id}` -- get
- `PATCH /v1/memories/{id}` -- update
- `DELETE /v1/memories/{id}` -- delete
- `POST /v1/search` -- cosine search
- `POST /v1/synthesize` -- recombine
- `POST /v1/edges` -- link
- `DELETE /v1/edges/{id}` -- unlink
- `GET /v1/memories/{id}/related` -- fetch linked memories
- `DELETE /v1/scope` -- reset scope (`?user_id=&agent_id=`)
- `GET /v1/count` -- count in scope

OpenAPI spec at `/docs`.

---

## Errors

All errors inherit from `genome.errors.GenomeError` which in turn inherits from `ValueError` for backward compatibility.

| Class | When raised |
|---|---|
| `MemoryNotFoundError(memory_id)` | Any operation on a missing memory |
| `SynthesisError` | `synthesize()` with <2 parents or bad op |
| `OperatorError` | Recombination operator failure |
| `InvalidEmbeddingError` | Wrong shape/dtype embedding |
| `ScopeError` | Cross-scope op without explicit opt-in |
| `ConfigError` | Bad DSN, missing env, etc |
| `CorruptedStoreError` | Store invariants violated |

Every error carries `.hint` with a specific next step.

---

## Observability

```python
from genome.observability import configure_logging, get_metrics

configure_logging(level="INFO", json_output=True)
metrics = get_metrics()
metrics.set_sink(lambda name, value, tags: forward_to_otel(...))

# Snapshot for /metrics endpoint
snap = metrics.snapshot()   # {"counters": {...}, "histograms": {...}}
```

Memory emits:
- `memory.add.count` (counter) + `memory.add.duration` (histogram)
- `memory.search.count` + `memory.search.duration` + `memory.search.cache_hit`

All tagged with `user_id`.
