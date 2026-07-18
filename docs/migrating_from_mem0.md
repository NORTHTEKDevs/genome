# Migrating from Mem0

GENOME mirrors Mem0's core verbs (`add` / `search` / `get` / `update` / `delete` /
`reset`), so most code moves over with small edits. This page is honest about what's
the same and what differs — so the swap doesn't surprise you on first contact.

## The one-line idea

```python
# Mem0
from mem0 import Memory
m = Memory()
m.add("I love pour-over coffee", user_id="alice")
hits = m.search("what drinks does the user like?", user_id="alice")

# GENOME
from genome import Memory
m = Memory(storage="memories.db")          # local embedder by default, no API key
m.add("I love pour-over coffee", user_id="alice")
hits = m.search("what drinks does the user like?", user_id="alice", limit=5)
```

## Signature deltas (what actually changes)

| Operation | Mem0 | GENOME |
|---|---|---|
| Construct | `Memory()` / `Memory.from_config({...})` | `Memory(storage=..., embedding_provider=..., ...)` — plain kwargs, no config dict |
| Add | `add(messages, user_id=...)` — `messages` is usually a list of `{"role","content"}` dicts | `add(text: str, user_id=..., agent_id=..., metadata=...)` — a **plain string**; returns the created record(s) |
| Search | `search(query, user_id=..., limit=...)` → `{"results": [...]}` | `search(query, user_id=..., limit=...)` → **a list** of hits with `.score`, `.content`, `.record` |
| Get / update / delete | `get(id)` / `update(id, data=...)` / `delete(id)` | `get(id, user_id=...)` / `update(id, content=..., metadata=...)` / `delete(id, user_id=...)` — pass the scope to enforce tenant isolation |
| Clear a user | `delete_all(user_id=...)` | `reset(user_id=...)` |
| Extraction LLM | required on every `add()` (that's the cost) | **optional** — omit `llm_call` and `add()` stores the raw message locally with zero LLM calls; pass `llm_call=...` only if you want atomic-fact extraction |

### The two edits that catch people

1. **`add()` takes a string, not a messages list.** If you were passing
   `[{"role": "user", "content": "..."}]`, pass the `content` string (join a turn's
   text yourself if you're storing a whole exchange).
2. **`search()` returns a list, not a dict.** Replace `res["results"]` with the list
   directly, and read `hit.content` / `hit.score` instead of `hit["memory"]`.

## What GENOME adds on top (no Mem0 equivalent)

- `synthesize(...)` — recombine memories' embeddings into a new hybrid memory.
- `link(...)` / `related(...)` — typed graph relations (`SUPERSEDES`, `CONTRADICTS`, ...).
- `build_raptor_tree(...)` / `search_at_level(...)` — hierarchical summaries.
- Bi-temporal belief state — point-in-time (`as-of`) answering + `explain_belief` audit.
- `AsyncMemory` — the same API, `await`-able.

## What GENOME does NOT do (yet)

- No hosted cloud tier — GENOME is local-first (that's the air-gapped/audit moat).
- No Mem0-style `from_config({...})` dict; configuration is explicit constructor kwargs.
- Graph memory is typed-relation + RAPTOR + entity extraction, not Mem0's graph-store config.

If a migration snag isn't covered here, please open a
[Discussion](https://github.com/NORTHTEKDevs/genome/discussions) — that feedback shapes
this page.
