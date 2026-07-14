# Troubleshooting

Common problems and their fixes.

## Installation

### `torch` wheel fails to install on Python 3.14

sentence-transformers depends on torch, which doesn't yet publish wheels for Python 3.14. **Fix:** use Python 3.11, 3.12, or 3.13.

```bash
pyenv install 3.12
pyenv local 3.12
python -m venv .venv
source .venv/bin/activate
pip install genome
```

### `ImportError: genome[postgres] requires psycopg`

You imported `PostgresMemoryStore` without installing the extra.
**Fix:** `pip install "genome[postgres]"`.

### `ImportError: genome.server requires fastapi`

**Fix:** `pip install "genome[fastapi]"` or `pip install "genome[all]"`.

## First-run

### Embedding model download hangs or fails

The default provider downloads `sentence-transformers/all-MiniLM-L6-v2` (~80 MB) on first use.
- Confirm you have network access.
- On air-gapped systems, pre-download with `huggingface-cli download sentence-transformers/all-MiniLM-L6-v2`.
- Set `HF_HOME=/path/to/cache` if you want a custom cache location.

### First `m.add()` takes 30-60 seconds

Normal. The model is being loaded into memory. Subsequent calls are fast.

### Slow on Windows + no Developer Mode

You'll see:
> `huggingface_hub` cache-system uses symlinks by default ... your machine does not support them ...

This is a warning, not an error. Caching still works (just uses more disk). Enable Developer Mode in Windows Settings if you want symlinks.

## Retrieval

### My search returns parents of synthesized memories instead of the hybrids

This is caused by *not* using parent filtering (default is ON).
**Fix:** use `m.search(..., filter_parents=True)` or omit the argument. If you explicitly want the parents, pass `filter_parents=False`.

### Retrieval quality seems bad

- **Check the embedding model**: `all-MiniLM-L6-v2` is fast but not SOTA. Try `all-mpnet-base-v2` (slower, better).
- **Check your query**: embedding-space queries work best with full sentences, not keywords.
- **Check for scope mismatch**: is the `user_id` you're querying the same one you stored under?
- **Run the benchmark**: `python -m genome.memory_benchmark` shows expected metrics against the baseline.

### Search returns the same item repeatedly across queries (cache too eager)

Typically not a cache issue -- the cache invalidates on add/update/delete. If you're sure the cache is wrong:
```python
m.clear_cache()
```
Or disable: `Memory(enable_cache=False)`.

## Performance

### SQLite write throughput is low at high concurrency

SQLite serializes writes. For write-heavy workloads with many concurrent writers, switch to Postgres:
```python
from genome.memory.postgres_store import PostgresMemoryStore
store = PostgresMemoryStore(dsn="postgresql://...", embedding_dim=384)
m = Memory(storage=store)
```

### Search is O(n) over the user's memories

The SQLite backend does naive cosine scan -- fine up to ~100k memories per user. Beyond that, use Postgres + pgvector (HNSW index, sub-second on millions).

### Cache hit rate is low

The cache is scope-fingerprinted: it invalidates on every add/update/delete. If your agent mutates memory between every search, the cache won't help. Consider batching adds.

## Data integrity

### "memory not found" after deleting a parent

Expected behavior. If you `delete()` a memory that's a parent of a synthesized hybrid, edges are cleaned up but the hybrid's `parents` list still references the now-gone id. The hybrid itself is valid (its embedding is still useful); only explicit references to the dead parent fail.

### My edges disappeared

Edge cascade-deletes when *either* endpoint is deleted. This is intentional for referential integrity. If you need to preserve an edge, delete it explicitly or re-link to a new target.

## Postgres-specific

### "extension vector does not exist"

Enable pgvector in your Postgres:
```sql
CREATE EXTENSION vector;
```
Or use the `pgvector/pgvector:pg16` Docker image which ships it enabled.

### "dimension mismatch: got N expected M"

The pgvector column type is fixed at `vector(N)` where `N = embedding_dim`. If you change your embedding model, you must either:
- Drop and recreate the table.
- Start with a fresh schema/database.

Use `PostgresMemoryStore(dsn=..., embedding_dim=<new_dim>)` and make sure the embedding provider matches.

### Connection pooling

`PostgresMemoryStore` opens a single connection. For high-concurrency servers, wrap it in your own pool (psycopg's built-in pool or pgbouncer). Future versions will ship a pooled store.

## REST API

### `/health` returns 200 but no data stored

The health check only confirms the store responds. Ensure you're POSTing to `/v1/memories`, not `/memories`. The `v1` prefix is required.

### `401 Unauthorized` in browser

`GENOME_API_KEY` is set. Include `X-API-Key: <value>` in every request, or unset the env var.

### CORS errors in browser apps

genome doesn't add CORS middleware by default. Add it yourself:
```python
from fastapi.middleware.cors import CORSMiddleware
from genome.server.app import create_app

app = create_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Docker

### Container OOMs on startup

The default Dockerfile pre-downloads the embedding model at build time (~80 MB), and loading it into RAM needs ~500 MB. Give the container at least 1 GB:
```yaml
deploy:
  resources:
    limits:
      memory: 1G
```

### Can't connect to Postgres from genome container

In docker-compose, the genome service talks to `postgres:5432` (the service name, not `localhost`). Verify your `GENOME_STORAGE` env is:
```
postgresql://genome:genome@postgres:5432/memory
```

## Still stuck?

Internal contact: info@northtek.io. Include:
- Python version + OS
- Minimal reproduction script
- Full traceback
- `pip list | grep -E "genome|sentence|torch|psycopg|fastapi"`
