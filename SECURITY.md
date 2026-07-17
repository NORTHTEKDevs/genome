# Security Policy

## Reporting a vulnerability

Email **info@northtek.io** with the details (proof-of-concept appreciated). You will
get an acknowledgment within 72 hours. Please do not open a public issue for
security reports until a fix is released.

## Scope notes

- The **library and the MCP server** make **no network calls** in the default
  write path — memories are embedded locally and stored in local SQLite/Postgres.
  There is no telemetry.
- The **optional REST server** (`genome.server`) is **safe by default**: it binds
  `127.0.0.1` (loopback) and **refuses to start** on a non-loopback interface
  unless `GENOME_API_KEY` is set, because the API is destructive
  (add/update/delete/reset). To expose it beyond localhost, set `GENOME_API_KEY`
  (sent in the `X-API-Key` header) and `GENOME_HOST`. The shipped `docker-compose.yml`
  requires `GENOME_API_KEY` and fails fast if it is unset.
- **Default-deny auth:** independently of how it's launched (including
  `uvicorn ... --host 0.0.0.0`, which bypasses the bind guard), the server returns
  `503` on every endpoint when no `GENOME_API_KEY` is configured, unless the operator
  explicitly sets `GENOME_ALLOW_NO_AUTH=1` for local development. It cannot be run
  unauthenticated by accident.
- Memory *content* is treated as data, not instructions: extraction and conflict
  prompts sanitize stored text against prompt-injection delimiters (see
  `genome/memory/conflict.py`, `extraction.py`), and the test suite includes
  security and tenant-isolation tests (`tests/memory/test_security.py`,
  `test_tenant_isolation.py`).

## Supported versions

The latest minor release receives fixes.
