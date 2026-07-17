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
  `uvicorn ... --host 0.0.0.0` or `gunicorn --bind`, which bypass the bind guard),
  the server returns `503` on every endpoint when no `GENOME_API_KEY` is configured.
  The only exception is `GENOME_ALLOW_NO_AUTH=1`, and even that is **confined to
  loopback peers**: an unauthenticated request from a non-loopback client is
  refused regardless of how the process was bound, because the check reads the
  actual peer address (which the app sees) rather than the bind argument (which it
  can't). So the opt-in cannot expose the API to the network even if you bind
  `0.0.0.0` by mistake. To serve real clients, set `GENOME_API_KEY`. Caveat: the
  loopback check reads the real socket peer (it deliberately ignores
  `X-Forwarded-For`, which a client can forge), so if you put a **reverse proxy on
  the same host** in front of the server the peer becomes loopback and the opt-in
  would serve through it. Behind any proxy, set `GENOME_API_KEY` — do not rely on
  `GENOME_ALLOW_NO_AUTH`, which is strictly a keyless-local-dev convenience.
- **Single-key trust model (by design):** the REST `GENOME_API_KEY` is a single,
  full-access operator credential. Per-tenant isolation is enforced only when the
  caller passes `user_id`/`agent_id`; a request that holds the key and omits them
  can read or delete across tenants, and `DELETE /v1/scope?confirm=true` is a
  global wipe. This is intentional (the key holder is the trusted operator), but
  if you build a multi-tenant service on the REST API you must pass the tenant
  identity on every call and treat the key as an admin secret. The embedded
  library and MCP server enforce tenant scoping directly and are not affected.
- **Docker:** the shipped `docker-compose.yml` requires both `GENOME_API_KEY` and
  `POSTGRES_PASSWORD` (fails fast if unset) and publishes Postgres only on
  `127.0.0.1`, so the database is not reachable from the LAN and cannot be used to
  bypass the API's auth gate.
- Memory *content* is treated as data, not instructions: extraction and conflict
  prompts sanitize stored text against prompt-injection delimiters (see
  `genome/memory/conflict.py`, `extraction.py`), and the test suite includes
  security and tenant-isolation tests (`tests/memory/test_security.py`,
  `test_tenant_isolation.py`).

## Supported versions

The latest minor release receives fixes.
