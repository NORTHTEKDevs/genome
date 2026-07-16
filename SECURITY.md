# Security Policy

## Reporting a vulnerability

Email **info@northtek.io** with the details (proof-of-concept appreciated). You will
get an acknowledgment within 72 hours. Please do not open a public issue for
security reports until a fix is released.

## Scope notes

- GENOME's default write path makes **no network calls** — memories are embedded
  locally and stored in local SQLite/Postgres. There is no telemetry.
- The optional REST server (`genome.server`) is intended to run inside your own
  network; put it behind your own auth/agent gateway if exposed.
- Memory *content* is treated as data, not instructions: extraction and conflict
  prompts sanitize stored text against prompt-injection delimiters (see
  `genome/memory/conflict.py`, `extraction.py`), and the test suite includes
  security and tenant-isolation tests (`tests/memory/test_security.py`,
  `test_tenant_isolation.py`).

## Supported versions

The latest minor release receives fixes.
