# @northtek/genome-memory

TypeScript / JavaScript client for GENOME, an open-source memory layer for AI
agents (Apache-2.0).

Works against any running genome REST server. Mirrors the Python `Memory` API
shape.

## Install

```bash
npm install @northtek/genome-memory
```

ESM-only. Node 20+ or any modern browser.

## Quickstart

First, run the genome server (from the Python package):

```bash
pip install -e ".[fastapi]"
python -m genome.server  # listens on :8080
```

Then from your Node/TS app:

```ts
import { Memory } from "@northtek/genome-memory";

const mem = new Memory({
  baseUrl: "http://localhost:8080",
  apiKey: process.env.GENOME_API_KEY, // optional
});

// Add
await mem.add({
  text: "I love pour-over coffee",
  userId: "alice",
});

// Search (parent filtering ON by default)
const results = await mem.search({
  query: "what drinks does the user like?",
  userId: "alice",
  limit: 5,
});
for (const r of results) console.log(r.content, r.score);

// Synthesize a hybrid from N parent memories (recombination primitive)
const ids = results.map(r => r.id);
const hybrid = await mem.synthesize({
  memoryIds: ids,
  userId: "alice",
  operator: "uniform_crossover",
});
console.log("hybrid:", hybrid.id, hybrid.content);

// Typed graph edges
const alice = (await mem.add({ text: "Alice left NYC", userId: "alice" }))[0]!;
const old = (await mem.search({ query: "lives in NYC", userId: "alice" }))[0]!;
await mem.link({
  fromId: alice.id,
  toId: old.id,
  relation: "supersedes",
  weight: 0.9,
});

// Enforce tenant isolation on reads
await mem.get("mem_xyz", { userId: "alice" }); // 404 if it's actually bob's
```

## API

All methods are async. Full type definitions ship with the package.

| Method | HTTP |
|---|---|
| `new Memory({ baseUrl, apiKey?, fetch?, timeoutMs? })` | — |
| `mem.health()` | `GET /health` |
| `mem.add({ text, userId?, agentId?, metadata? })` | `POST /v1/memories` |
| `mem.get(id, { userId?, agentId? })` | `GET /v1/memories/:id` |
| `mem.update(id, { content?, metadata?, reEmbed? }, scope?)` | `PATCH /v1/memories/:id` |
| `mem.delete(id, { userId?, agentId? })` | `DELETE /v1/memories/:id` |
| `mem.search({ query, userId?, agentId?, limit?, filterParents? })` | `POST /v1/search` |
| `mem.synthesize({ memoryIds, operator?, userId?, ... })` | `POST /v1/synthesize` |
| `mem.link({ fromId, toId, relation, weight?, metadata? })` | `POST /v1/edges` |
| `mem.unlink(edgeId)` | `DELETE /v1/edges/:id` |
| `mem.related(id, { relation?, direction?, userId?, agentId? })` | `GET /v1/memories/:id/related` |
| `mem.reset({ userId?, agentId?, confirm? })` | `DELETE /v1/scope` |
| `mem.count({ userId?, agentId? })` | `GET /v1/count` |

## Error handling

```ts
import { GenomeError } from "@northtek/genome-memory";

try {
  await mem.synthesize({ memoryIds: [a, b], userId: "alice" });
} catch (e) {
  if (e instanceof GenomeError) {
    console.error("status:", e.status, "detail:", e.detail);
  }
}
```

`delete()` and `unlink()` return `false` on 404 instead of throwing — consistent with the sync Python behavior.

## Requirements

- Node 20+ (uses global `fetch`) or any modern browser
- A running genome server (self-host with `python -m genome.server` or via Docker)

## Build from source

```bash
cd sdks/typescript
npm install
npm run build   # -> dist/
npm test        # node --test on src/*.test.ts
```

## License

Apache License 2.0. See [`../../LICENSE`](../../LICENSE) at the repo root.
