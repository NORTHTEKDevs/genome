/*
 * Copyright 2026 Northtek (FrostByte Digital LLC)
 * SPDX-License-Identifier: BUSL-1.1
 */

import { strict as assert } from "node:assert";
import { test } from "node:test";
import { GenomeError, Memory } from "./index.ts";

// ---------- test helpers ----------

interface CapturedCall {
  url: string;
  init: RequestInit;
}

function mockFetch(response: {
  status?: number;
  body?: unknown;
  throwOn?: string;
}): { fetch: typeof fetch; calls: CapturedCall[] } {
  const calls: CapturedCall[] = [];
  const fetchImpl = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({ url, init: init ?? {} });
    if (response.throwOn && url.includes(response.throwOn)) {
      throw new Error("network blew up");
    }
    const status = response.status ?? 200;
    return new Response(
      response.body !== undefined ? JSON.stringify(response.body) : null,
      {
        status,
        headers: { "Content-Type": "application/json" },
      },
    );
  }) as typeof fetch;
  return { fetch: fetchImpl, calls };
}

// ---------- constructor ----------

test("Memory requires fetch", () => {
  const originalFetch = globalThis.fetch;
  // @ts-expect-error -- deliberately deleting
  globalThis.fetch = undefined;
  try {
    assert.throws(() => new Memory({ baseUrl: "http://localhost:8080" }));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Memory strips trailing slash from baseUrl", async () => {
  const { fetch, calls } = mockFetch({ body: { status: "ok", memory_count: 0, cache_hits: 0, cache_misses: 0, cache_hit_rate: 0, version: "test" } });
  const mem = new Memory({ baseUrl: "http://localhost:8080/", fetch });
  await mem.health();
  assert.equal(calls.length, 1);
  assert.ok(calls[0]!.url.startsWith("http://localhost:8080/health"));
});

test("Memory sends X-API-Key when configured", async () => {
  const { fetch, calls } = mockFetch({ body: { status: "ok", memory_count: 0, cache_hits: 0, cache_misses: 0, cache_hit_rate: 0, version: "test" } });
  const mem = new Memory({ baseUrl: "http://localhost:8080", fetch, apiKey: "secret-123" });
  await mem.health();
  const hdrs = calls[0]!.init.headers as Record<string, string>;
  assert.equal(hdrs["X-API-Key"], "secret-123");
});

test("Memory omits X-API-Key when not configured", async () => {
  const { fetch, calls } = mockFetch({ body: { status: "ok", memory_count: 0, cache_hits: 0, cache_misses: 0, cache_hit_rate: 0, version: "test" } });
  const mem = new Memory({ baseUrl: "http://localhost:8080", fetch });
  await mem.health();
  const hdrs = calls[0]!.init.headers as Record<string, string>;
  assert.equal(hdrs["X-API-Key"], undefined);
});

// ---------- add ----------

test("add POSTs /v1/memories with snake_case body", async () => {
  const { fetch, calls } = mockFetch({
    status: 201,
    body: [
      {
        id: "mem_1",
        content: "hello",
        user_id: "alice",
        agent_id: null,
        created_at: 1,
        accessed_at: 1,
        access_count: 0,
        parents: [],
        operator: null,
        metadata: {},
      },
    ],
  });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const recs = await mem.add({ text: "hello", userId: "alice" });
  assert.equal(recs.length, 1);
  assert.equal(recs[0]!.id, "mem_1");

  const c = calls[0]!;
  assert.equal(c.init.method, "POST");
  assert.ok(c.url.endsWith("/v1/memories"));
  const body = JSON.parse(c.init.body as string);
  assert.equal(body.text, "hello");
  assert.equal(body.user_id, "alice");
});

// ---------- get ----------

test("get with user_id adds query param", async () => {
  const { fetch, calls } = mockFetch({
    body: {
      id: "mem_1",
      content: "x",
      user_id: "alice",
      agent_id: null,
      created_at: 1,
      accessed_at: 1,
      access_count: 0,
      parents: [],
      operator: null,
      metadata: {},
    },
  });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const rec = await mem.get("mem_1", { userId: "alice" });
  assert.equal(rec.id, "mem_1");
  assert.match(calls[0]!.url, /user_id=alice/);
});

// ---------- delete ----------

test("delete returns true on success", async () => {
  const { fetch } = mockFetch({ body: { deleted: true, id: "mem_1" } });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  assert.equal(await mem.delete("mem_1"), true);
});

test("delete returns false on 404 (not an error)", async () => {
  const { fetch } = mockFetch({ status: 404, body: { detail: "memory not found" } });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  assert.equal(await mem.delete("mem_nope"), false);
});

test("delete rethrows on non-404 errors", async () => {
  const { fetch } = mockFetch({ status: 500, body: { detail: "boom" } });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  await assert.rejects(() => mem.delete("mem_1"), GenomeError);
});

// ---------- search ----------

test("search sends filter_parents=true by default", async () => {
  const { fetch, calls } = mockFetch({ body: [] });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  await mem.search({ query: "x", userId: "alice" });
  const body = JSON.parse(calls[0]!.init.body as string);
  assert.equal(body.filter_parents, true);
  assert.equal(body.user_id, "alice");
  assert.equal(body.limit, 10);
});

test("search respects custom filter_parents and limit", async () => {
  const { fetch, calls } = mockFetch({ body: [] });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  await mem.search({ query: "x", userId: "alice", filterParents: false, limit: 3 });
  const body = JSON.parse(calls[0]!.init.body as string);
  assert.equal(body.filter_parents, false);
  assert.equal(body.limit, 3);
});

// ---------- synthesize ----------

test("synthesize POSTs with memory_ids snake-cased", async () => {
  const { fetch, calls } = mockFetch({
    body: {
      id: "mem_hybrid",
      content: "hybrid",
      user_id: "alice",
      agent_id: null,
      created_at: 1,
      accessed_at: 1,
      access_count: 0,
      parents: ["a", "b"],
      operator: "uniform_crossover",
      metadata: {},
    },
  });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const hybrid = await mem.synthesize({
    memoryIds: ["a", "b"],
    userId: "alice",
    operator: "uniform_crossover",
  });
  assert.equal(hybrid.id, "mem_hybrid");
  const body = JSON.parse(calls[0]!.init.body as string);
  assert.deepEqual(body.memory_ids, ["a", "b"]);
});

// ---------- link ----------

test("link POSTs /v1/edges with from_id / to_id", async () => {
  const { fetch, calls } = mockFetch({
    body: { edge_id: "edge_1", from: "a", to: "b", relation: "supersedes" },
  });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const edge = await mem.link({ fromId: "a", toId: "b", relation: "supersedes" });
  assert.equal(edge.edge_id, "edge_1");
  const body = JSON.parse(calls[0]!.init.body as string);
  assert.equal(body.from_id, "a");
  assert.equal(body.to_id, "b");
});

// ---------- errors ----------

test("GenomeError carries status + detail", async () => {
  const { fetch } = mockFetch({
    status: 400,
    body: { detail: "bad request" },
  });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  try {
    await mem.add({ text: "x" });
    assert.fail("should have thrown");
  } catch (e) {
    assert.ok(e instanceof GenomeError);
    assert.equal(e.status, 400);
    assert.deepEqual(e.detail, { detail: "bad request" });
  }
});

// ---------- count ----------

test("count returns number from {count} envelope", async () => {
  const { fetch } = mockFetch({ body: { count: 42 } });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  assert.equal(await mem.count({ userId: "alice" }), 42);
});

// ---------- reset ----------

test("reset passes confirm=true for global wipe", async () => {
  const { fetch, calls } = mockFetch({ body: { deleted: 100 } });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const deleted = await mem.reset({ confirm: true });
  assert.equal(deleted, 100);
  assert.match(calls[0]!.url, /confirm=true/);
});

// ---------- related ----------

test("related encodes direction and relation in query", async () => {
  const { fetch, calls } = mockFetch({ body: [] });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  await mem.related("mem_1", { relation: "supersedes", direction: "in", userId: "alice" });
  assert.match(calls[0]!.url, /relation=supersedes/);
  assert.match(calls[0]!.url, /direction=in/);
  assert.match(calls[0]!.url, /user_id=alice/);
});

// ---------- unlink scope (R5 parity) ----------

test("unlink propagates userId/agentId to query string", async () => {
  const { fetch, calls } = mockFetch({ body: { deleted: true, id: "edge_x" } });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const ok = await mem.unlink("edge_x", { userId: "alice", agentId: "convo1" });
  assert.equal(ok, true);
  assert.match(calls[0]!.url, /user_id=alice/);
  assert.match(calls[0]!.url, /agent_id=convo1/);
});

test("unlink without scope still works for backward compat", async () => {
  const { fetch } = mockFetch({ body: { deleted: true, id: "edge_x" } });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const ok = await mem.unlink("edge_x");
  assert.equal(ok, true);
});

// ---------- observability methods (R-silent-lever fix) ----------

test("metrics() hits GET /v1/metrics and returns snapshot shape", async () => {
  const snapshot = {
    counters: { "memory.add.count": [{ tags: { user_id: "alice" }, value: 3 }] },
    histograms: {},
  };
  const { fetch, calls } = mockFetch({ body: snapshot });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const got = await mem.metrics();
  // method defaults to GET when not specified, so init.method may be undefined
  assert.ok(calls[0]!.init.method === "GET" || calls[0]!.init.method === undefined);
  assert.match(calls[0]!.url, /\/v1\/metrics$/);
  assert.deepEqual(got.counters["memory.add.count"][0].value, 3);
});

test("errors() defaults to grouped=true and forwards limit", async () => {
  const body = {
    groups: [
      { fingerprint: "abc", count: 5, error_type: "ValueError", message: "x", last_seen: 0, tags: {} },
    ],
  };
  const { fetch, calls } = mockFetch({ body });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const got = await mem.errors({ limit: 100 });
  assert.match(calls[0]!.url, /grouped=true/);
  assert.match(calls[0]!.url, /limit=100/);
  assert.equal(got.groups?.[0].count, 5);
});

test("errors({grouped:false}) returns recent stacks", async () => {
  const body = {
    recent: [
      {
        timestamp: 123,
        error_type: "RuntimeError",
        message: "boom",
        fingerprint: "deadbeef",
        stack: "Traceback...",
        tags: { path: "/v1/synthesize" },
      },
    ],
  };
  const { fetch, calls } = mockFetch({ body });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const got = await mem.errors({ grouped: false });
  assert.match(calls[0]!.url, /grouped=false/);
  assert.equal(got.recent?.[0].error_type, "RuntimeError");
});

test("clearErrors() returns boolean from {cleared}", async () => {
  const { fetch, calls } = mockFetch({ body: { cleared: true } });
  const mem = new Memory({ baseUrl: "http://x", fetch });
  const ok = await mem.clearErrors();
  assert.equal(ok, true);
  assert.equal(calls[0]!.init.method, "DELETE");
  assert.match(calls[0]!.url, /\/v1\/errors$/);
});
