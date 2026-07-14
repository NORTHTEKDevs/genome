/*
 * Copyright 2026 Northtek (FrostByte Digital LLC)
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * genome TypeScript SDK
 *
 * REST client for a genome server. Mirrors the Python `Memory` API shape.
 *
 * @example
 * ```ts
 * import { Memory } from "@frostbyte/genome-memory";
 *
 * const mem = new Memory({ baseUrl: "http://localhost:8080" });
 * await mem.add({ text: "I love pour-over coffee", userId: "alice" });
 * const results = await mem.search({
 *   query: "what drinks does the user like?",
 *   userId: "alice",
 *   limit: 5,
 * });
 * for (const r of results) console.log(r.content, r.score);
 * ```
 */

// ---------- types ----------

export interface MemoryRecord {
  id: string;
  content: string;
  user_id: string | null;
  agent_id: string | null;
  created_at: number;
  accessed_at: number;
  access_count: number;
  parents: string[];
  operator: string | null;
  metadata: Record<string, unknown>;
}

export interface SearchHit {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  memory_count: number;
  cache_hits: number;
  cache_misses: number;
  cache_hit_rate: number;
  version: string;
}

export interface MemoryEdge {
  edge_id: string;
  from: string;
  to: string;
  relation: string;
}

// ---------- client options ----------

export interface MemoryOptions {
  /** Genome server base URL (e.g. "http://localhost:8080"). */
  baseUrl: string;
  /** Optional X-API-Key value (set on server via GENOME_API_KEY). */
  apiKey?: string;
  /** Optional fetch implementation (defaults to globalThis.fetch). */
  fetch?: typeof fetch;
  /** Timeout per request, ms (default 30s). */
  timeoutMs?: number;
}

// ---------- request types ----------

export interface AddRequest {
  text: string;
  userId?: string;
  agentId?: string;
  metadata?: Record<string, unknown>;
}

export interface SearchRequest {
  query: string;
  userId?: string;
  agentId?: string;
  limit?: number;
  filterParents?: boolean;
}

export interface UpdateRequest {
  content?: string;
  metadata?: Record<string, unknown>;
  reEmbed?: boolean;
}

export interface SynthesizeRequest {
  memoryIds: string[];
  operator?: string;
  userId?: string;
  agentId?: string;
  content?: string;
  metadata?: Record<string, unknown>;
}

export interface LinkRequest {
  fromId: string;
  toId: string;
  relation: string;
  weight?: number;
  metadata?: Record<string, unknown>;
}

// ---------- observability ----------

export interface MetricsSnapshot {
  counters: Record<string, Array<{ tags: Record<string, string>; value: number }>>;
  histograms: Record<
    string,
    Array<{
      tags: Record<string, string>;
      count: number;
      sum: number;
      mean: number;
      mean_recent: number;
      max_recent: number;
    }>
  >;
}

export interface ErrorGroup {
  fingerprint: string;
  count: number;
  error_type: string;
  message: string;
  last_seen: number;
  tags: Record<string, string>;
}

export interface ErrorRecent {
  timestamp: number;
  error_type: string;
  message: string;
  fingerprint: string;
  stack: string;
  tags: Record<string, string>;
}

export interface ErrorsResponse {
  groups?: ErrorGroup[];
  recent?: ErrorRecent[];
}

// ---------- error ----------

export class GenomeError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown, message?: string) {
    super(message ?? `genome server returned ${status}: ${JSON.stringify(detail)}`);
    this.name = "GenomeError";
    this.status = status;
    this.detail = detail;
  }
}

// ---------- main client ----------

export class Memory {
  private baseUrl: string;
  private apiKey?: string;
  private fetch: typeof fetch;
  private timeoutMs: number;

  constructor(options: MemoryOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.fetch = options.fetch ?? globalThis.fetch;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    if (!this.fetch) {
      throw new Error(
        "No fetch implementation available. Pass one via options.fetch " +
        "or run on Node >= 18 / modern browser.",
      );
    }
  }

  private async request<T>(
    method: string,
    path: string,
    options: { body?: unknown; query?: Record<string, string | number | boolean | undefined> } = {},
  ): Promise<T> {
    const url = new URL(this.baseUrl + path);
    if (options.query) {
      for (const [k, v] of Object.entries(options.query)) {
        if (v !== undefined) url.searchParams.set(k, String(v));
      }
    }
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    if (this.apiKey) headers["X-API-Key"] = this.apiKey;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await this.fetch(url.toString(), {
        method,
        headers,
        body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
        signal: controller.signal,
      });
      if (!res.ok) {
        let detail: unknown = null;
        try {
          detail = await res.json();
        } catch {
          detail = await res.text();
        }
        throw new GenomeError(res.status, detail);
      }
      if (res.status === 204) return undefined as T;
      return (await res.json()) as T;
    } finally {
      clearTimeout(timeout);
    }
  }

  // ---------- public API ----------

  /** Get server health + cache stats. */
  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/health");
  }

  /** Add one memory (may be split into multiple atomic records by server-side extractor). */
  async add(req: AddRequest): Promise<MemoryRecord[]> {
    return this.request<MemoryRecord[]>("POST", "/v1/memories", {
      body: {
        text: req.text,
        user_id: req.userId,
        agent_id: req.agentId,
        metadata: req.metadata,
      },
    });
  }

  /** Get a memory by id; optionally scope-enforced. */
  async get(
    memoryId: string,
    opts: { userId?: string; agentId?: string } = {},
  ): Promise<MemoryRecord> {
    return this.request<MemoryRecord>("GET", `/v1/memories/${encodeURIComponent(memoryId)}`, {
      query: { user_id: opts.userId, agent_id: opts.agentId },
    });
  }

  /** Update a memory's content or metadata. */
  async update(
    memoryId: string,
    req: UpdateRequest,
    opts: { userId?: string; agentId?: string } = {},
  ): Promise<MemoryRecord> {
    return this.request<MemoryRecord>("PATCH", `/v1/memories/${encodeURIComponent(memoryId)}`, {
      body: {
        content: req.content,
        metadata: req.metadata,
        re_embed: req.reEmbed,
      },
      query: { user_id: opts.userId, agent_id: opts.agentId },
    });
  }

  /** Delete a memory. Returns true if deleted. */
  async delete(
    memoryId: string,
    opts: { userId?: string; agentId?: string } = {},
  ): Promise<boolean> {
    try {
      await this.request<{ deleted: boolean }>(
        "DELETE", `/v1/memories/${encodeURIComponent(memoryId)}`,
        { query: { user_id: opts.userId, agent_id: opts.agentId } },
      );
      return true;
    } catch (e) {
      if (e instanceof GenomeError && e.status === 404) return false;
      throw e;
    }
  }

  /** Cosine-similarity search within scope. Parent filtering ON by default. */
  async search(req: SearchRequest): Promise<SearchHit[]> {
    return this.request<SearchHit[]>("POST", "/v1/search", {
      body: {
        query: req.query,
        user_id: req.userId,
        agent_id: req.agentId,
        limit: req.limit ?? 10,
        filter_parents: req.filterParents ?? true,
      },
    });
  }

  /** Recombine 2+ parent memories into a hybrid. Parents must share scope. */
  async synthesize(req: SynthesizeRequest): Promise<MemoryRecord> {
    return this.request<MemoryRecord>("POST", "/v1/synthesize", {
      body: {
        memory_ids: req.memoryIds,
        operator: req.operator ?? "uniform_crossover",
        user_id: req.userId,
        agent_id: req.agentId,
        content: req.content,
        metadata: req.metadata,
      },
    });
  }

  /** Create a typed directed edge between two memories in the same scope. */
  async link(req: LinkRequest): Promise<MemoryEdge> {
    return this.request<MemoryEdge>("POST", "/v1/edges", {
      body: {
        from_id: req.fromId,
        to_id: req.toId,
        relation: req.relation,
        weight: req.weight ?? 1.0,
        metadata: req.metadata,
      },
    });
  }

  /**
   * Delete an edge.
   *
   * Pass `userId`/`agentId` to enforce tenant scope: the deletion is refused
   * (returns false) if the edge's `from` endpoint does not match the given
   * scope. Mirrors the Python `Memory.unlink()` signature so a TS caller
   * can't accidentally bypass the cross-tenant defense by guessing edge ids.
   */
  async unlink(
    edgeId: string,
    opts: { userId?: string; agentId?: string } = {},
  ): Promise<boolean> {
    try {
      await this.request<{ deleted: boolean }>(
        "DELETE", `/v1/edges/${encodeURIComponent(edgeId)}`,
        {
          query: {
            user_id: opts.userId,
            agent_id: opts.agentId,
          },
        },
      );
      return true;
    } catch (e) {
      if (e instanceof GenomeError && e.status === 404) return false;
      throw e;
    }
  }

  /** Get memories linked to/from this one via typed edges. */
  async related(
    memoryId: string,
    opts: {
      relation?: string;
      direction?: "out" | "in" | "both";
      userId?: string;
      agentId?: string;
    } = {},
  ): Promise<MemoryRecord[]> {
    return this.request<MemoryRecord[]>(
      "GET",
      `/v1/memories/${encodeURIComponent(memoryId)}/related`,
      {
        query: {
          relation: opts.relation,
          direction: opts.direction ?? "out",
          user_id: opts.userId,
          agent_id: opts.agentId,
        },
      },
    );
  }

  /** Reset a scope. If both userId and agentId are omitted, pass confirm=true. */
  async reset(
    opts: { userId?: string; agentId?: string; confirm?: boolean } = {},
  ): Promise<number> {
    const res = await this.request<{ deleted: number }>("DELETE", "/v1/scope", {
      query: {
        user_id: opts.userId,
        agent_id: opts.agentId,
        confirm: opts.confirm,
      },
    });
    return res.deleted;
  }

  /** Count memories in a scope. */
  async count(opts: { userId?: string; agentId?: string } = {}): Promise<number> {
    const res = await this.request<{ count: number }>("GET", "/v1/count", {
      query: { user_id: opts.userId, agent_id: opts.agentId },
    });
    return res.count;
  }

  /**
   * Snapshot of in-process metrics (counters + histograms) from the server's
   * MetricsRegistry. Forward to Prometheus / OTel via the Python-side
   * `get_metrics().set_sink(...)` if you need long-term storage.
   */
  async metrics(): Promise<MetricsSnapshot> {
    return this.request<MetricsSnapshot>("GET", "/v1/metrics");
  }

  /**
   * Captured server-side errors. Genome's built-in Sentry-equivalent.
   *
   * - `grouped: true` (default) returns deduped fingerprints with counts --
   *   like Sentry's "Issues" view.
   * - `grouped: false` returns the most recent N raw captures with stacks.
   */
  async errors(
    opts: { grouped?: boolean; limit?: number } = {},
  ): Promise<ErrorsResponse> {
    return this.request<ErrorsResponse>("GET", "/v1/errors", {
      query: {
        grouped: opts.grouped ?? true,
        limit: opts.limit ?? 50,
      },
    });
  }

  /** Reset the captured-error buffer. Returns true if cleared. */
  async clearErrors(): Promise<boolean> {
    const res = await this.request<{ cleared: boolean }>("DELETE", "/v1/errors");
    return res.cleared;
  }
}

// Re-export for convenience
export default Memory;
