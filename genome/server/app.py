"""FastAPI server exposing genome Memory over HTTP.

Design goals:
- 1:1 mapping to Memory API (add/search/get/update/delete/reset/synthesize/link/related)
- Auto-generated OpenAPI spec at /docs
- Health check at /health (touches the store)
- Configuration via env vars so Docker deployment is trivial

Env vars:
- GENOME_STORAGE             -- SQLite path, ":memory:", or postgresql:// DSN
- GENOME_EMBED_MODEL         -- sentence-transformers model name
- GENOME_CACHE_SIZE          -- response cache LRU capacity
- GENOME_API_KEY             -- if set, required in X-API-Key header on all endpoints
- GENOME_MAX_REQUEST_BYTES   -- request body size limit (default 1 MiB)
- GENOME_LAZY_INIT           -- if set to "1", build Memory on first request
                                 (recommended for uvicorn --reload)

Example:
    GENOME_STORAGE=memories.db \\
    uvicorn genome.server.app:app --host 0.0.0.0 --port 8080
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hmac
import os
from typing import Any

import genome
from genome.memory.facade import Memory
from genome.observability import get_error_capture, get_metrics
from genome.server.models import (
    _MAX_ID_LEN,
    AddRequest,
    ClearedResponse,
    CountResponse,
    DeleteResponse,
    EdgeResponse,
    ErrorsResponse,
    HealthResponse,
    LinkRequest,
    MetricsSnapshot,
    RecordOut,
    ResetResponse,
    SearchHit,
    SearchRequest,
    SynthesizeRequest,
    UpdateRequest,
)


def _require_fastapi():
    try:
        from fastapi import FastAPI  # noqa: F401
        from pydantic import BaseModel  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "genome.server requires fastapi + pydantic. "
            "Install: pip install \"genome[fastapi]\""
        ) from e


def validate_env_config() -> list[str]:
    """Validate server env vars at startup. Returns a list of issue strings.

    Called from `_build_memory_from_env` before any heavy init. Fails fast on
    misconfiguration so operators see the error at startup rather than on the
    first request hours later.
    """
    issues: list[str] = []
    storage = os.environ.get("GENOME_STORAGE", ":memory:")

    # Postgres DSN sanity
    if storage.startswith(("postgresql://", "postgres://")):
        if "@" not in storage or "/" not in storage.split("@", 1)[-1]:
            issues.append(
                f"GENOME_STORAGE looks malformed (expected "
                f"postgresql://user:pass@host:port/dbname): {storage}"
            )

    # Numeric config: must parse
    for name in ("GENOME_CACHE_SIZE", "GENOME_MAX_REQUEST_BYTES", "GENOME_PORT"):
        v = os.environ.get(name)
        if v is not None and v != "":
            try:
                n = int(v)
                if n <= 0:
                    issues.append(f"{name} must be positive, got {n}")
            except ValueError:
                issues.append(f"{name} must be an integer, got {v!r}")

    # API key presence warning: if the server listens on 0.0.0.0 without an
    # API key, that's almost certainly a misconfiguration. We warn but don't
    # refuse (e.g. local dev on localhost is fine).
    host = os.environ.get("GENOME_HOST", "")
    if host in ("0.0.0.0", "::") and not os.environ.get("GENOME_API_KEY"):
        issues.append(
            "GENOME_HOST is bound to all interfaces but GENOME_API_KEY is not "
            "set. This exposes the memory layer unauthenticated. Set "
            "GENOME_API_KEY or bind to 127.0.0.1."
        )

    return issues


def _build_memory_from_env() -> Memory:
    # Fail fast on bad config before doing expensive init (model download etc).
    issues = validate_env_config()
    if issues:
        from genome.errors import ConfigError
        raise ConfigError(
            "invalid env configuration:\n  - " + "\n  - ".join(issues),
            hint="Fix the env vars above and restart. See docs/troubleshooting.md.",
        )

    storage = os.environ.get("GENOME_STORAGE", ":memory:")
    cache_size = int(os.environ.get("GENOME_CACHE_SIZE", "1024"))

    if storage.startswith(("postgresql://", "postgres://")):
        from genome.embeddings import EmbeddingProvider
        from genome.memory.postgres_store import PostgresMemoryStore
        model = os.environ.get(
            "GENOME_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        provider = EmbeddingProvider(model_name=model)
        store = PostgresMemoryStore(
            dsn=storage,
            embedding_dim=provider.dim,
        )
        return Memory(
            storage=store,
            embedding_provider=provider,
            cache_size=cache_size,
        )
    return Memory(storage=storage, cache_size=cache_size)


def _constant_time_api_key_eq(provided: str | None, expected: str) -> bool:
    """Timing-safe API key comparison using hmac.compare_digest.

    Protects against timing oracles that could reveal the secret one byte at
    a time. The fleet-wide FrostByte review-patterns crystal flags this in
    every Next.js sibling repo (FROST, FORGE, APEX, SCOUT, NEXUS); matching the
    secure pattern here so genome doesn't inherit that class.
    """
    if not expected:
        return True  # no key configured == no auth
    if provided is None:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def create_app(memory: Memory | None = None):
    """Create the FastAPI app. Accepts an optional pre-built Memory for testing."""
    _require_fastapi()
    from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
    from fastapi.responses import JSONResponse

    # Lazy memory: wrap in a box so the first request builds it and subsequent
    # requests reuse. Avoids the ~45s embedding-model load at module import.
    _mem_box: dict[str, Memory | None] = {"mem": memory}

    def _memory() -> Memory:
        if _mem_box["mem"] is None:
            _mem_box["mem"] = _build_memory_from_env()
        return _mem_box["mem"]  # type: ignore[return-value]

    app = FastAPI(
        title="genome",
        description=(
            "DNA-inspired memory layer with recombination, graph, and "
            "hierarchical summaries"
        ),
        version=genome.__version__,
    )

    # Opt-in CORS via GENOME_CORS_ORIGINS. Comma-separated list, or "*" for any.
    # Off by default so server-to-server deployments don't pay the cost.
    cors_origins = os.environ.get("GENOME_CORS_ORIGINS", "").strip()
    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware
        origins = ["*"] if cors_origins == "*" else [
            o.strip() for o in cors_origins.split(",") if o.strip()
        ]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    API_KEY = os.environ.get("GENOME_API_KEY", "")
    MAX_REQUEST_BYTES = int(os.environ.get("GENOME_MAX_REQUEST_BYTES", 1 << 20))
    ERROR_CAPTURE = get_error_capture()
    METRICS = get_metrics()

    @app.exception_handler(Exception)
    async def _capture_unhandled(request, exc):  # type: ignore[no-untyped-def]
        """Capture every unhandled exception to the genome-native ErrorCapture
        before letting FastAPI convert it to a 500. HTTPExceptions are handled
        by FastAPI's own machinery and are not captured (they're expected
        client-error signals, not faults)."""
        from fastapi.exceptions import HTTPException as _HTTPExc
        if isinstance(exc, _HTTPExc):
            raise exc
        ERROR_CAPTURE.capture(
            exc, tags={"path": request.url.path, "method": request.method},
        )
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if API_KEY and not _constant_time_api_key_eq(x_api_key, API_KEY):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        """Refuse requests larger than GENOME_MAX_REQUEST_BYTES (default 1 MiB).

        Pydantic's input validators cap individual field lengths but the HTTP
        layer can still buffer the whole body before Pydantic sees it. Refusing
        early prevents a cheap DoS.
        """
        from fastapi.responses import JSONResponse
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                f"request body exceeds {MAX_REQUEST_BYTES} bytes"
                            ),
                        },
                    )
            except ValueError:
                pass  # malformed header, let handlers deal with it
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse, dependencies=[Depends(require_api_key)])
    def health():
        mem = _memory()
        total = mem.count()
        stats = mem.cache_stats
        return HealthResponse(
            status="ok",
            memory_count=total,
            cache_hits=stats.hits if stats else 0,
            cache_misses=stats.misses if stats else 0,
            cache_hit_rate=stats.hit_rate if stats else 0.0,
            version=genome.__version__,
        )

    @app.post(
        "/v1/memories", response_model=list[RecordOut], status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    def add_memory(req: AddRequest):
        mem = _memory()
        records = mem.add(
            req.text,
            user_id=req.user_id,
            agent_id=req.agent_id,
            metadata=req.metadata,
        )
        return [RecordOut.from_record(r) for r in records]

    @app.get(
        "/v1/memories/{memory_id}", response_model=RecordOut,
        dependencies=[Depends(require_api_key)],
    )
    def get_memory(
        memory_id: str,
        user_id: str | None = Query(default=None),
        agent_id: str | None = Query(default=None),
    ):
        r = _memory().get(memory_id, user_id=user_id, agent_id=agent_id)
        if r is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return RecordOut.from_record(r)

    @app.patch(
        "/v1/memories/{memory_id}", response_model=RecordOut,
        dependencies=[Depends(require_api_key)],
    )
    def update_memory(
        memory_id: str,
        req: UpdateRequest,
        user_id: str | None = Query(default=None),
        agent_id: str | None = Query(default=None),
    ):
        r = _memory().update(
            memory_id,
            content=req.content,
            metadata=req.metadata,
            re_embed=req.re_embed,
            user_id=user_id,
            agent_id=agent_id,
        )
        if r is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return RecordOut.from_record(r)

    @app.delete(
        "/v1/memories/{memory_id}", response_model=DeleteResponse,
        dependencies=[Depends(require_api_key)],
    )
    def delete_memory(
        memory_id: str,
        user_id: str | None = Query(default=None),
        agent_id: str | None = Query(default=None),
    ):
        ok = _memory().delete(memory_id, user_id=user_id, agent_id=agent_id)
        if not ok:
            raise HTTPException(status_code=404, detail="memory not found")
        return DeleteResponse(deleted=True, id=memory_id)

    @app.post(
        "/v1/search", response_model=list[SearchHit],
        dependencies=[Depends(require_api_key)],
    )
    def search(req: SearchRequest):
        results = _memory().search(
            req.query,
            user_id=req.user_id,
            agent_id=req.agent_id,
            limit=req.limit,
            filter_parents=req.filter_parents,
        )
        return [
            SearchHit(
                id=r.id,
                content=r.content,
                score=r.score,
                metadata=r.record.metadata,
            )
            for r in results
        ]

    @app.post(
        "/v1/synthesize", response_model=RecordOut,
        dependencies=[Depends(require_api_key)],
    )
    def synthesize(req: SynthesizeRequest):
        try:
            r = _memory().synthesize(
                memory_ids=req.memory_ids,
                operator=req.operator,
                user_id=req.user_id,
                agent_id=req.agent_id,
                content=req.content,
                metadata=req.metadata,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return RecordOut.from_record(r)

    @app.post(
        "/v1/edges", response_model=EdgeResponse,
        dependencies=[Depends(require_api_key)],
    )
    def link(req: LinkRequest):
        try:
            e = _memory().link(
                req.from_id, req.to_id, req.relation,
                weight=req.weight, metadata=req.metadata,
            )
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return EdgeResponse.model_validate(
            {"edge_id": e.id, "from": e.from_id, "to": e.to_id, "relation": e.relation}
        )

    @app.delete(
        "/v1/edges/{edge_id}", response_model=DeleteResponse,
        dependencies=[Depends(require_api_key)],
    )
    def unlink(
        edge_id: str,
        user_id: str | None = Query(default=None),
        agent_id: str | None = Query(default=None),
    ):
        ok = _memory().unlink(edge_id, user_id=user_id, agent_id=agent_id)
        if not ok:
            raise HTTPException(status_code=404, detail="edge not found")
        return DeleteResponse(deleted=True, id=edge_id)

    @app.get(
        "/v1/memories/{memory_id}/related",
        response_model=list[RecordOut],
        dependencies=[Depends(require_api_key)],
    )
    def related(
        memory_id: str,
        relation: str | None = None,
        direction: str = "out",
        user_id: str | None = Query(default=None, max_length=_MAX_ID_LEN),
        agent_id: str | None = Query(default=None, max_length=_MAX_ID_LEN),
    ):
        # A bad `direction` is client error (400), not a server fault (500).
        # Mirrors the try/except pattern used by synthesize/link.
        try:
            recs = _memory().related(
                memory_id, relation=relation, direction=direction,
                user_id=user_id, agent_id=agent_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return [RecordOut.from_record(r) for r in recs]

    @app.delete(
        "/v1/scope", response_model=ResetResponse,
        dependencies=[Depends(require_api_key)],
    )
    def reset_scope(
        user_id: str | None = Query(default=None),
        agent_id: str | None = Query(default=None),
        confirm: bool = Query(default=False),
    ):
        """Reset a scope. If both user_id and agent_id are omitted, requires
        `confirm=true` as a guardrail against accidental global wipe.
        """
        if user_id is None and agent_id is None and not confirm:
            raise HTTPException(
                status_code=400,
                detail=(
                    "global reset requires ?confirm=true. Pass user_id or "
                    "agent_id to scope, or explicitly confirm to wipe all "
                    "memories."
                ),
            )
        count = _memory().reset(user_id=user_id, agent_id=agent_id)
        return ResetResponse(deleted=count)

    @app.get(
        "/v1/count", response_model=CountResponse,
        dependencies=[Depends(require_api_key)],
    )
    def count(user_id: str | None = None, agent_id: str | None = None):
        return CountResponse(count=_memory().count(user_id=user_id, agent_id=agent_id))

    @app.get(
        "/v1/metrics", response_model=MetricsSnapshot,
        dependencies=[Depends(require_api_key)],
    )
    def metrics():
        """Snapshot of in-process metrics (counters + histograms).

        Includes built-in `memory.add.*`, `memory.search.*` counters/timings.
        Forward to Prometheus/OTel via `get_metrics().set_sink(...)`.
        """
        return MetricsSnapshot.model_validate(METRICS.snapshot())

    @app.get(
        "/v1/errors", response_model=ErrorsResponse,
        dependencies=[Depends(require_api_key)],
    )
    def errors(
        grouped: bool = Query(default=True),
        limit: int = Query(default=50, ge=1, le=500),
    ):
        """Captured errors (genome-native, no external service required).

        - `grouped=true` (default): returns deduped fingerprints with counts
          and a sample message -- like Sentry's "Issues" view.
        - `grouped=false`: returns the most recent N raw captures with stacks.
        """
        if grouped:
            return ErrorsResponse.model_validate({"groups": ERROR_CAPTURE.grouped()})
        recents = ERROR_CAPTURE.recent(limit=limit)
        return ErrorsResponse.model_validate(
            {
                "recent": [
                    {
                        "timestamp": e.timestamp,
                        "error_type": e.error_type,
                        "message": e.message,
                        "fingerprint": e.fingerprint,
                        "stack": e.stack,
                        "tags": dict(e.tags),
                    }
                    for e in recents
                ],
            }
        )

    @app.delete(
        "/v1/errors", response_model=ClearedResponse,
        dependencies=[Depends(require_api_key)],
    )
    def clear_errors():
        """Reset the captured-error buffer."""
        ERROR_CAPTURE.reset()
        return ClearedResponse(cleared=True)

    app.state.memory_provider = _memory
    return app


def _get_default_app():
    """Build the ASGI app. If GENOME_LAZY_INIT=1, defer Memory build to first
    request -- lets uvicorn start fast without loading the embedding model.
    """
    if os.environ.get("GENOME_LAZY_INIT", "") == "1":
        return create_app()
    try:
        return create_app()
    except ImportError:
        return None


# Default ASGI app for `uvicorn genome.server.app:app`
app: Any = _get_default_app()
