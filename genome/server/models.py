"""Pydantic request/response models for the genome FastAPI server.

Kept at module level so FastAPI's type introspection resolves them correctly.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = dict  # type: ignore[assignment,misc]

    def Field(*args, **kw):  # type: ignore[no-redef]
        return None


# Mirrors MemoryRecord.MAX_USER_ID_LEN / MAX_AGENT_ID_LEN. Enforce at the REST
# boundary so oversize ids surface as a 422 (Pydantic) instead of a 500
# (MemoryRecord rejecting deep in the stack with a generic "internal error").
_MAX_ID_LEN = 256
_MAX_CONTENT_LEN = 100_000  # mirrors MemoryRecord.MAX_CONTENT_LEN


class AddRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=_MAX_CONTENT_LEN)
    user_id: str | None = Field(default=None, max_length=_MAX_ID_LEN)
    agent_id: str | None = Field(default=None, max_length=_MAX_ID_LEN)
    metadata: dict[str, Any] | None = None


class UpdateRequest(BaseModel):
    content: str | None = Field(default=None, max_length=_MAX_CONTENT_LEN)
    metadata: dict[str, Any] | None = None
    re_embed: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_MAX_CONTENT_LEN)
    user_id: str | None = Field(default=None, max_length=_MAX_ID_LEN)
    agent_id: str | None = Field(default=None, max_length=_MAX_ID_LEN)
    limit: int = Field(default=10, ge=1, le=100)
    filter_parents: bool = True


class SynthesizeRequest(BaseModel):
    # Cap the fan-out: synthesize does one store.get() per id in a loop, so an
    # unbounded list is a cheap DoS within the 1 MiB body limit.
    memory_ids: list[str] = Field(..., min_length=2, max_length=100)
    operator: str = "uniform_crossover"
    user_id: str | None = Field(default=None, max_length=_MAX_ID_LEN)
    agent_id: str | None = Field(default=None, max_length=_MAX_ID_LEN)
    content: str | None = Field(default=None, max_length=_MAX_CONTENT_LEN)
    metadata: dict[str, Any] | None = None


class LinkRequest(BaseModel):
    from_id: str = Field(..., max_length=_MAX_ID_LEN)
    to_id: str = Field(..., max_length=_MAX_ID_LEN)
    relation: str = Field(..., min_length=1, max_length=_MAX_ID_LEN)
    weight: float = 1.0
    metadata: dict[str, Any] | None = None


class RecordOut(BaseModel):
    id: str
    content: str
    user_id: str | None
    agent_id: str | None
    created_at: float
    accessed_at: float
    access_count: int
    parents: list[str]
    operator: str | None
    metadata: dict[str, Any]

    @classmethod
    def from_record(cls, r) -> RecordOut:
        return cls(
            id=r.id,
            content=r.content,
            user_id=r.user_id,
            agent_id=r.agent_id,
            created_at=r.created_at,
            accessed_at=r.accessed_at,
            access_count=r.access_count,
            parents=r.parents,
            operator=r.operator,
            metadata=r.metadata,
        )


class SearchHit(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    memory_count: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    version: str


class EdgeResponse(BaseModel):
    """Response shape for POST /v1/edges (link)."""
    model_config = ConfigDict(populate_by_name=True)

    edge_id: str
    from_id: str = Field(..., alias="from")
    to_id: str = Field(..., alias="to")
    relation: str


class DeleteResponse(BaseModel):
    """Response shape for DELETE /v1/memories/{id} and /v1/edges/{id}."""
    deleted: bool
    id: str


class ResetResponse(BaseModel):
    """Response shape for DELETE /v1/scope."""
    deleted: int


class CountResponse(BaseModel):
    """Response shape for GET /v1/count."""
    count: int


class ClearedResponse(BaseModel):
    """Response shape for DELETE /v1/errors."""
    cleared: bool


class ErrorGroup(BaseModel):
    """One row in GET /v1/errors?grouped=true."""
    fingerprint: str
    count: int
    error_type: str
    message: str
    last_seen: float
    tags: dict[str, str]


class ErrorRecent(BaseModel):
    """One row in GET /v1/errors?grouped=false."""
    timestamp: float
    error_type: str
    message: str
    fingerprint: str
    stack: str
    tags: dict[str, str]


class ErrorsResponse(BaseModel):
    """Union response for GET /v1/errors. Exactly one field is populated
    depending on the `grouped` query param."""
    groups: list[ErrorGroup] | None = None
    recent: list[ErrorRecent] | None = None


class MetricsSnapshot(BaseModel):
    """Loose response shape for GET /v1/metrics. Counters and histograms
    are arbitrarily-named, so the inner shape is dict-of-list."""
    counters: dict[str, list[dict[str, Any]]]
    histograms: dict[str, list[dict[str, Any]]]
