# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""GENOME_REQUIRE_SCOPE forces per-tenant isolation at the API boundary: every
data operation must carry user_id/agent_id, and the global (all-tenant) reset is
disabled. Closes the single-global-key footgun for multi-tenant deployments."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from genome.memory.facade import Memory  # noqa: E402
from genome.server.app import create_app  # noqa: E402
from tests.memory._fake_embed import FakeEmbeddingProvider  # noqa: E402


def _client(monkeypatch, strict):
    monkeypatch.delenv("GENOME_API_KEY", raising=False)
    monkeypatch.setenv("GENOME_ALLOW_NO_AUTH", "1")
    if strict:
        monkeypatch.setenv("GENOME_REQUIRE_SCOPE", "1")
    else:
        monkeypatch.delenv("GENOME_REQUIRE_SCOPE", raising=False)
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    return TestClient(create_app(memory=mem)), mem


def test_strict_add_without_scope_is_400(monkeypatch):
    c, mem = _client(monkeypatch, strict=True)
    try:
        r = c.post("/v1/memories", json={"text": "unscoped"})
        assert r.status_code == 400
        assert "user_id" in r.json()["detail"]
        assert mem.count() == 0
    finally:
        mem.close()


def test_strict_add_with_scope_ok(monkeypatch):
    c, mem = _client(monkeypatch, strict=True)
    try:
        r = c.post("/v1/memories", json={"text": "scoped", "user_id": "alice"})
        assert r.status_code == 201
    finally:
        mem.close()


def test_strict_get_and_search_and_count_without_scope_400(monkeypatch):
    c, mem = _client(monkeypatch, strict=True)
    try:
        assert c.get("/v1/memories/whatever").status_code == 400
        assert c.post("/v1/search", json={"query": "x"}).status_code == 400
        assert c.get("/v1/count").status_code == 400
    finally:
        mem.close()


def test_strict_blank_scope_is_400(monkeypatch):
    # A whitespace-only scope is not a real tenant; strict mode rejects it.
    c, mem = _client(monkeypatch, strict=True)
    try:
        assert c.get("/v1/count", params={"user_id": "   "}).status_code == 400
        r = c.post("/v1/memories", json={"text": "x", "user_id": " "})
        assert r.status_code == 400
        assert mem.count() == 0
    finally:
        mem.close()


def test_strict_global_reset_forbidden_even_with_confirm(monkeypatch):
    c, mem = _client(monkeypatch, strict=True)
    try:
        mem.add("fact", user_id="alice")
        r = c.delete("/v1/scope", params={"confirm": "true"})
        assert r.status_code == 403
        assert mem.count(user_id="alice") == 1  # nothing wiped
    finally:
        mem.close()


def test_strict_scoped_reset_ok(monkeypatch):
    c, mem = _client(monkeypatch, strict=True)
    try:
        mem.add("fact", user_id="alice")
        r = c.delete("/v1/scope", params={"user_id": "alice"})
        assert r.status_code == 200
        assert r.json()["deleted"] == 1
    finally:
        mem.close()


def test_nonstrict_global_reset_still_works(monkeypatch):
    # Default (no GENOME_REQUIRE_SCOPE): the confirm=true global wipe is preserved.
    c, mem = _client(monkeypatch, strict=False)
    try:
        mem.add("fact", user_id="alice")
        assert c.post("/v1/memories", json={"text": "unscoped"}).status_code == 201
        r = c.delete("/v1/scope", params={"confirm": "true"})
        assert r.status_code == 200
        assert mem.count() == 0
    finally:
        mem.close()
