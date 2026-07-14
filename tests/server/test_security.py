"""Security-focused server tests covering the R1 fixes."""

import pytest

from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from genome.server.app import _constant_time_api_key_eq, create_app  # noqa: E402

# ---------- timing-safe API key ----------

def test_constant_time_api_key_empty_expected_passes():
    assert _constant_time_api_key_eq(None, "") is True
    assert _constant_time_api_key_eq("anything", "") is True


def test_constant_time_api_key_missing_provided_fails():
    assert _constant_time_api_key_eq(None, "secret") is False


def test_constant_time_api_key_match():
    assert _constant_time_api_key_eq("secret", "secret") is True


def test_constant_time_api_key_mismatch():
    assert _constant_time_api_key_eq("wrong", "secret") is False


# ---------- reset guardrail ----------

def test_reset_scope_without_args_requires_confirm(monkeypatch):
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    client = TestClient(app)
    try:
        mem.add("fact", user_id="alice")

        # No args, no confirm -> 400
        r = client.delete("/v1/scope")
        assert r.status_code == 400
        assert "confirm=true" in r.json()["detail"]

        # Data still there
        assert mem.count(user_id="alice") == 1

        # With confirm -> succeeds
        r2 = client.delete("/v1/scope", params={"confirm": "true"})
        assert r2.status_code == 200
        assert mem.count() == 0
    finally:
        mem.close()


def test_reset_scope_with_user_id_no_confirm_needed():
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    client = TestClient(app)
    try:
        mem.add("fact", user_id="alice")
        r = client.delete("/v1/scope", params={"user_id": "alice"})
        assert r.status_code == 200
        assert r.json()["deleted"] == 1
    finally:
        mem.close()


# ---------- request size limit ----------

def test_request_size_limit_rejects_large_body(monkeypatch):
    monkeypatch.setenv("GENOME_MAX_REQUEST_BYTES", "1024")  # 1 KB limit
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    client = TestClient(app)
    try:
        big_text = "x" * 10_000
        r = client.post(
            "/v1/memories",
            json={"text": big_text, "user_id": "u"},
        )
        assert r.status_code == 413
        assert "exceeds" in r.json()["detail"]
    finally:
        mem.close()


def test_request_size_limit_default_allows_normal():
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    client = TestClient(app)
    try:
        r = client.post(
            "/v1/memories",
            json={"text": "normal size", "user_id": "u"},
        )
        assert r.status_code == 201
    finally:
        mem.close()


# ---------- version pulled from package ----------

def test_health_reports_package_version():
    import genome
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    client = TestClient(app)
    try:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["version"] == genome.__version__
    finally:
        mem.close()


def test_app_openapi_version_matches_package():
    """The FastAPI app's declared version must match the package, not a hardcode."""
    import genome
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    try:
        assert app.version == genome.__version__
    finally:
        mem.close()


# ---------- tenant isolation via REST ----------

def test_rest_cross_tenant_synthesize_refused():
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    client = TestClient(app)
    try:
        alice = client.post(
            "/v1/memories", json={"text": "alice", "user_id": "alice"}
        ).json()[0]["id"]
        bob = client.post(
            "/v1/memories", json={"text": "bob", "user_id": "bob"}
        ).json()[0]["id"]

        # Alice tries to synthesize using Bob's memory -> 400
        r = client.post(
            "/v1/synthesize",
            json={
                "memory_ids": [alice, bob],
                "user_id": "alice",
                "operator": "simple_average",
            },
        )
        assert r.status_code == 400
        assert "user_id" in r.json()["detail"].lower()
    finally:
        mem.close()


def test_rest_cross_tenant_delete_returns_404():
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    client = TestClient(app)
    try:
        alice_id = client.post(
            "/v1/memories", json={"text": "alice", "user_id": "alice"}
        ).json()[0]["id"]

        # Bob tries to delete Alice's memory scoped as bob
        r = client.delete(
            f"/v1/memories/{alice_id}",
            params={"user_id": "bob"},
        )
        assert r.status_code == 404

        # Alice's memory still exists
        r2 = client.get(f"/v1/memories/{alice_id}")
        assert r2.status_code == 200
    finally:
        mem.close()


def test_rest_cross_tenant_get_returns_404():
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    client = TestClient(app)
    try:
        alice_id = client.post(
            "/v1/memories", json={"text": "alice", "user_id": "alice"}
        ).json()[0]["id"]

        # Scoped as bob -> 404
        r = client.get(
            f"/v1/memories/{alice_id}",
            params={"user_id": "bob"},
        )
        assert r.status_code == 404

        # Scoped as alice -> 200
        r2 = client.get(
            f"/v1/memories/{alice_id}",
            params={"user_id": "alice"},
        )
        assert r2.status_code == 200
    finally:
        mem.close()
