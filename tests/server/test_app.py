import pytest

from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from genome.server.app import create_app  # noqa: E402


@pytest.fixture
def client():
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=16))
    app = create_app(memory=mem)
    c = TestClient(app)
    yield c
    mem.close()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["memory_count"] == 0


def test_add_memory(client):
    r = client.post(
        "/v1/memories",
        json={"text": "user likes coffee", "user_id": "alice"},
    )
    assert r.status_code == 201
    records = r.json()
    assert len(records) == 1
    assert records[0]["content"] == "user likes coffee"
    assert records[0]["user_id"] == "alice"
    assert records[0]["id"].startswith("mem_")


def test_get_memory(client):
    create = client.post(
        "/v1/memories",
        json={"text": "hello", "user_id": "alice"},
    ).json()
    mem_id = create[0]["id"]

    r = client.get(f"/v1/memories/{mem_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["content"] == "hello"

    # 404 on missing
    r404 = client.get("/v1/memories/mem_nonexistent")
    assert r404.status_code == 404


def test_update_memory(client):
    mem_id = client.post(
        "/v1/memories",
        json={"text": "original", "user_id": "alice"},
    ).json()[0]["id"]

    r = client.patch(
        f"/v1/memories/{mem_id}",
        json={"content": "updated"},
    )
    assert r.status_code == 200
    assert r.json()["content"] == "updated"


def test_delete_memory(client):
    mem_id = client.post(
        "/v1/memories",
        json={"text": "bye", "user_id": "alice"},
    ).json()[0]["id"]

    r = client.delete(f"/v1/memories/{mem_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    # Second delete: 404
    r2 = client.delete(f"/v1/memories/{mem_id}")
    assert r2.status_code == 404


def test_search(client):
    client.post("/v1/memories", json={"text": "fact 1", "user_id": "alice"})
    client.post("/v1/memories", json={"text": "fact 2", "user_id": "alice"})
    client.post("/v1/memories", json={"text": "other user", "user_id": "bob"})

    r = client.post("/v1/search", json={"query": "fact", "user_id": "alice", "limit": 5})
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 2
    for h in hits:
        assert "id" in h and "content" in h and "score" in h


def test_synthesize(client):
    a_id = client.post("/v1/memories", json={"text": "alpha", "user_id": "u"}).json()[0]["id"]
    b_id = client.post("/v1/memories", json={"text": "beta", "user_id": "u"}).json()[0]["id"]

    r = client.post(
        "/v1/synthesize",
        json={
            "memory_ids": [a_id, b_id],
            "operator": "simple_average",
            "user_id": "u",
        },
    )
    assert r.status_code == 200
    hybrid = r.json()
    assert set(hybrid["parents"]) == {a_id, b_id}
    assert hybrid["operator"] == "simple_average"


def test_synthesize_single_parent_400(client):
    a_id = client.post("/v1/memories", json={"text": "alpha", "user_id": "u"}).json()[0]["id"]
    # Only 1 parent -> Pydantic validation kicks in (min_length=2)
    r = client.post(
        "/v1/synthesize",
        json={"memory_ids": [a_id], "user_id": "u"},
    )
    assert r.status_code == 422


def test_synthesize_missing_parent_400(client):
    a_id = client.post("/v1/memories", json={"text": "alpha", "user_id": "u"}).json()[0]["id"]
    r = client.post(
        "/v1/synthesize",
        json={"memory_ids": [a_id, "mem_nonexistent"], "user_id": "u"},
    )
    assert r.status_code == 400


def test_link_and_related(client):
    a_id = client.post("/v1/memories", json={"text": "A", "user_id": "u"}).json()[0]["id"]
    b_id = client.post("/v1/memories", json={"text": "B", "user_id": "u"}).json()[0]["id"]

    r = client.post(
        "/v1/edges",
        json={
            "from_id": a_id,
            "to_id": b_id,
            "relation": "relates_to",
            "weight": 0.8,
        },
    )
    assert r.status_code == 200
    edge_id = r.json()["edge_id"]

    # related
    r2 = client.get(f"/v1/memories/{a_id}/related", params={"relation": "relates_to"})
    assert r2.status_code == 200
    related = r2.json()
    assert len(related) == 1
    assert related[0]["id"] == b_id

    # unlink
    r3 = client.delete(f"/v1/edges/{edge_id}")
    assert r3.status_code == 200


def test_reset_scope(client):
    client.post("/v1/memories", json={"text": "f1", "user_id": "u"})
    client.post("/v1/memories", json={"text": "f2", "user_id": "u"})
    r = client.delete("/v1/scope", params={"user_id": "u"})
    assert r.status_code == 200
    assert r.json()["deleted"] == 2


def test_count(client):
    client.post("/v1/memories", json={"text": "x", "user_id": "alice"})
    client.post("/v1/memories", json={"text": "y", "user_id": "bob"})
    r = client.get("/v1/count", params={"user_id": "alice"})
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_openapi_docs_reachable(client):
    r = client.get("/docs")
    assert r.status_code == 200


def test_api_key_required(monkeypatch):
    """When GENOME_API_KEY is set, endpoints require it."""
    monkeypatch.setenv("GENOME_API_KEY", "secret-123")
    # Build a fresh app that reads the env
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(memory=mem)
    c = TestClient(app)

    # No key -> 401
    r = c.post("/v1/memories", json={"text": "x", "user_id": "u"})
    assert r.status_code == 401

    # With key -> 201
    r2 = c.post(
        "/v1/memories",
        json={"text": "x", "user_id": "u"},
        headers={"X-API-Key": "secret-123"},
    )
    assert r2.status_code == 201
    mem.close()
