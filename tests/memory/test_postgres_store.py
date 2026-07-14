"""Postgres backend tests.

Pure-Python tests (no DB required):
- Import guards
- Vector literal formatting
- SQL construction

Integration tests (require POSTGRES_DSN env var pointing at a pg+pgvector instance):
- Full CRUD roundtrip
- Search with pgvector HNSW

If POSTGRES_DSN is not set, integration tests are skipped.
"""
import os

import numpy as np
import pytest

from genome.memory.postgres_store import PostgresMemoryStore


def test_vec_literal_format():
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lit = PostgresMemoryStore._vec_literal(vec)
    assert lit.startswith("[")
    assert lit.endswith("]")
    # Values parseable
    parsed = [float(x) for x in lit.strip("[]").split(",")]
    np.testing.assert_allclose(parsed, [0.1, 0.2, 0.3], atol=1e-5)


def test_vec_literal_handles_large_vector():
    vec = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    lit = PostgresMemoryStore._vec_literal(vec)
    parsed = [float(x) for x in lit.strip("[]").split(",")]
    assert len(parsed) == 384


def test_missing_psycopg_raises_clean_error(monkeypatch):
    """If psycopg isn't installed, instantiation should raise a clear ImportError
    pointing the user to the right install command.

    We simulate the missing import by renaming the module.
    """
    import builtins
    import sys
    real_import = builtins.__import__

    def fake_import(name, *args, **kw):
        if name == "psycopg":
            raise ImportError("No module named 'psycopg'")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Pop any cached module
    sys.modules.pop("psycopg", None)

    with pytest.raises(ImportError) as ei:
        PostgresMemoryStore(dsn="postgresql://fake", embedding_dim=4)
    assert "psycopg" in str(ei.value)
    assert "genome[postgres]" in str(ei.value) or "pip install" in str(ei.value)


# ---------- Live integration tests (opt-in via POSTGRES_DSN) ----------

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "")


def _drop_tables(dsn: str) -> None:
    """Clean slate: drop memories + edges so the next test can pick its own dim."""
    import psycopg
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS edges CASCADE")
        cur.execute("DROP TABLE IF EXISTS memories CASCADE")


@pytest.fixture
def clean_pg():
    """Fixture: drop tables before/after each test so tests don't cross-contaminate."""
    if POSTGRES_DSN:
        _drop_tables(POSTGRES_DSN)
    yield
    if POSTGRES_DSN:
        _drop_tables(POSTGRES_DSN)


@pytest.mark.skipif(not POSTGRES_DSN, reason="POSTGRES_DSN not set")
def test_integration_add_and_get(clean_pg):
    store = PostgresMemoryStore(dsn=POSTGRES_DSN, embedding_dim=8)
    try:
        from genome.memory.schema import MemoryRecord
        rec = MemoryRecord(
            content="hello pg",
            embedding=np.ones(8, dtype=np.float32),
            user_id="test_user",
        )
        store.add(rec)
        back = store.get(rec.id)
        assert back is not None
        assert back.content == "hello pg"
        np.testing.assert_allclose(back.embedding, rec.embedding, atol=1e-4)
        store.delete(rec.id)
    finally:
        store.close()


@pytest.mark.skipif(not POSTGRES_DSN, reason="POSTGRES_DSN not set")
def test_integration_search_returns_nearest(clean_pg):
    store = PostgresMemoryStore(dsn=POSTGRES_DSN, embedding_dim=4)
    try:
        from genome.memory.schema import MemoryRecord
        for i, vec in enumerate([
            np.array([1, 0, 0, 0], dtype=np.float32),
            np.array([0, 1, 0, 0], dtype=np.float32),
            np.array([0, 0, 1, 0], dtype=np.float32),
        ]):
            store.add(MemoryRecord(
                content=f"v{i}",
                embedding=vec,
                user_id="search_test",
            ))
        results = store.search(
            np.array([0.99, 0.01, 0, 0], dtype=np.float32),
            user_id="search_test",
            limit=3,
        )
        assert len(results) == 3
        assert results[0].record.content == "v0"
    finally:
        store.close()


@pytest.mark.skipif(not POSTGRES_DSN, reason="POSTGRES_DSN not set")
def test_integration_dim_mismatch_raises_clear_error(clean_pg):
    """Creating a second store with a different embedding_dim must raise
    a clear error instead of silently keeping the old schema."""
    # First store: 8-dim
    s1 = PostgresMemoryStore(dsn=POSTGRES_DSN, embedding_dim=8)
    s1.close()
    # Second store with different dim on same schema -> ValueError
    with pytest.raises(ValueError) as ei:
        PostgresMemoryStore(dsn=POSTGRES_DSN, embedding_dim=4)
    assert "vector(" in str(ei.value)
    assert "embedding_dim=4" in str(ei.value)
