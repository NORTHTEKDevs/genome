"""Security / input-validation tests."""
import numpy as np
import pytest

from genome.memory.facade import Memory
from genome.memory.schema import MemoryRecord
from tests.memory._fake_embed import FakeEmbeddingProvider


def test_oversized_content_rejected():
    with pytest.raises(ValueError, match="exceeds max length"):
        MemoryRecord(
            content="x" * 200_000,
            embedding=np.zeros(4, dtype=np.float32),
        )


def test_oversized_user_id_rejected():
    with pytest.raises(ValueError, match="user_id too long"):
        MemoryRecord(
            content="x",
            embedding=np.zeros(4, dtype=np.float32),
            user_id="a" * 300,
        )


def test_oversized_agent_id_rejected():
    with pytest.raises(ValueError, match="agent_id too long"):
        MemoryRecord(
            content="x",
            embedding=np.zeros(4, dtype=np.float32),
            agent_id="a" * 300,
        )


def test_metadata_with_too_many_keys_rejected():
    with pytest.raises(ValueError, match="metadata has too many keys"):
        MemoryRecord(
            content="x",
            embedding=np.zeros(4, dtype=np.float32),
            metadata={f"key_{i}": i for i in range(200)},
        )


def test_sql_injection_attempts_handled_safely():
    """SQLite parameter binding must neutralize injection attempts."""
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        # Classic SQL injection strings as content
        payloads = [
            "Robert'); DROP TABLE memories;--",
            "1' OR '1'='1",
            "x\"; SELECT * FROM memories;",
        ]
        for p in payloads:
            rec = m.add(p, user_id="attacker")[0]
            # Retrieval should return the exact string as stored -- not executed as SQL
            back = m.get(rec.id)
            assert back.content == p

        # And the store should still be intact
        assert m.count(user_id="attacker") == len(payloads)
    finally:
        m.close()


def test_user_id_sql_injection_safely_scoped():
    """An attacker-controlled user_id cannot escape scope isolation."""
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        m.add("alice secret", user_id="alice")
        m.add("bob secret", user_id="bob")
        # Attacker tries to get everything via injection in user_id param
        results = m.search("'; DROP TABLE memories; --", user_id="attacker")
        assert results == []
        # Store intact
        assert m.count(user_id="alice") == 1
        assert m.count(user_id="bob") == 1
    finally:
        m.close()


def test_json_metadata_with_weird_payload():
    """Metadata with adversarial JSON values should round-trip safely."""
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        weird = {
            "key": "value with quotes \" and backslash \\",
            "nested": {"a": [1, 2, "\"hello\""]},
            "unicode": "émoji 🎉",
        }
        rec = m.add("x", user_id="u", metadata=weird)[0]
        back = m.get(rec.id)
        assert back.metadata == weird
    finally:
        m.close()
