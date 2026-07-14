import time

import numpy as np
import pytest

from genome.memory.schema import MemoryRecord, SearchResult


def test_memory_record_minimal():
    rec = MemoryRecord(content="hello", embedding=np.zeros(8, dtype=np.float32))
    assert rec.content == "hello"
    assert rec.embedding.shape == (8,)
    assert rec.id.startswith("mem_")
    assert rec.parents == []
    assert rec.operator is None
    assert rec.access_count == 0
    assert not rec.is_synthesized


def test_memory_record_coerces_embedding_dtype():
    rec = MemoryRecord(content="x", embedding=np.zeros(4, dtype=np.float64))
    assert rec.embedding.dtype == np.float32


def test_memory_record_rejects_non_ndarray():
    with pytest.raises(TypeError):
        MemoryRecord(content="x", embedding=[0, 0, 0])  # type: ignore[arg-type]


def test_memory_record_rejects_2d_embedding():
    with pytest.raises(ValueError):
        MemoryRecord(content="x", embedding=np.zeros((2, 4), dtype=np.float32))


def test_memory_record_rejects_empty_content():
    with pytest.raises(ValueError):
        MemoryRecord(content="", embedding=np.zeros(4, dtype=np.float32))


def test_memory_record_synthesized_has_parents():
    rec = MemoryRecord(
        content="hybrid",
        embedding=np.zeros(4, dtype=np.float32),
        parents=["mem_a", "mem_b"],
        operator="uniform_crossover",
    )
    assert rec.is_synthesized
    assert rec.parents == ["mem_a", "mem_b"]
    assert rec.operator == "uniform_crossover"


def test_memory_record_age_seconds_nonneg():
    rec = MemoryRecord(content="x", embedding=np.zeros(4, dtype=np.float32))
    assert rec.age_seconds >= 0
    # Same record right after creation: age <= 1 sec
    assert rec.age_seconds < 2.0


def test_memory_record_custom_timestamps():
    t0 = time.time() - 1000
    rec = MemoryRecord(
        content="x",
        embedding=np.zeros(4, dtype=np.float32),
        created_at=t0,
        accessed_at=t0,
    )
    assert rec.age_seconds > 900


def test_search_result_delegates():
    rec = MemoryRecord(content="hello", embedding=np.zeros(4, dtype=np.float32))
    sr = SearchResult(record=rec, score=0.87)
    assert sr.content == "hello"
    assert sr.id == rec.id
    assert sr.score == 0.87
