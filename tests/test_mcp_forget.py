# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""The MCP ``forget`` tool must never delete an unrelated memory.

External review (2026-08-29): ``forget`` searched with ``limit=1`` and deleted
``hits[0]`` unconditionally. ``Memory.search`` has no relevance floor anywhere,
so any query against a non-empty scope deleted SOMETHING: the nearest
neighbour, however far away. The floor has to live in the tool.
"""

from __future__ import annotations

import numpy as np
import pytest

from genome import Memory
from genome.mcp import server


class KeywordEmbedder:
    """Orthogonal by construction: texts mentioning a dentist map to e0, every
    other text to e1. Cosine is exactly 1.0 within a group and 0.0 across."""

    dim = 4
    model_name = "keyword"

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        v[0 if "dentist" in text.lower() else 1] = 1.0
        return v

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])


@pytest.fixture
def mcp_memory(monkeypatch):
    m = Memory(storage=":memory:", embedding_provider=KeywordEmbedder())
    monkeypatch.setattr(server, "_mem", m)
    yield m
    m.close()


DENTIST = "The user's dentist is Dr. Okafor"


def test_unrelated_query_does_not_delete_the_nearest_neighbour(mcp_memory):
    server.remember(DENTIST, user_id="u")
    out = server.forget("what is the weather like today", user_id="u")
    assert out.startswith("Not forgetting"), out
    assert "0.00" in out, "the refusal names the best candidate's score"
    assert "Dr. Okafor" in server.recall("dentist", user_id="u")


def test_matching_query_deletes(mcp_memory):
    server.remember(DENTIST, user_id="u")
    out = server.forget("who is my dentist", user_id="u")
    assert out.startswith("Forgot:"), out
    assert server.recall("dentist", user_id="u").startswith("No relevant memories")


def test_min_score_zero_restores_nearest_neighbour_semantics(mcp_memory):
    server.remember(DENTIST, user_id="u")
    out = server.forget("what is the weather like today", user_id="u", min_score=0.0)
    assert out.startswith("Forgot:"), out


def test_empty_scope_forgets_nothing(mcp_memory):
    assert server.forget("dentist", user_id="nobody").startswith("Nothing to forget")


def test_scope_isolation_survives_the_floor(mcp_memory):
    server.remember(DENTIST, user_id="alice")
    out = server.forget("who is my dentist", user_id="bob")
    assert out.startswith("Nothing to forget"), out
    assert "Dr. Okafor" in server.recall("dentist", user_id="alice")
