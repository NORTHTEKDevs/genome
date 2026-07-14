"""Fake embedding provider for fast memory tests.

Produces deterministic pseudo-embeddings via hashing. Doesn't load any model.
"""
from __future__ import annotations

import hashlib

import numpy as np


class FakeEmbeddingProvider:
    """Deterministic, zero-dep fake embedder. Same string -> same vector."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.model_name = "fake"

    def encode(self, text: str) -> np.ndarray:
        return _hash_embed(text, self.dim)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])


def _hash_embed(text: str, dim: int) -> np.ndarray:
    """Use SHA-256 as a pseudo-random source seeded by the text."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Stretch the hash to `dim` floats in [-1, 1]
    reps = (dim * 4 + len(h) - 1) // len(h)
    raw = (h * reps)[: dim * 4]
    ints = np.frombuffer(raw, dtype=np.int32)
    floats = ints.astype(np.float32) / (2**31)
    return floats[:dim].astype(np.float32)
