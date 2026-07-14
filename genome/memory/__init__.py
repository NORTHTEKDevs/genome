"""GENOME memory layer.

A Mem0-shaped memory API with one differentiator: recombination as a first-class
memory operation. Any two (or N) memories can be synthesized into a new memory
whose embedding is a biologically-inspired recombination of its parents, with
full provenance tracking and parent-filtered retrieval.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from genome.memory.graph import (
    CAUSES,
    CONTRADICTS,
    DERIVED_FROM,
    RELATES_TO,
    SUPERSEDES,
    MemoryEdge,
)
from genome.memory.schema import MemoryRecord, SearchResult

# Memory facades are imported lazily to avoid circular imports during partial builds.

__all__ = [
    "MemoryRecord",
    "SearchResult",
    "MemoryEdge",
    "SUPERSEDES",
    "CONTRADICTS",
    "DERIVED_FROM",
    "RELATES_TO",
    "CAUSES",
    "Memory",
    "AsyncMemory",
]


def __getattr__(name: str):  # pragma: no cover - import shim
    if name == "Memory":
        from genome.memory.facade import Memory
        return Memory
    if name == "AsyncMemory":
        from genome.memory.async_facade import AsyncMemory
        return AsyncMemory
    raise AttributeError(f"module 'genome.memory' has no attribute {name!r}")
