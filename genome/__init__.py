"""GENOME: DNA-inspired memory primitives for AI."""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

__version__ = "1.0.0"


_RELATION_CONSTANTS = {"SUPERSEDES", "CONTRADICTS", "DERIVED_FROM", "RELATES_TO", "CAUSES"}


_OBSERVABILITY_EXPORTS = {
    "ErrorCapture", "CapturedError", "get_error_capture",
    "MetricsRegistry", "get_metrics", "configure_logging", "get_logger",
}

_CONFLICT_EXPORTS = {"ConflictResolver", "ConflictDecision"}


def __getattr__(name: str):
    # Lazy imports to avoid loading heavy deps (torch, sentence-transformers) on
    # every `import genome`. The user pays for what they use.
    if name == "Memory":
        from genome.memory import Memory
        return Memory
    if name == "AsyncMemory":
        from genome.memory import AsyncMemory
        return AsyncMemory
    if name == "MemoryRecord":
        from genome.memory import MemoryRecord
        return MemoryRecord
    if name == "SearchResult":
        from genome.memory import SearchResult
        return SearchResult
    if name == "MemoryEdge":
        from genome.memory import MemoryEdge
        return MemoryEdge
    if name == "EmbeddingProvider":
        from genome.embeddings import EmbeddingProvider
        return EmbeddingProvider
    if name in _RELATION_CONSTANTS:
        from genome.memory import graph as _graph
        return getattr(_graph, name)
    if name in _OBSERVABILITY_EXPORTS:
        from genome import observability as _obs
        return getattr(_obs, name)
    if name in _CONFLICT_EXPORTS:
        from genome.memory import conflict as _conflict
        return getattr(_conflict, name)
    raise AttributeError(f"module 'genome' has no attribute {name!r}")


__all__ = [
    "Memory",
    "AsyncMemory",
    "MemoryRecord",
    "SearchResult",
    "MemoryEdge",
    "EmbeddingProvider",
    "SUPERSEDES",
    "CONTRADICTS",
    "DERIVED_FROM",
    "RELATES_TO",
    "CAUSES",
    "ErrorCapture",
    "CapturedError",
    "get_error_capture",
    "MetricsRegistry",
    "get_metrics",
    "configure_logging",
    "get_logger",
    "ConflictResolver",
    "ConflictDecision",
    "__version__",
]
