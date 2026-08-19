"""GENOME: DNA-inspired memory primitives for AI."""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

# Single-sourced from package metadata (pyproject.toml) so the module can never
# disagree with the released version again; 1.0.4-1.0.6 shipped self-reporting 1.0.3
# because this string was bumped by hand and forgotten.
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("genome-memory")
except Exception:  # running from a source tree without installed dist metadata
    __version__ = "0.0.0+source"


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
