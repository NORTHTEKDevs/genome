"""Actionable errors for genome.

Each error carries a short `hint` attribute with a concrete next step for the user.
These are raised instead of bare ValueError / RuntimeError wherever the failure
reason is diagnosable.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations


class GenomeError(ValueError):
    """Base exception. Carries a `hint` with a concrete fix suggestion.

    Inherits from ValueError so existing code that catches ValueError still
    works unchanged, but callers who want structured errors can catch
    `GenomeError` or its subclasses.
    """

    hint: str = ""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint or self.hint

    def __str__(self) -> str:
        base = super().__str__()
        if self.hint:
            return f"{base}\n  Hint: {self.hint}"
        return base


class MemoryNotFoundError(GenomeError):
    """Raised when a memory id doesn't exist."""

    def __init__(self, memory_id: str) -> None:
        super().__init__(
            f"memory not found: {memory_id}",
            hint=(
                "Call Memory.list_all(user_id=...) to see available ids, "
                "or check your scoping (user_id / agent_id)."
            ),
        )


class InvalidEmbeddingError(GenomeError):
    """Raised when an embedding has the wrong shape/dtype."""

    hint = "EmbeddingProvider.encode() must return a 1-D float32 numpy array."


class ScopeError(GenomeError):
    """Raised when an operation spans scopes that should be isolated."""


class OperatorError(GenomeError):
    """Raised when a recombination operator is invalid or fails."""


class SynthesisError(GenomeError):
    """Raised for synthesize-specific failures (too few parents, mixed scopes, ...)."""


class CorruptedStoreError(GenomeError):
    """Raised when persisted state is inconsistent (edge pointing to a missing
    memory, orphaned synthesized record with missing parents, etc.)."""


class ConfigError(GenomeError):
    """Raised on invalid configuration (e.g., bad DSN, missing env var)."""


__all__ = [
    "GenomeError",
    "MemoryNotFoundError",
    "InvalidEmbeddingError",
    "ScopeError",
    "OperatorError",
    "SynthesisError",
    "CorruptedStoreError",
    "ConfigError",
]
