"""Tool-call schemas for the agent memory API.

Exposes the AgentMemory methods as tool schemas that LLMs can call. We ship
both Anthropic- and OpenAI-format schemas so users can pass whichever their
SDK expects.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

# The tool schemas below describe the 6 agent-memory operations:
# core_append, core_replace, archival_insert, archival_search,
# archival_delete, synthesize_memories.

ANTHROPIC_TOOL_SCHEMAS: list[dict] = [
    {
        "name": "core_append",
        "description": (
            "Append text to a core memory block. Core memory is always in the "
            "system prompt so the agent should keep it concise. Use this to "
            "record persistent facts about the user or assistant persona."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Which core block to append to (e.g. 'persona', 'user', 'scratch').",
                },
                "text": {
                    "type": "string",
                    "description": "The text to append.",
                },
            },
            "required": ["label", "text"],
        },
    },
    {
        "name": "core_replace",
        "description": (
            "Replace a substring in a core memory block. Use to update facts "
            "that have changed (e.g. user moved, job changed)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "old": {"type": "string", "description": "Exact substring to replace."},
                "new": {"type": "string", "description": "Replacement text."},
            },
            "required": ["label", "old", "new"],
        },
    },
    {
        "name": "archival_insert",
        "description": (
            "Save a memory to archival (long-term) storage. Use this for "
            "facts that don't need to be always-in-context but should be "
            "retrievable later."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory text to store.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "archival_search",
        "description": (
            "Retrieve relevant memories from archival storage using semantic "
            "search. Use before answering questions where you might already "
            "know something."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "archival_delete",
        "description": "Delete an archival memory by id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "synthesize_memories",
        "description": (
            "Combine 2+ archival memories into a single hybrid memory using "
            "embedding-space recombination. Useful for compressing several "
            "related observations into a summary without an LLM round-trip."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "The ids of memories to combine (must share scope).",
                },
                "operator": {
                    "type": "string",
                    "enum": [
                        "uniform_crossover",
                        "frequency_crossover",
                        "simple_average",
                        "uniform_crossover_with_mutation",
                    ],
                    "default": "uniform_crossover",
                    "description": "Recombination operator.",
                },
            },
            "required": ["memory_ids"],
        },
    },
]


def _anthropic_to_openai(schema: dict) -> dict:
    """Translate an Anthropic tool schema into OpenAI's function-call format."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        },
    }


OPENAI_TOOL_SCHEMAS: list[dict] = [
    _anthropic_to_openai(s) for s in ANTHROPIC_TOOL_SCHEMAS
]


def tool_schemas(format: str = "anthropic") -> list[dict]:
    """Return tool schemas in the requested format.

    `format`: "anthropic" (default) or "openai".
    """
    format = format.lower()
    if format == "anthropic":
        return list(ANTHROPIC_TOOL_SCHEMAS)
    if format == "openai":
        return list(OPENAI_TOOL_SCHEMAS)
    raise ValueError(f"unknown format {format!r}; use 'anthropic' or 'openai'")


__all__ = [
    "ANTHROPIC_TOOL_SCHEMAS",
    "OPENAI_TOOL_SCHEMAS",
    "tool_schemas",
]
