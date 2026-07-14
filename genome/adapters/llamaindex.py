"""LlamaIndex adapter.

Exposes `GenomeChatMemory`, a BaseMemory-compatible class for LlamaIndex
agents/engines that delegates storage + retrieval to genome.

Install: `pip install llama-index-core` (genome does not depend on it).
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from genome.memory.facade import Memory


def _require_llamaindex():
    try:
        from llama_index.core.llms import ChatMessage
    except ImportError as e:
        raise ImportError(
            "llamaindex adapter requires llama-index-core. "
            "Install with: pip install llama-index-core"
        ) from e
    return ChatMessage


class GenomeChatMemory:
    """A minimal LlamaIndex-compatible ChatMemory backed by genome.

    Stores each turn as a memory with role metadata. On retrieval, returns the
    ordered sequence via `get()`, or the top-k semantically relevant via
    `get_relevant(query)`.

    Usage::

        from genome import Memory
        from genome.adapters.llamaindex import GenomeChatMemory
        from llama_index.core.llms import ChatMessage, MessageRole

        mem = Memory(storage="agent.db")
        chat_memory = GenomeChatMemory(memory=mem, session_id="session_1")
        chat_memory.put(ChatMessage(role=MessageRole.USER, content="hello"))
        messages = chat_memory.get()  # ordered history
        relevant = chat_memory.get_relevant("what did I say about coffee?", top_k=3)
    """

    def __init__(
        self,
        memory: Memory,
        *,
        session_id: str,
        token_limit: int | None = None,
    ) -> None:
        _require_llamaindex()
        self.memory = memory
        self.session_id = session_id
        self.token_limit = token_limit

    def put(self, message: Any) -> None:
        """Add a ChatMessage to memory."""
        role = str(getattr(getattr(message, "role", None), "value", message.role)).lower()
        content = str(message.content)
        self.memory.add(
            content,
            user_id=self.session_id,
            metadata={"role": role, "source": "chat"},
        )

    def set(self, messages: list[Any]) -> None:
        """Replace all messages in this session."""
        self.memory.reset(user_id=self.session_id)
        for m in messages:
            self.put(m)

    def get(self, input: str | None = None, **kwargs: Any) -> list[Any]:
        """Return the full chronological message history for this session.

        Accepts LlamaIndex's `get(input=..., **kwargs)` call signature (the
        agent runtime passes the current user message as `input`) so this
        object is drop-in usable there. `input` is accepted for compatibility;
        chronological history is returned regardless -- use `get_relevant()`
        for retrieval-aware recall.
        """
        from llama_index.core.llms import ChatMessage, MessageRole
        records = self.memory.list_all(user_id=self.session_id)
        records.sort(key=lambda r: r.created_at)
        out: list[ChatMessage] = []
        for r in records:
            role_str = r.metadata.get("role", "user").lower()
            role = {
                "user": MessageRole.USER,
                "human": MessageRole.USER,
                "ai": MessageRole.ASSISTANT,
                "assistant": MessageRole.ASSISTANT,
                "system": MessageRole.SYSTEM,
            }.get(role_str, MessageRole.USER)
            out.append(ChatMessage(role=role, content=r.content))
        return out

    def get_all(self, **kwargs: Any) -> list[Any]:
        """Alias for get() -- matches LlamaIndex BaseMemory naming."""
        return self.get()

    def get_relevant(self, query: str, top_k: int = 5) -> list[Any]:
        """Return the top-k semantically relevant historical messages for the query.

        genome-specific: this is the recall-style retrieval that matters for
        long-running agents whose full history exceeds the context window.
        """
        from llama_index.core.llms import ChatMessage, MessageRole
        results = self.memory.search(query, user_id=self.session_id, limit=top_k)
        out: list[ChatMessage] = []
        for r in results:
            role_str = r.record.metadata.get("role", "user").lower()
            role = {
                "user": MessageRole.USER,
                "human": MessageRole.USER,
                "ai": MessageRole.ASSISTANT,
                "assistant": MessageRole.ASSISTANT,
                "system": MessageRole.SYSTEM,
            }.get(role_str, MessageRole.USER)
            out.append(ChatMessage(role=role, content=r.content))
        return out

    def reset(self) -> None:
        """Clear this session's history."""
        self.memory.reset(user_id=self.session_id)


__all__ = ["GenomeChatMemory"]
