"""LangChain adapter.

Two classes exposed:

1. `GenomeChatMessageHistory` -- implements `BaseChatMessageHistory`. Persists
   conversation turns as memories in a genome store, scoped by `session_id`.
   Drop-in replacement for `ChatMessageHistory` / `FileChatMessageHistory`.

2. `GenomeRetrieverMemory` -- implements `BaseMemory` for retrieval-augmented
   chat. Calls `Memory.search` to surface relevant historical memories for the
   current prompt. Use this in LangChain chains that want semantic recall.

Install: `pip install langchain-core` (genome does not depend on it).
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from genome.memory.facade import Memory

if TYPE_CHECKING:
    # Only imported for type checking; runtime import happens lazily below.
    from langchain_core.messages import BaseMessage


def _require_langchain():
    try:
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    except ImportError as e:
        raise ImportError(
            "langchain adapter requires langchain-core. "
            "Install with: pip install langchain-core"
        ) from e
    return AIMessage, HumanMessage, BaseMessage


class GenomeChatMessageHistory:
    """A BaseChatMessageHistory backed by a genome Memory store.

    Each message (human or AI) becomes a memory. The session_id scopes them via
    user_id, so multiple concurrent conversations stay isolated.

    Usage::

        from genome import Memory
        from genome.adapters.langchain import GenomeChatMessageHistory

        mem = Memory(storage="chat.db")
        history = GenomeChatMessageHistory(memory=mem, session_id="session_123")
        # Use as a normal LangChain BaseChatMessageHistory
    """

    def __init__(self, memory: Memory, session_id: str) -> None:
        _require_langchain()  # will raise if not installed
        self.memory = memory
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        from langchain_core.messages import AIMessage, HumanMessage
        records = self.memory.list_all(user_id=self.session_id)
        records.sort(key=lambda r: r.created_at)
        out: list[BaseMessage] = []
        for r in records:
            role = r.metadata.get("role", "human")
            if role == "ai":
                out.append(AIMessage(content=r.content))
            else:
                out.append(HumanMessage(content=r.content))
        return out

    def add_message(self, message: BaseMessage) -> None:
        role = "ai" if message.__class__.__name__ == "AIMessage" else "human"
        self.memory.add(
            str(message.content),
            user_id=self.session_id,
            metadata={"role": role, "source": "chat"},
        )

    def add_messages(self, messages) -> None:
        """Bulk add. Part of the current BaseChatMessageHistory contract --
        RunnableWithMessageHistory calls this rather than add_message."""
        for m in messages:
            self.add_message(m)

    def add_user_message(self, message: str) -> None:
        self.memory.add(
            message,
            user_id=self.session_id,
            metadata={"role": "human", "source": "chat"},
        )

    def add_ai_message(self, message: str) -> None:
        self.memory.add(
            message,
            user_id=self.session_id,
            metadata={"role": "ai", "source": "chat"},
        )

    def clear(self) -> None:
        self.memory.reset(user_id=self.session_id)


class GenomeRetrieverMemory:
    """A BaseMemory-compatible class that injects relevant historical memories
    into the chain's variables on each call.

    Usage::

        from genome import Memory
        from genome.adapters.langchain import GenomeRetrieverMemory
        from langchain_core.prompts import ChatPromptTemplate

        mem = Memory(storage="chat.db")
        retriever = GenomeRetrieverMemory(
            memory=mem, user_id="alice", memory_key="history", top_k=5,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Relevant memories:\\n{history}"),
            ("user", "{input}"),
        ])
        chain = prompt | llm
        chain.invoke({"input": "what do I like?", **retriever.load_memory_variables({"input": "what do I like?"})})
    """

    def __init__(
        self,
        memory: Memory,
        *,
        user_id: str,
        memory_key: str = "history",
        input_key: str = "input",
        top_k: int = 5,
        agent_id: str | None = None,
    ) -> None:
        _require_langchain()
        self.memory = memory
        self.user_id = user_id
        self.memory_key = memory_key
        self.input_key = input_key
        self.top_k = top_k
        self.agent_id = agent_id

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        query = inputs.get(self.input_key) or inputs.get("query") or ""
        if not query:
            return {self.memory_key: ""}
        results = self.memory.search(
            str(query),
            user_id=self.user_id,
            agent_id=self.agent_id,
            limit=self.top_k,
        )
        lines = [f"- {r.content}" for r in results]
        return {self.memory_key: "\n".join(lines) if lines else "(no relevant memories)"}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Persist the turn. Both the user input and AI output become memories."""
        user_text = str(inputs.get(self.input_key, "") or "")
        ai_text = str(outputs.get("output", "") or outputs.get("response", "") or "")
        if user_text:
            self.memory.add(
                user_text,
                user_id=self.user_id,
                agent_id=self.agent_id,
                metadata={"role": "human", "source": "chat"},
            )
        if ai_text:
            self.memory.add(
                ai_text,
                user_id=self.user_id,
                agent_id=self.agent_id,
                metadata={"role": "ai", "source": "chat"},
            )

    def clear(self) -> None:
        self.memory.reset(user_id=self.user_id, agent_id=self.agent_id)


__all__ = ["GenomeChatMessageHistory", "GenomeRetrieverMemory"]
