"""Hierarchical agent memory: core (in-prompt, editable) + archival (persistent).

Core memory: small, always-in-context blocks (persona, user profile, scratch).
The agent edits these via `core_append` / `core_replace` tool calls. They get
rendered into the system prompt on every turn.

Archival memory: the regular `genome.Memory` store. The agent calls
`archival_insert` / `archival_search` to work with long-term memory beyond
the context window.

Design: Letta's MemGPT hierarchy, but stripped down to just the memory
primitives. The agent loop, context-window paging policy, and summarization
are the caller's responsibility (genome provides tools, not an agent runtime).
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genome.memory.facade import Memory
from genome.memory.schema import MemoryRecord

CORE_MEMORY_OPERATOR = "core_memory"


@dataclass
class CoreBlock:
    """One labeled block of core (always-in-context) memory.

    Blocks are typically short: a persona ("You are a helpful assistant"), a
    user profile (extracted facts about the current user), or a scratch pad.
    They have a character cap so they don't eat the context window.
    """
    label: str
    value: str = ""
    max_chars: int = 2000
    description: str = ""

    def append(self, text: str) -> None:
        new = self.value + ("\n" if self.value else "") + text
        if len(new) > self.max_chars:
            raise ValueError(
                f"core block {self.label!r} would exceed {self.max_chars} chars "
                f"(has {len(self.value)}, trying to add {len(text)}). "
                f"Use core_replace instead, or bump max_chars."
            )
        self.value = new

    def replace(self, old: str, new: str) -> None:
        if old not in self.value:
            raise ValueError(
                f"'{old}' not found in core block {self.label!r}"
            )
        self.value = self.value.replace(old, new, 1)
        if len(self.value) > self.max_chars:
            raise ValueError(
                f"replace would exceed max_chars of {self.label!r}"
            )


@dataclass
class AgentMemory:
    """Combined core + archival memory for a single agent session.

    Parameters
    ----------
    memory : Memory
        The underlying archival store (existing genome.Memory).
    user_id : str
        Scope id -- all archival writes/reads are scoped to this user.
    session_id : str
        Conversation identifier; stored as agent_id on archival records so
        multiple sessions for the same user stay separable.
    core_blocks : dict[str, CoreBlock]
        Named in-prompt blocks. Defaults: persona, user, scratch.

    Notes
    -----
    Each AgentMemory instance is scoped to a unique (user_id, session_id)
    pair via deterministic core-block ids. Reusing the SAME underlying
    Memory across many (user_id, session_id) pairs leaves orphaned core-block
    records for the previous sessions in the store. This is by design --
    scope isolation prevents cross-session reads -- but if you want to
    reclaim that storage you must call `Memory.reset(user_id=..., agent_id=...)`
    explicitly when a session ends. genome does not auto-evict.
    """
    memory: Memory
    user_id: str
    session_id: str = "default"
    core_blocks: dict[str, CoreBlock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.core_blocks:
            self.core_blocks = {
                "persona": CoreBlock(
                    label="persona",
                    description="The assistant's persistent persona / behavior.",
                    max_chars=1500,
                ),
                "user": CoreBlock(
                    label="user",
                    description="Facts the assistant has learned about the user.",
                    max_chars=2000,
                ),
                "scratch": CoreBlock(
                    label="scratch",
                    description="Working scratchpad the assistant can use.",
                    max_chars=1000,
                ),
            }
        # Try to load any persisted core blocks from archival
        self._load_core_from_archival()

    # ---------- core memory rendering ----------

    def render_core(self) -> str:
        """Render core blocks as a system-prompt-friendly string."""
        parts: list[str] = []
        for block in self.core_blocks.values():
            body = block.value if block.value else "(empty)"
            parts.append(f"## {block.label}\n{body}")
        return "\n\n".join(parts)

    def _core_persistent_id(self, label: str) -> str:
        """Deterministic id for a persisted core block so it can be overwritten."""
        return f"mem_core_{self.user_id}_{self.session_id}_{label}"

    def _persist_core_block(self, label: str) -> None:
        """Write a core block to archival so it survives restarts."""
        block = self.core_blocks[label]
        # We use metadata as the source of truth; content mirrors value for search.
        content = f"[{block.label}] {block.value}" if block.value else f"[{block.label}]"
        # Delete any existing persisted version (labels are unique per session)
        existing_id = self._core_persistent_id(label)
        self.memory.delete(existing_id, user_id=self.user_id, agent_id=self.session_id)
        # Re-persist
        import numpy as np
        vec = np.asarray(self.memory.embed.encode(content), dtype=np.float32)
        rec = MemoryRecord(
            id=existing_id,
            content=content,
            embedding=vec,
            user_id=self.user_id,
            agent_id=self.session_id,
            operator=CORE_MEMORY_OPERATOR,
            metadata={
                "core_label": block.label,
                "max_chars": block.max_chars,
                "description": block.description,
                "value": block.value,
            },
        )
        self.memory.store.add(rec)
        if hasattr(self.memory, "_scope_epochs"):
            self.memory._scope_epochs.bump(self.user_id, self.session_id)

    def _load_core_from_archival(self) -> None:
        """Load any previously-persisted core blocks for this session."""
        for rec in self.memory.store.list_by_scope(
            user_id=self.user_id, agent_id=self.session_id,
        ):
            if rec.operator != CORE_MEMORY_OPERATOR:
                continue
            label = rec.metadata.get("core_label")
            if label and label in self.core_blocks:
                self.core_blocks[label].value = rec.metadata.get("value", "")

    # ---------- tool handlers (called by handle_tool_call) ----------

    def core_append(self, label: str, text: str) -> dict[str, Any]:
        if label not in self.core_blocks:
            return {"error": f"unknown core label: {label}"}
        try:
            self.core_blocks[label].append(text)
        except ValueError as e:
            return {"error": str(e)}
        self._persist_core_block(label)
        return {"ok": True, "new_length": len(self.core_blocks[label].value)}

    def core_replace(self, label: str, old: str, new: str) -> dict[str, Any]:
        if label not in self.core_blocks:
            return {"error": f"unknown core label: {label}"}
        try:
            self.core_blocks[label].replace(old, new)
        except ValueError as e:
            return {"error": str(e)}
        self._persist_core_block(label)
        return {"ok": True, "new_length": len(self.core_blocks[label].value)}

    def archival_insert(self, content: str) -> dict[str, Any]:
        """Add a memory to archival (long-term) storage."""
        recs = self.memory.add(
            content, user_id=self.user_id, agent_id=self.session_id,
            metadata={"source": "agent_tool"},
        )
        return {"ok": True, "ids": [r.id for r in recs]}

    def archival_search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Search long-term memory. Core-memory records are filtered out."""
        results = self.memory.search(
            query,
            user_id=self.user_id,
            agent_id=self.session_id,
            limit=limit * 2,
            filter_parents=True,
        )
        filtered = [
            {"id": r.id, "content": r.content, "score": r.score}
            for r in results
            if r.record.operator != CORE_MEMORY_OPERATOR
        ][:limit]
        return {"results": filtered}

    def archival_delete(self, memory_id: str) -> dict[str, Any]:
        """Delete an archival memory by id."""
        ok = self.memory.delete(
            memory_id, user_id=self.user_id, agent_id=self.session_id,
        )
        return {"ok": ok}

    def synthesize_memories(
        self,
        memory_ids: list[str],
        operator: str = "uniform_crossover",
    ) -> dict[str, Any]:
        """Create a hybrid memory by recombining N existing archival memories."""
        try:
            hybrid = self.memory.synthesize(
                memory_ids=memory_ids,
                user_id=self.user_id,
                agent_id=self.session_id,
                operator=operator,
            )
        except ValueError as e:
            return {"error": str(e)}
        return {"ok": True, "id": hybrid.id, "content": hybrid.content}

    # ---------- dispatch ----------

    def handle_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call by name. Use this inside your agent loop
        when the LLM emits a tool call matching one of our schemas."""
        handler = {
            "core_append": lambda: self.core_append(args["label"], args["text"]),
            "core_replace": lambda: self.core_replace(
                args["label"], args["old"], args["new"],
            ),
            "archival_insert": lambda: self.archival_insert(args["content"]),
            "archival_search": lambda: self.archival_search(
                args["query"], int(args.get("limit", 5)),
            ),
            "archival_delete": lambda: self.archival_delete(args["memory_id"]),
            "synthesize_memories": lambda: self.synthesize_memories(
                args["memory_ids"], args.get("operator", "uniform_crossover"),
            ),
        }.get(name)
        if handler is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return handler()
        except KeyError as e:
            return {"error": f"missing required argument: {e}"}

    # ---------- context management ----------

    def archival_count(self) -> int:
        """Count of archival memories in this session."""
        return sum(
            1
            for r in self.memory.list_all(
                user_id=self.user_id, agent_id=self.session_id,
            )
            if r.operator not in {CORE_MEMORY_OPERATOR}
        )


__all__ = ["AgentMemory", "CoreBlock", "CORE_MEMORY_OPERATOR"]
