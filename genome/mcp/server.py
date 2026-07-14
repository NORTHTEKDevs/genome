# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""GENOME MCP server -- fully-local, zero-LLM agent memory over the Model Context
Protocol.

Every tool runs entirely on the local machine: memories are embedded with a local
model and stored in a local SQLite file. No LLM calls, no API keys, no network, no
data leaves the machine. Add it to any MCP client (Claude Desktop, Claude Code,
Cursor, ...) and the agent gets persistent memory across sessions.

Storage: `$GENOME_MCP_DB` if set, else `~/.genome/memories.db` (created on first use).
Run:     `genome-mcp`  (stdio transport)  or  `python -m genome.mcp.server`
"""
from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from genome import Memory

mcp = FastMCP("genome-memory")

_mem: Memory | None = None


def _db_path() -> str:
    p = os.environ.get("GENOME_MCP_DB")
    if p:
        return p
    home = Path.home() / ".genome"
    home.mkdir(parents=True, exist_ok=True)
    return str(home / "memories.db")


def memory() -> Memory:
    """Lazily open the persistent local store (local embedder, no network)."""
    global _mem
    if _mem is None:
        _mem = Memory(storage=_db_path())
    return _mem


@mcp.tool()
def remember(content: str, user_id: str = "default") -> str:
    """Store a fact or note in long-term memory.

    Fully local -- embeds the text with a local model and writes it to a local
    SQLite file. No LLM call, no network. Use this whenever the user tells you
    something worth remembering across sessions (preferences, facts, decisions).

    Args:
        content: The text to remember (a fact, preference, or note).
        user_id: Namespace to store under (default "default"). Use per-end-user.
    """
    content = (content or "").strip()
    if not content:
        return "Nothing to remember: `content` was empty."
    memory().add(content, user_id=user_id)
    return f"Remembered (user={user_id}): {content}"


@mcp.tool()
def recall(query: str, limit: int = 5, user_id: str = "default") -> str:
    """Search long-term memory for information relevant to a query.

    Read-only, fully local semantic search. Call this before answering when the
    user refers to past context, preferences, or previously-shared facts.

    Args:
        query: What to look for (a question or topic).
        limit: Max memories to return (1-50, default 5).
        user_id: Namespace to search (default "default").
    """
    query = (query or "").strip()
    if not query:
        return "Provide a non-empty `query` to search memory."
    limit = max(1, min(int(limit), 50))
    hits = memory().search(query, user_id=user_id, limit=limit)
    if not hits:
        return f"No relevant memories found for '{query}' (user={user_id})."
    return "\n".join(f"- {h.content}  (relevance {h.score:.2f})" for h in hits)


@mcp.tool()
def forget(query: str, user_id: str = "default") -> str:
    """Delete the single memory most relevant to `query`.

    Destructive. Finds the best-matching memory and removes it. If nothing
    matches, nothing is deleted.

    Args:
        query: Describes the memory to remove.
        user_id: Namespace to delete from (default "default").
    """
    query = (query or "").strip()
    if not query:
        return "Provide a non-empty `query` describing what to forget."
    hits = memory().search(query, user_id=user_id, limit=1)
    if not hits:
        return f"Nothing to forget: no memory matched '{query}' (user={user_id})."
    top = hits[0]
    ok = memory().delete(top.id, user_id=user_id)
    return f"Forgot: {top.content}" if ok else f"Could not delete the matched memory ({top.id})."


@mcp.tool()
def reset_memories(user_id: str = "default") -> str:
    """Delete ALL memories for a user. Destructive and irreversible.

    Args:
        user_id: Namespace to clear (default "default"). This never clears other
            users' namespaces.
    """
    n = memory().reset(user_id=user_id)
    return f"Cleared {n} memories for user '{user_id}'."


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
