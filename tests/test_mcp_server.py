# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""The MCP server must register its tools through the protocol layer on both
supported SDK lines (mcp 1.x ``FastMCP`` and mcp 2.x ``MCPServer``).

The other MCP tests call the tool functions directly, which passes even when
nothing is exposed to a client. This asks the server what a client would see.
"""

from __future__ import annotations

import asyncio

from genome.mcp import server

EXPECTED = {
    "remember": {"content", "user_id"},
    "recall": {"query", "limit", "user_id"},
    "forget": {"query", "user_id", "min_score"},
    "reset_memories": {"user_id"},
}


def _tools():
    return asyncio.run(server.mcp.list_tools())


def _schema(tool):
    # mcp 1.x names the field inputSchema; 2.x renamed it input_schema
    return getattr(tool, "inputSchema", None) or tool.input_schema


def test_exactly_the_four_tools_are_registered():
    assert {t.name for t in _tools()} == set(EXPECTED)


def test_tool_argument_schemas():
    for t in _tools():
        assert set(_schema(t)["properties"]) == EXPECTED[t.name], t.name
        assert t.description, f"{t.name} has no description for the client"


def test_destructive_tools_require_their_target():
    required = {t.name: set(_schema(t).get("required", [])) for t in _tools()}
    assert required["forget"] == {"query"}
    assert required["remember"] == {"content"}
    assert required["reset_memories"] == set()


def test_server_name():
    assert server.mcp.name == "genome-memory"
