# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""The MCP server must register its tools through the protocol layer on both
supported SDK lines (mcp 1.x ``FastMCP`` and mcp 2.x ``MCPServer``).

The other MCP tests call the tool functions directly, which passes even when
nothing is exposed to a client. This asks the server what a client would see,
including the serverInfo it reports when a client connects.
"""

from __future__ import annotations

import asyncio
import importlib.metadata

import pytest

import genome
from genome.mcp import server

_MCP_MAJOR = int(importlib.metadata.version("mcp").split(".")[0])
# An mcp 2.x client either runs the classic initialize handshake ("legacy") or
# tries the newer server/discover path first ("auto"); 1.x has only the former.
_MODES = ["legacy", "auto"] if _MCP_MAJOR >= 2 else ["legacy"]

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


async def _server_info(mcp_server, mode):
    """serverInfo as a client receives it, over the SDK's in-process transport."""
    if _MCP_MAJOR >= 2:
        from mcp import Client

        async with Client(mcp_server, mode=mode) as client:
            return client.server_info
    from mcp.shared.memory import create_connected_server_and_client_session

    # The 1.x helper takes the lowlevel server (1.2-1.16 accept nothing else) and
    # runs initialize itself without keeping the result, so initialize again.
    async with create_connected_server_and_client_session(mcp_server._mcp_server) as session:
        return (await session.initialize()).serverInfo


@pytest.mark.parametrize("mode", _MODES)
def test_clients_see_genomes_version_not_the_sdks(mode):
    # Before 1.3.1 this was mcp's own version on 1.x and an empty string on 2.x.
    info = asyncio.run(_server_info(server.mcp, mode))
    assert info is not None, "the handshake carried no serverInfo"
    assert info.name == "genome-memory"
    assert info.version == genome.__version__


@pytest.mark.parametrize("mode", _MODES)
def test_the_reported_version_is_the_one_the_server_was_built_with(mode):
    # A value no SDK release has, so this cannot pass by coincidence.
    info = asyncio.run(_server_info(server._new_server("0.0.0.dev0+sentinel"), mode))
    assert info is not None, "the handshake carried no serverInfo"
    assert info.version == "0.0.0.dev0+sentinel"
