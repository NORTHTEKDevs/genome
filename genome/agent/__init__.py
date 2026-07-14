"""Agent runtime for genome.

Letta-style hierarchical memory (core + archival) with tool-callable
interfaces. Works with any LLM SDK (Anthropic, OpenAI, any tool-calling
model) by exposing tool schemas the agent can invoke.

Usage::

    from genome import Memory
    from genome.agent import AgentMemory, tool_schemas

    mem = Memory(storage="agent.db")
    agent_mem = AgentMemory(memory=mem, user_id="alice", session_id="s1")

    # Render the core memory block into your system prompt
    system_prompt = f"You are a helpful assistant.\\n\\n{agent_mem.render_core()}"

    # Advertise the memory tools to your LLM
    tools = tool_schemas()  # Anthropic format

    # When the LLM calls a tool:
    result = agent_mem.handle_tool_call(tool_name, tool_args)
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from genome.agent.memory import AgentMemory, CoreBlock
from genome.agent.tools import (
    ANTHROPIC_TOOL_SCHEMAS,
    OPENAI_TOOL_SCHEMAS,
    tool_schemas,
)

__all__ = [
    "AgentMemory",
    "CoreBlock",
    "tool_schemas",
    "ANTHROPIC_TOOL_SCHEMAS",
    "OPENAI_TOOL_SCHEMAS",
]
