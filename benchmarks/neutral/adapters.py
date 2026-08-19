# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Adapters for the neutral memory-system harness.

Every adapter speaks the same three-method protocol the baselines already use:

    ingest(conversation) -> None      # feed a LocomoConversation into the system
    answer(question) -> (text, retrieved: list[str], latency_s)
    close() -> None

The point of this file is that adding a system is ~40 lines, and the harness
gives every system the SAME responder, judge, embedder tier, and question set.
If you ship a memory system and think this harness treats you unfairly, the fix
is a pull request, not a rebuttal thread.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

_WIRING = {
    "zep": (
        "Zep adapter is not bundled: Zep's OSS quickstart needs a running Zep "
        "server (docker) and the graphiti pipeline needs an LLM key for graph "
        "construction. Wire it in ~40 lines: subclass the protocol in this file, "
        "point it at your Zep instance, and map conversation turns to "
        "memory.add / graph search calls. See ADAPTERS.md for the exact recipe "
        "and the fairness rules (same responder, same judge, same embedder tier)."
    ),
    "letta": (
        "Letta adapter is not bundled: Letta is an agent runtime, so a fair "
        "mapping (its self-editing memory vs a passive store) needs a decision "
        "about tool budgets that should not be made silently. ADAPTERS.md "
        "documents the mapping we propose; implement it in this file's protocol "
        "and the harness treats it identically to every other system."
    ),
}


def make_system(
    name: str, responder: Callable[[str], str], *, top_k: int, provider: str, model: str
) -> Any:
    """Build a baseline-protocol system by registry name.

    'genome' is intentionally NOT here: the harness runs GENOME through the same
    published eval path (genome.evals.locomo) used for every public number, so
    a curious reader can diff the harness against the paper's pipeline.
    """
    if name == "mem0":
        from head_to_head import Mem0Baseline, _Mem0LocalBaseline

        if provider == "openrouter":
            return _Mem0LocalBaseline(responder, top_k=top_k, llm_model=model)
        return Mem0Baseline(
            responder,
            top_k=top_k,
            llm_model=model,
            embed_model="text-embedding-3-small",
            llm_provider=provider,
        )
    if name == "fullcontext":
        from genome.evals.baselines import FullContextBaseline

        return FullContextBaseline(responder)
    if name in _WIRING:
        raise SystemExit(f"{name}: {_WIRING[name]}")
    raise SystemExit(
        f"unknown system {name!r}. Bundled: genome, mem0, fullcontext. "
        f"Documented wiring: {', '.join(sorted(_WIRING))}."
    )
