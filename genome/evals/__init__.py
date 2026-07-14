"""Evaluation harnesses for genome.

Ships with LOCOMO (LOng COnversation MemOry) — the standard conversational
memory benchmark published in "Evaluating Very Long-Term Conversational
Memory of LLM Agents" (Maharana et al., 2024; https://arxiv.org/abs/2402.17753).

Run via:
    python -m genome.evals.locomo --config all
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from genome.evals.locomo import (
    LocomoConfig,
    LocomoConversation,
    LocomoQuestion,
    LocomoResult,
    load_locomo,
    run_locomo_eval,
)

__all__ = [
    "LocomoConfig",
    "LocomoConversation",
    "LocomoQuestion",
    "LocomoResult",
    "load_locomo",
    "run_locomo_eval",
]
