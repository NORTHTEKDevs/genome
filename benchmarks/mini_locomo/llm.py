"""Sync LLM wrapper for the mini-LoCoMo benchmark.

Uses the standard `anthropic` SDK with an API key read from the
ANTHROPIC_API_KEY environment variable. Single Sonnet 4.6 call per request,
~1-3 second wall time per call.

The legacy MaxLLM class (claude-agent-sdk via Max OAuth) was removed because
the SDK spawns a fresh ~1GB claude.exe subprocess per query, making it
unusable for benchmark workloads with more than a handful of calls.
"""

from __future__ import annotations

import os
import time

from anthropic import Anthropic


class ApiKeyLLM:
    """Sync callable wrapping anthropic.Anthropic.messages.create.

    Implements GENOME's LLMCallFn protocol: takes a prompt str, returns a str.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5-20250929",
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY env var is required for ApiKeyLLM. "
                "Set it in your shell before running the benchmark."
            )
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.system = system or (
            "You are a precise, terse text generator for a memory-system "
            "benchmark. Follow the user's instructions exactly. Do not add "
            "explanation, preamble, or commentary unless explicitly asked."
        )
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_seconds = 0.0

    def __call__(self, prompt: str) -> str:
        t0 = time.time()
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": prompt}],
        )
        self.calls += 1
        self.input_tokens += msg.usage.input_tokens
        self.output_tokens += msg.usage.output_tokens
        self.total_seconds += time.time() - t0
        # Extract concatenated text from content blocks
        text_parts = [
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        ]
        return "".join(text_parts).strip()

    def close(self) -> None:
        # Anthropic client has no explicit close
        pass

    @property
    def cost_estimate_usd(self) -> float:
        """Sonnet 4.5 pricing as of 2026-04: $3/MTok input, $15/MTok output."""
        return (
            self.input_tokens / 1_000_000 * 3.0
            + self.output_tokens / 1_000_000 * 15.0
        )
