"""Async wrapper over Memory.

Why: most modern LLM agent stacks (LangChain LCEL, LlamaIndex async, LiveKit
agents, Anthropic SDK async, OpenAI async) operate in asyncio. Forcing them
to block on sync Memory calls introduces head-of-line latency on every
`m.add` or `m.search`.

Implementation: delegate to `asyncio.to_thread` for the sync store ops (SQLite
is fast locally; no point rewriting to aiosqlite for v0.4). Accept either sync
or async `llm_call` for LLMExtractor -- if async, we await it directly.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from genome.embeddings import EmbeddingProvider
from genome.memory.extraction import FactExtractor, IdentityExtractor, LLMExtractor
from genome.memory.facade import Memory
from genome.memory.graph import MemoryEdge
from genome.memory.schema import MemoryRecord, SearchResult
from genome.memory.store import MemoryStore

AsyncLLMCallFn = Callable[[str], Awaitable[str]]
MaybeAsyncLLMCallFn = Callable[[str], str] | AsyncLLMCallFn


class _AsyncLoopRunner:
    """Process-singleton background event loop for sync->async bridging.

    Runs one asyncio event loop on a dedicated daemon thread for the lifetime
    of the process. `run(coro)` schedules a coroutine onto it via
    `run_coroutine_threadsafe` and blocks until it returns.

    Replaces the prior `asyncio.new_event_loop()` per-call pattern, which:
      - paid loop construction + transport setup on every call,
      - prevented connection pools / aiohttp sessions from being reused across
        calls (each new loop = fresh client state).
    """

    _instance: _AsyncLoopRunner | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="genome-async-bridge",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def get(cls) -> _AsyncLoopRunner:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def run(self, coro: Awaitable[Any]) -> Any:
        """Schedule `coro` onto the background loop and block on the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def shutdown(self) -> None:
        """Stop the background loop. Mostly for tests; daemon thread is auto-
        reaped at process exit otherwise."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)
        if not self._loop.is_closed():
            self._loop.close()
        with type(self)._instance_lock:
            if type(self)._instance is self:
                type(self)._instance = None


class _AsyncAdaptingExtractor:
    """Wraps an async LLM function as a sync-looking FactExtractor.

    Internal use: AsyncMemory builds one of these from an async llm_call so that
    the underlying sync Memory.add() can still invoke it from a worker thread
    via the singleton `_AsyncLoopRunner`.
    """

    def __init__(self, async_llm: AsyncLLMCallFn) -> None:
        self._async_llm = async_llm

    def extract(self, text: str) -> list[str]:
        runner = _AsyncLoopRunner.get()
        response = runner.run(self._async_llm(self._build_prompt(text)))
        from genome.memory.extraction import _parse_facts
        return _parse_facts(response)

    @staticmethod
    def _build_prompt(text: str) -> str:
        from genome.memory.extraction import FACT_EXTRACTION_PROMPT
        return FACT_EXTRACTION_PROMPT.format(text=text)


def _is_coroutine_fn(fn: Any) -> bool:
    return asyncio.iscoroutinefunction(fn)


class AsyncMemory:
    """Async-compatible Memory API.

    Same methods and semantics as `Memory`, but everything is a coroutine. Accepts
    either a sync or async `llm_call` for fact extraction.

    Usage::

        import asyncio
        from genome import AsyncMemory

        async def main():
            async def my_async_claude(prompt: str) -> str:
                ...  # your AsyncAnthropic call

            m = AsyncMemory(llm_call=my_async_claude)
            await m.add("I love coffee", user_id="alice")
            results = await m.search("drinks?", user_id="alice")
            await m.close()

        asyncio.run(main())
    """

    def __init__(
        self,
        *,
        storage: str | Path | MemoryStore = ":memory:",
        embedding_provider: EmbeddingProvider | None = None,
        llm_call: MaybeAsyncLLMCallFn | None = None,
        extractor: FactExtractor | None = None,
    ) -> None:
        # Resolve extractor
        resolved_extractor: FactExtractor
        if extractor is not None:
            resolved_extractor = extractor
        elif llm_call is not None:
            if _is_coroutine_fn(llm_call):
                resolved_extractor = _AsyncAdaptingExtractor(llm_call)  # type: ignore[arg-type]
            else:
                resolved_extractor = LLMExtractor(llm_call)  # type: ignore[arg-type]
        else:
            resolved_extractor = IdentityExtractor()

        # Underlying sync Memory
        self._sync = Memory(
            storage=storage,
            embedding_provider=embedding_provider,
            extractor=resolved_extractor,
        )

    # ---------- core ops (async) ----------

    async def add(
        self,
        text: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict | None = None,
    ) -> list[MemoryRecord]:
        return await asyncio.to_thread(
            self._sync.add,
            text,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata,
        )

    async def get(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> MemoryRecord | None:
        return await asyncio.to_thread(
            self._sync.get, memory_id, user_id=user_id, agent_id=agent_id,
        )

    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict | None = None,
        re_embed: bool = True,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> MemoryRecord | None:
        return await asyncio.to_thread(
            self._sync.update,
            memory_id,
            content=content,
            metadata=metadata,
            re_embed=re_embed,
            user_id=user_id,
            agent_id=agent_id,
        )

    async def delete(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._sync.delete, memory_id, user_id=user_id, agent_id=agent_id,
        )

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        filter_parents: bool = True,
        exclude_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(
            self._sync.search,
            query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            filter_parents=filter_parents,
            exclude_ids=exclude_ids,
        )

    async def synthesize(
        self,
        memory_ids: list[str],
        *,
        operator: str = "uniform_crossover",
        user_id: str | None = None,
        agent_id: str | None = None,
        content: str | None = None,
        metadata: dict | None = None,
        **operator_kwargs: Any,
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self._sync.synthesize,
            memory_ids,
            operator=operator,
            user_id=user_id,
            agent_id=agent_id,
            content=content,
            metadata=metadata,
            **operator_kwargs,
        )

    # ---------- graph ops (async) ----------

    async def link(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        *,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> MemoryEdge:
        return await asyncio.to_thread(
            self._sync.link, from_id, to_id, relation,
            weight=weight, metadata=metadata,
        )

    async def unlink(self, edge_id: str) -> bool:
        return await asyncio.to_thread(self._sync.unlink, edge_id)

    async def related(
        self,
        memory_id: str,
        relation: str | None = None,
        *,
        direction: str = "out",
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryRecord]:
        return await asyncio.to_thread(
            self._sync.related, memory_id, relation,
            direction=direction, user_id=user_id, agent_id=agent_id,
        )

    # ---------- housekeeping (async) ----------

    async def count(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> int:
        return await asyncio.to_thread(
            self._sync.count, user_id=user_id, agent_id=agent_id
        )

    async def list_all(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> list[MemoryRecord]:
        return await asyncio.to_thread(
            self._sync.list_all, user_id=user_id, agent_id=agent_id
        )

    async def reset(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> int:
        return await asyncio.to_thread(
            self._sync.reset, user_id=user_id, agent_id=agent_id
        )

    async def consolidate(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        max_memories: int = 500,
        half_life_days: float = 30.0,
        synthesize_before_prune: bool = False,
        synthesis_operator: str = "frequency_crossover",
    ):
        return await asyncio.to_thread(
            self._sync.consolidate,
            user_id=user_id,
            agent_id=agent_id,
            max_memories=max_memories,
            half_life_days=half_life_days,
            synthesize_before_prune=synthesize_before_prune,
            synthesis_operator=synthesis_operator,
        )

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)

    # Async context manager
    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
