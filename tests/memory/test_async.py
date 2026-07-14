import asyncio

from genome.memory.async_facade import AsyncMemory
from genome.memory.graph import RELATES_TO
from tests.memory._fake_embed import FakeEmbeddingProvider


def _run(coro):
    """Run a coroutine in a fresh event loop (pytest-asyncio-free)."""
    return asyncio.new_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_async_add_search_roundtrip():
    async def main():
        m = AsyncMemory(embedding_provider=FakeEmbeddingProvider(dim=8))
        try:
            recs = await m.add("user likes coffee", user_id="alice")
            assert len(recs) == 1
            results = await m.search("drinks", user_id="alice", limit=5)
            assert len(results) == 1
            assert results[0].content == "user likes coffee"
        finally:
            await m.close()
    _run(main())


def test_async_synthesize():
    async def main():
        m = AsyncMemory(embedding_provider=FakeEmbeddingProvider(dim=8))
        try:
            a = (await m.add("alpha", user_id="u"))[0]
            b = (await m.add("beta", user_id="u"))[0]
            hybrid = await m.synthesize(
                memory_ids=[a.id, b.id], user_id="u", operator="simple_average"
            )
            assert hybrid.is_synthesized
            assert len(hybrid.parents) == 2
        finally:
            await m.close()
    _run(main())


def test_async_graph():
    async def main():
        m = AsyncMemory(embedding_provider=FakeEmbeddingProvider(dim=8))
        try:
            a = (await m.add("a", user_id="u"))[0]
            b = (await m.add("b", user_id="u"))[0]
            edge = await m.link(a.id, b.id, relation=RELATES_TO)
            assert edge.from_id == a.id
            related = await m.related(a.id, relation=RELATES_TO)
            assert len(related) == 1
            assert related[0].id == b.id
            assert await m.unlink(edge.id) is True
            assert await m.related(a.id, relation=RELATES_TO) == []
        finally:
            await m.close()
    _run(main())


def test_async_llm_call_coroutine():
    async def my_async_llm(prompt: str) -> str:
        await asyncio.sleep(0)  # cooperative
        return "- user likes coffee\n- user lives in Tokyo"

    async def main():
        m = AsyncMemory(
            embedding_provider=FakeEmbeddingProvider(dim=8),
            llm_call=my_async_llm,
        )
        try:
            recs = await m.add("I love coffee and moved to Tokyo", user_id="alice")
            assert len(recs) == 2
            contents = {r.content for r in recs}
            assert contents == {"user likes coffee", "user lives in Tokyo"}
        finally:
            await m.close()
    _run(main())


def test_async_llm_call_sync_still_works():
    def sync_llm(prompt: str) -> str:
        return "- user likes tea"

    async def main():
        m = AsyncMemory(
            embedding_provider=FakeEmbeddingProvider(dim=8),
            llm_call=sync_llm,
        )
        try:
            recs = await m.add("I love tea", user_id="u")
            assert len(recs) == 1
            assert recs[0].content == "user likes tea"
        finally:
            await m.close()
    _run(main())


def test_async_context_manager():
    async def main():
        async with AsyncMemory(embedding_provider=FakeEmbeddingProvider(dim=8)) as m:
            await m.add("hello", user_id="u")
            assert await m.count(user_id="u") == 1
    _run(main())


def test_async_concurrent_ops_dont_deadlock():
    """Two concurrent adds should complete without stepping on each other."""
    async def main():
        m = AsyncMemory(embedding_provider=FakeEmbeddingProvider(dim=8))
        try:
            results = await asyncio.gather(
                m.add("fact 1", user_id="u"),
                m.add("fact 2", user_id="u"),
                m.add("fact 3", user_id="u"),
            )
            assert sum(len(r) for r in results) == 3
            assert await m.count(user_id="u") == 3
        finally:
            await m.close()
    _run(main())


def test_async_loop_runner_singleton():
    """The loop runner is a process-singleton; calls reuse the same loop."""
    from genome.memory.async_facade import _AsyncLoopRunner
    a = _AsyncLoopRunner.get()
    b = _AsyncLoopRunner.get()
    assert a is b
    # The background thread is alive and pinned to the same loop
    assert a._thread.is_alive()
    assert a._loop is b._loop


def test_async_loop_runner_runs_coroutines():
    """Schedule arbitrary coroutines onto the runner and get results back."""
    from genome.memory.async_facade import _AsyncLoopRunner

    async def add(x, y):
        await asyncio.sleep(0.001)
        return x + y

    runner = _AsyncLoopRunner.get()
    assert runner.run(add(2, 3)) == 5
    # Confirm reuse: second call should use the same loop, not spin a new one
    same_runner = _AsyncLoopRunner.get()
    assert same_runner is runner
    assert same_runner.run(add(10, 20)) == 30


def test_async_extractor_uses_singleton_runner_no_loop_per_call():
    """_AsyncAdaptingExtractor should NOT create a new event loop per extract.
    Verify by snapshotting the runner's loop id across multiple extract calls."""
    from genome.memory.async_facade import _AsyncAdaptingExtractor, _AsyncLoopRunner

    async def fake_llm(prompt):
        return "- fact 1\n- fact 2"

    extractor = _AsyncAdaptingExtractor(fake_llm)
    runner = _AsyncLoopRunner.get()
    loop_id_before = id(runner._loop)
    facts1 = extractor.extract("anything")
    facts2 = extractor.extract("anything else")
    loop_id_after = id(_AsyncLoopRunner.get()._loop)
    assert facts1 and facts2  # parsed something
    assert loop_id_before == loop_id_after  # loop reused, not recreated


def test_async_loop_runner_shutdown_idempotent():
    """shutdown() twice must not raise. Test re-creates the singleton after
    so we don't leave the rest of the suite without a loop."""
    from genome.memory.async_facade import _AsyncLoopRunner

    runner = _AsyncLoopRunner.get()
    runner.shutdown()
    # Second call: should be a no-op (loop already stopped + closed)
    runner.shutdown()
    # Singleton instance has been cleared; next get() must rebuild cleanly
    new_runner = _AsyncLoopRunner.get()
    assert new_runner is not runner
    # Verify the new runner works
    async def echo(x):
        return x
    assert new_runner.run(echo("ok")) == "ok"


def test_async_loop_runner_shutdown_releases_loop():
    """After shutdown, the loop must be closed and the singleton cleared."""
    from genome.memory.async_facade import _AsyncLoopRunner

    runner = _AsyncLoopRunner.get()
    loop = runner._loop
    runner.shutdown()
    assert loop.is_closed()
    # Singleton instance pointer must be cleared so a future get() rebuilds
    assert _AsyncLoopRunner._instance is None
    # Restore for downstream tests
    _AsyncLoopRunner.get()
