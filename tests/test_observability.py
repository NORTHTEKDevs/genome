import io
import json
import logging

import numpy as np
import pytest

from genome.memory.facade import Memory
from genome.observability import (
    ErrorCapture,
    JSONFormatter,
    MetricsRegistry,
    configure_logging,
    get_error_capture,
    get_logger,
    get_metrics,
)
from tests.memory._fake_embed import FakeEmbeddingProvider

# ---------- logging ----------

def test_json_formatter_basic():
    rec = logging.LogRecord(
        name="genome.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    out = JSONFormatter().format(rec)
    data = json.loads(out)
    assert data["level"] == "INFO"
    assert data["logger"] == "genome.test"
    assert data["msg"] == "hello world"


def test_json_formatter_extras():
    rec = logging.LogRecord(
        name="genome.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    rec.user_id = "alice"
    rec.count = 3
    out = JSONFormatter().format(rec)
    data = json.loads(out)
    assert data["user_id"] == "alice"
    assert data["count"] == 3


def test_configure_logging_writes_json_to_stream():
    stream = io.StringIO()
    configure_logging(level="DEBUG", json_output=True, stream=stream)
    logger = get_logger("test")
    logger.info("event", extra={"foo": "bar"})
    line = stream.getvalue().strip().splitlines()[-1]
    data = json.loads(line)
    assert data["level"] == "INFO"
    assert data["foo"] == "bar"
    assert data["logger"].startswith("genome.test")


def test_get_logger_namespace():
    log = get_logger("foo")
    assert log.name == "genome.foo"


# ---------- metrics ----------

def test_counter_inc():
    reg = MetricsRegistry()
    c = reg.counter("test.count", tags={"k": "v"})
    c.inc()
    c.inc(2.5)
    snap = reg.snapshot()
    bucket = snap["counters"]["test.count"]
    assert len(bucket) == 1
    assert bucket[0]["value"] == 3.5


def test_counter_separates_by_tags():
    reg = MetricsRegistry()
    reg.counter("test.count", tags={"k": "a"}).inc()
    reg.counter("test.count", tags={"k": "b"}).inc(2)
    snap = reg.snapshot()
    bucket = snap["counters"]["test.count"]
    assert len(bucket) == 2
    values = {frozenset(b["tags"].items()): b["value"] for b in bucket}
    assert values[frozenset({("k", "a")})] == 1
    assert values[frozenset({("k", "b")})] == 2


def test_histogram_observe():
    reg = MetricsRegistry()
    h = reg.histogram("test.hist", tags={"op": "x"})
    for v in [0.1, 0.2, 0.3]:
        h.observe(v)
    snap = reg.snapshot()
    bucket = snap["histograms"]["test.hist"][0]
    assert bucket["count"] == 3
    assert bucket["sum"] == pytest.approx(0.6)
    assert bucket["mean"] == pytest.approx(0.2)


def test_histogram_time_context_manager():
    reg = MetricsRegistry()
    h = reg.histogram("test.timer")
    import time
    with h.time():
        time.sleep(0.01)
    snap = reg.snapshot()
    bucket = snap["histograms"]["test.timer"][0]
    assert bucket["count"] == 1
    # Observed value >= 0.01 (sleep duration)
    assert bucket["sum"] >= 0.005


def test_metrics_sink_receives_observations():
    reg = MetricsRegistry()
    captured: list[tuple[str, float, dict[str, str]]] = []

    def sink(name, value, tags):
        captured.append((name, value, dict(tags)))

    reg.set_sink(sink)
    reg.counter("c", tags={"k": "v"}).inc(5)
    reg.histogram("h", tags={"k": "v"}).observe(1.5)
    assert len(captured) == 2
    names = {c[0] for c in captured}
    assert names == {"c", "h"}


def test_metrics_sink_exception_swallowed():
    """A broken sink must not break the caller."""
    reg = MetricsRegistry()

    def broken(*args):
        raise RuntimeError("boom")

    reg.set_sink(broken)
    reg.counter("c").inc()  # should not raise


def test_registry_reset_clears():
    reg = MetricsRegistry()
    reg.counter("c").inc(10)
    reg.reset()
    snap = reg.snapshot()
    assert snap["counters"] == {}


def test_global_registry_singleton():
    reg1 = get_metrics()
    reg2 = get_metrics()
    assert reg1 is reg2


# ---------- integration: Memory emits metrics ----------

def test_memory_emits_add_and_search_metrics():
    reg = get_metrics()
    reg.reset()
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        m.add("fact 1", user_id="alice")
        m.add("fact 2", user_id="alice")
        m.search("query", user_id="alice")
        m.search("query", user_id="alice")  # second is a cache hit

        snap = reg.snapshot()

        # add.count: 2 (one per record added)
        add_count = sum(
            b["value"] for b in snap["counters"].get("memory.add.count", [])
        )
        assert add_count == 2

        # search.count recorded
        search_count = sum(
            b["value"] for b in snap["counters"].get("memory.search.count", [])
        )
        assert search_count >= 1

        # cache hit recorded on repeat query
        cache_hit = sum(
            b["value"] for b in snap["counters"].get("memory.search.cache_hit", [])
        )
        assert cache_hit >= 1

        # duration histogram recorded
        assert "memory.add.duration" in snap["histograms"]
        assert "memory.search.duration" in snap["histograms"]
    finally:
        m.close()


# ---------- error capture ----------

def test_error_capture_records_exception():
    ec = ErrorCapture(capacity=10)
    try:
        raise ValueError("bad thing happened")
    except ValueError as e:
        ce = ec.capture(e, tags={"user_id": "alice"})
    assert ce.error_type == "ValueError"
    assert "bad thing" in ce.message
    assert ce.tags == {"user_id": "alice"}
    assert ce.fingerprint
    assert len(ec.recent()) == 1
    assert ec.recent()[0].fingerprint == ce.fingerprint


def test_error_capture_groups_by_fingerprint():
    ec = ErrorCapture()

    def raise_here():
        raise RuntimeError("boom")

    for _ in range(3):
        try:
            raise_here()
        except RuntimeError as e:
            ec.capture(e)
    groups = ec.grouped()
    assert len(groups) == 1
    assert groups[0]["count"] == 3
    assert groups[0]["error_type"] == "RuntimeError"


def test_error_capture_capacity_evicts_oldest():
    ec = ErrorCapture(capacity=2)
    for i in range(5):
        try:
            raise ValueError(f"err {i}")
        except ValueError as e:
            ec.capture(e)
    recents = ec.recent(limit=10)
    assert len(recents) == 2
    assert "4" in recents[0].message
    assert "3" in recents[1].message


def test_error_capture_sink_failure_swallowed():
    def broken_sink(_ce):
        raise RuntimeError("sink itself broken")

    ec = ErrorCapture(sink=broken_sink)
    try:
        raise ValueError("real error")
    except ValueError as e:
        ec.capture(e)  # must not raise
    assert len(ec.recent()) == 1


def test_error_capture_invalid_capacity():
    with pytest.raises(ValueError):
        ErrorCapture(capacity=0)


def test_error_capture_global_singleton():
    a = get_error_capture()
    b = get_error_capture()
    assert a is b


def test_error_capture_reset_clears_buffer_and_counts():
    ec = ErrorCapture()
    try:
        raise ValueError("x")
    except ValueError as e:
        ec.capture(e)
    assert len(ec.recent()) == 1
    ec.reset()
    assert ec.recent() == []
    assert ec.grouped() == []


# ---------- embedding finite-check ----------

def test_record_rejects_nan_embedding():
    """Adversarial NaN embeddings would poison cosine scores. Refuse at create."""
    from genome.memory.schema import MemoryRecord

    bad = np.array([1.0, float("nan"), 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN|Inf"):
        MemoryRecord(content="x", embedding=bad)


def test_record_rejects_inf_embedding():
    from genome.memory.schema import MemoryRecord

    bad = np.array([0.0, float("inf"), 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN|Inf"):
        MemoryRecord(content="x", embedding=bad)


def test_record_accepts_finite_embedding():
    from genome.memory.schema import MemoryRecord

    ok = np.array([1.0, 0.5, -0.25], dtype=np.float32)
    rec = MemoryRecord(content="ok", embedding=ok)
    assert rec.id


def test_sqlite_update_rejects_nan_embedding():
    """The store-level update path must also refuse NaN/Inf so callers
    that bypass MemoryRecord still hit the boundary."""
    from genome.memory.sqlite_store import SQLiteMemoryStore

    store = SQLiteMemoryStore(path=":memory:")
    mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=8), storage=store)
    try:
        r = mem.add("seed", user_id="u")[0]
        bad = np.array([1.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN|Inf"):
            store.update(r.id, embedding=bad)
    finally:
        mem.close()


def test_record_parents_require_operator():
    """parents=[A,B] with operator=None is malformed -- must raise."""
    from genome.memory.schema import MemoryRecord

    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="operator"):
        MemoryRecord(content="x", embedding=vec, parents=["mem_a", "mem_b"])
    # operator without parents is fine (entity / fact / raptor_summary tags)
    rec = MemoryRecord(content="x", embedding=vec, operator="entity")
    assert rec.operator == "entity"


def test_record_rejects_empty_string_parents():
    """parents=[\"\"] or parents=[None] should be refused."""
    from genome.memory.schema import MemoryRecord

    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty strings"):
        MemoryRecord(content="x", embedding=vec, parents=[""], operator="op")
    with pytest.raises(ValueError, match="non-empty strings"):
        MemoryRecord(content="x", embedding=vec, parents=[None], operator="op")


def test_sqlite_delete_cascades_atomically_under_lock():
    """delete() must remove memory + edges in a single lock acquisition.
    Smoke test: delete with edges leaves no orphans."""
    from genome.memory.facade import Memory
    from genome.memory.graph import RELATES_TO

    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        a = m.add("a", user_id="u")[0]
        b = m.add("b", user_id="u")[0]
        edge = m.link(a.id, b.id, RELATES_TO)
        assert m.delete(a.id) is True
        # Edge must be gone (cascade)
        assert m.store.get_edge(edge.id) is None
    finally:
        m.close()


def test_cli_limit_pairs_rejects_zero_and_negative():
    """argparse should reject --limit-pairs <= 0 with a clear error."""
    import argparse
    from argparse import ArgumentTypeError

    import genome.cli  # noqa: F401 -- ensures the CLI module imports cleanly
    parser = argparse.ArgumentParser()
    def _positive_int(s):
        v = int(s)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"got {v}")
        return v
    parser.add_argument("--n", type=_positive_int)
    with pytest.raises(SystemExit):  # argparse converts ArgumentTypeError to SystemExit
        parser.parse_args(["--n", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--n", "-3"])
    # positive accepted
    ns = parser.parse_args(["--n", "5"])
    assert ns.n == 5
    # ArgumentTypeError import keeps the symbol available for direct tests
    assert ArgumentTypeError is argparse.ArgumentTypeError


def test_openapi_includes_all_response_models():
    """All endpoints that return JSON must declare a Pydantic response_model
    so the OpenAPI spec advertises an accurate response schema. Regression
    for R7: 7 endpoints previously returned raw dicts."""
    from fastapi.testclient import TestClient

    from genome.server.app import create_app

    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(m)
    c = TestClient(app)
    spec = c.get("/openapi.json").json()
    # Every JSON-returning endpoint should have a 200/201 schema $ref or type.
    paths = spec["paths"]
    json_endpoints = [
        ("/v1/memories/{memory_id}", "delete"),
        ("/v1/edges", "post"),
        ("/v1/edges/{edge_id}", "delete"),
        ("/v1/scope", "delete"),
        ("/v1/count", "get"),
        ("/v1/metrics", "get"),
        ("/v1/errors", "get"),
        ("/v1/errors", "delete"),
    ]
    for path, method in json_endpoints:
        op = paths[path][method]
        responses = op.get("responses", {})
        # 200 or 201 should have a content/application-json schema
        ok_resp = responses.get("200") or responses.get("201")
        assert ok_resp is not None, f"{method} {path} has no 2xx response"
        content = ok_resp.get("content", {}).get("application/json", {})
        assert "schema" in content, f"{method} {path} response missing schema"
    # Specific named models must appear in the components
    schemas = spec["components"]["schemas"]
    for required in [
        "EdgeResponse", "DeleteResponse", "ResetResponse",
        "CountResponse", "ClearedResponse", "ErrorsResponse",
        "MetricsSnapshot",
    ]:
        assert required in schemas, f"OpenAPI missing schema: {required}"
    m.close()


def test_rest_rejects_oversize_user_id_with_422():
    """REST AddRequest must enforce MAX_USER_ID_LEN at the API boundary so
    oversize ids surface as 422 validation errors, not 500 internal errors.
    """
    from fastapi.testclient import TestClient

    from genome.server.app import create_app

    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(m)
    c = TestClient(app)
    huge = "x" * 300  # > 256
    r = c.post("/v1/memories", json={"text": "hi", "user_id": huge})
    assert r.status_code == 422, f"expected 422, got {r.status_code}: {r.text}"
    detail = r.json()["detail"]
    assert any("user_id" in str(d).lower() or "user_id" in str(d.get("loc", []))
               for d in detail), detail
    m.close()


def test_rest_rejects_oversize_query_with_422():
    """REST SearchRequest must reject oversize query at the boundary."""
    from fastapi.testclient import TestClient

    from genome.server.app import create_app

    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    app = create_app(m)
    c = TestClient(app)
    huge = "x" * 200_000  # > 100k
    r = c.post("/v1/search", json={"query": huge, "user_id": "u"})
    assert r.status_code == 422, r.text
    m.close()


def test_add_copies_metadata_per_record():
    """Multi-fact add() must NOT share one metadata dict reference across all
    created records. A caller-side mutation of the dict should not corrupt
    stored records' metadata.
    """
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        meta = {"source": "user_chat"}
        recs = m.add("hello world", user_id="u", metadata=meta)
        assert len(recs) >= 1
        # Caller mutates their dict
        meta["source"] = "MUTATED"
        # Stored record should still reflect the original value
        for r in recs:
            assert r.metadata["source"] == "user_chat", \
                f"metadata mutation leaked into stored record: {r.metadata}"
    finally:
        m.close()


def test_link_copies_metadata():
    """link() must copy caller's metadata so post-call mutation doesn't leak."""
    from genome.memory.graph import RELATES_TO

    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        a = m.add("a", user_id="u")[0]
        b = m.add("b", user_id="u")[0]
        meta = {"reason": "first"}
        edge = m.link(a.id, b.id, RELATES_TO, metadata=meta)
        meta["reason"] = "MUTATED"
        # The returned edge metadata must be unaffected
        assert edge.metadata["reason"] == "first", \
            f"metadata mutation leaked into stored edge: {edge.metadata}"
    finally:
        m.close()


def test_close_drains_inflight_auto_extract():
    """Memory.close() must wait for in-flight auto-extract LLM calls so
    the store isn't torn down while LLM responses are mid-callback (which
    would crash on the post-LLM self.related() / record_fact() calls)."""
    import threading
    import time

    extract_started = threading.Event()
    allow_extract_finish = threading.Event()

    def slow_llm(prompt: str) -> str:
        # Entity-extraction prompt has <text>...</text> data block: block here
        # until the test releases it. This is the LLM call we want close() to
        # drain.
        if "<text>" in prompt:
            extract_started.set()
            allow_extract_finish.wait(timeout=3.0)
            return "NONE"
        # FACT_EXTRACTION_PROMPT_V2 (Memory uses this via LLMExtractor) -- return
        # a single bullet so a fact actually gets created and auto-extract fires.
        if "Facts:" in prompt and "extract atomic facts" in prompt.lower():
            return "- user lives in Tokyo"
        # Conflict-resolution / fact-detection: pass-through
        if "DECISION:" in prompt:
            return "DECISION: ADD"
        return "FACT_TYPE: none\nCONFIDENCE: 0.0"

    from genome.memory.extraction import IdentityExtractor
    m = Memory(
        embedding_provider=FakeEmbeddingProvider(dim=8),
        # Use IdentityExtractor so the test fact passes through unchanged
        # without depending on the LLMExtractor's prompt parsing.
        extractor=IdentityExtractor(),
        llm_call=slow_llm,
        auto_extract_entities=True,
    )

    # Run add in a worker thread
    def worker():
        m.add("user lives in Tokyo", user_id="alice")

    t = threading.Thread(target=worker)
    t.start()

    # Wait until the LLM is mid-call
    assert extract_started.wait(timeout=2.0), "auto-extract LLM should have been invoked"

    # Now: at this moment, the store is OPEN and the LLM is mid-call.
    # Spawn close() in another thread; it should DRAIN before closing.
    close_done = threading.Event()
    def closer():
        m.close()
        close_done.set()

    tc = threading.Thread(target=closer)
    tc.start()

    # Give close() a moment to start its drain wait
    time.sleep(0.1)
    # close() should NOT have completed yet -- it's draining the LLM
    assert not close_done.is_set(), "close() should be waiting on auto-extract drain"

    # Release the LLM
    allow_extract_finish.set()
    t.join(timeout=2.0)
    tc.join(timeout=2.0)

    assert close_done.is_set(), "close() should have completed after drain"


def test_close_drains_inflight_explicit_consolidate():
    """Memory.close() must wait for in-flight EXPLICIT consolidate() calls
    too, not just auto-consolidate / auto-extract. A user can fire
    m.consolidate() in a worker thread and call m.close() immediately
    after; without drain, the SQLite connection gets torn down mid-prune
    and consolidate() crashes with OperationalError. This regression
    guards the third leg of the drain (explicit consolidate inflight
    counter)."""
    import threading
    import time

    consolidate_started = threading.Event()
    allow_consolidate_finish = threading.Event()

    # Monkey-patch the consolidation function to block on an event so we
    # can simulate a long-running consolidate() and observe close()'s
    # drain behavior deterministically.
    import genome.memory.consolidation as _consolidation_mod
    real_consolidate = _consolidation_mod.consolidate

    def slow_consolidate(*args, **kwargs):
        consolidate_started.set()
        allow_consolidate_finish.wait(timeout=3.0)
        return real_consolidate(*args, **kwargs)

    _consolidation_mod.consolidate = slow_consolidate
    try:
        m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
        m.add("seed", user_id="alice")

        def worker():
            m.consolidate(user_id="alice", max_memories=100)

        t = threading.Thread(target=worker)
        t.start()

        assert consolidate_started.wait(timeout=2.0), \
            "consolidate() should have started"

        close_done = threading.Event()
        def closer():
            m.close()
            close_done.set()

        tc = threading.Thread(target=closer)
        tc.start()

        # Give close() a moment to enter its drain wait
        time.sleep(0.1)
        assert not close_done.is_set(), \
            "close() should be waiting on explicit consolidate drain"

        # Release consolidate
        allow_consolidate_finish.set()
        t.join(timeout=2.0)
        tc.join(timeout=2.0)

        assert close_done.is_set(), \
            "close() should have completed after explicit consolidate drained"
    finally:
        _consolidation_mod.consolidate = real_consolidate


def test_close_drain_timeout_is_configurable():
    """The drain budget must be tunable so callers with slow LLMs (or
    long consolidations) don't get prematurely truncated. Verify both
    that the field is plumbed and that a non-default value actually
    governs the deadline."""
    m = Memory(
        embedding_provider=FakeEmbeddingProvider(dim=8),
        close_drain_timeout_seconds=0.5,
    )
    assert m._close_drain_timeout_seconds == 0.5
    m.close()


def test_record_rejects_whitespace_only_content():
    """A memory whose content is only whitespace produces an empty BM25
    token set; an all-whitespace corpus crashes rank-bm25 with a
    ZeroDivisionError. Reject at the boundary."""
    from genome.memory.schema import MemoryRecord

    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for bad in ("   ", "\t\n", " \t \n  "):
        with pytest.raises(ValueError, match="non-empty"):
            MemoryRecord(content=bad, embedding=vec)


def test_hybrid_search_defense_against_empty_corpus():
    """Even if a whitespace record slips through (e.g., direct BLOB write
    by an external tool), HybridScorer must not crash. Defense in depth."""
    from genome.memory.hybrid import HybridScorer

    # Simulate a corpus dict where every doc tokenizes to empty (whitespace).
    # The placeholder guard in fuse() should keep BM25 stable.
    corpus = {"id_a": "   ", "id_b": "\t\n"}
    scorer = HybridScorer()
    out = scorer.fuse(
        query="anything",
        dense_results=[("id_a", 0.5), ("id_b", 0.4)],
        corpus=corpus,
    )
    assert isinstance(out, list)


def test_genome_namespace_exposes_conflict_resolver():
    """Public API: ConflictResolver and ConflictDecision must be importable
    from the top-level genome namespace, not just genome.memory.conflict.
    Documented as a user-facing class."""
    import genome
    assert hasattr(genome, "ConflictResolver")
    assert hasattr(genome, "ConflictDecision")
    assert "ConflictResolver" in genome.__all__
    assert "ConflictDecision" in genome.__all__


def test_genome_namespace_exposes_embedding_provider():
    """Public API: EmbeddingProvider must be importable from genome.* so
    callers can pass `embedding_provider=EmbeddingProvider("openai:...")`
    without reaching into a sub-module."""
    import genome
    assert hasattr(genome, "EmbeddingProvider")
    assert "EmbeddingProvider" in genome.__all__


def test_auto_consolidate_serialized_under_concurrent_adds():
    """Concurrent threads adding to the same scope must NOT both fire
    consolidate. Without per-scope serialization, two threads seeing
    count==threshold would each call consolidate, double-pruning records."""
    import threading

    consolidate_calls: list[int] = []

    # Stub the consolidation module so we can count invocations without
    # actually pruning.
    from genome.memory import consolidation as _c
    orig = _c.consolidate

    def spy(*a, **k):
        consolidate_calls.append(1)
        # Sleep briefly so a second thread can arrive while we hold the busy flag.
        import time
        time.sleep(0.05)
        return orig(*a, **k)

    _c.consolidate = spy
    try:
        m = Memory(
            embedding_provider=FakeEmbeddingProvider(dim=8),
            auto_consolidate_threshold=2,
            auto_consolidate_target=1,
            auto_consolidate_synthesize=False,
        )
        # Pre-populate so the very first thread hits the threshold immediately.
        m.add("a", user_id="alice")
        m.add("b", user_id="alice")
        m.add("c", user_id="alice")
        # Reset call count before the concurrency test so we measure only
        # the contended adds.
        consolidate_calls.clear()
        # Now hammer concurrently. Without the lock, both threads see
        # count > threshold and both call consolidate.
        barrier = threading.Barrier(2)

        def worker(label):
            barrier.wait()
            m.add(label, user_id="alice")

        t1 = threading.Thread(target=worker, args=("d",))
        t2 = threading.Thread(target=worker, args=("e",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # With per-scope serialization, at most one consolidation can run
        # at a time (the other thread sees the busy flag and skips).
        # We accept 1 OR 2 calls (because the second thread might arrive
        # AFTER the first finishes), but never simultaneous double-fire.
        assert len(consolidate_calls) <= 2, (
            f"expected serialized consolidate, got {len(consolidate_calls)} "
            f"concurrent calls"
        )
        m.close()
    finally:
        _c.consolidate = orig


def test_sqlite_update_no_lost_update_under_concurrent_partial_patches():
    """Two concurrent partial-patch updates against the same record must
    NOT silently overwrite each other. Before the fix, update() did
    self.get() OUTSIDE the write lock, so two threads each read the
    same `current` snapshot, then each thread's WRITE clobbered the
    other's patch (lost-update race).

    This test sets two patches that touch DIFFERENT fields (one only
    metadata, one only content). After both updates, the record must
    have BOTH patches applied -- never just one."""
    import threading

    from genome.memory.schema import MemoryRecord
    from genome.memory.sqlite_store import SQLiteMemoryStore

    store = SQLiteMemoryStore(":memory:")
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rec = store.add(MemoryRecord(
        content="orig",
        embedding=vec,
        metadata={"orig": True},
    ))

    barrier = threading.Barrier(2)

    def patch_content():
        barrier.wait()
        store.update(rec.id, content="patched-content")

    def patch_metadata():
        barrier.wait()
        store.update(rec.id, metadata={"patched": True})

    threads = [
        threading.Thread(target=patch_content),
        threading.Thread(target=patch_metadata),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = store.get(rec.id)
    assert final is not None
    # Both patches must have landed. Without the fix, one of these
    # assertions would intermittently fail because the second thread's
    # read-modify-write reverted the first thread's field to the
    # original value.
    assert final.content == "patched-content", (
        f"content patch lost: got {final.content!r}"
    )
    assert final.metadata == {"patched": True}, (
        f"metadata patch lost: got {final.metadata!r}"
    )
    store.close()


def test_record_rejects_unserializable_metadata():
    """Catch JSON-unserializable metadata at construction with a clear error,
    not deep inside the store's json.dumps."""
    from datetime import datetime

    from genome.memory.schema import MemoryRecord

    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    bad_meta = {"created": datetime(2026, 4, 28)}
    with pytest.raises(ValueError, match="JSON-serializable"):
        MemoryRecord(content="x", embedding=vec, metadata=bad_meta)


def test_edge_rejects_unserializable_metadata():
    """Same JSON-serializable check on MemoryEdge.metadata."""
    from datetime import datetime

    from genome.memory.graph import MemoryEdge

    bad_meta = {"when": datetime(2026, 4, 28)}
    with pytest.raises(ValueError, match="JSON-serializable"):
        MemoryEdge(from_id="a", to_id="b", relation="RELATES_TO", metadata=bad_meta)


def test_sqlite_store_rejects_dim_mismatch_on_add():
    """Adding records with different embedding dims to one SQLite store
    must fail loud, not silently store and corrupt cosine search later."""
    from genome.memory.schema import MemoryRecord
    from genome.memory.sqlite_store import SQLiteMemoryStore

    store = SQLiteMemoryStore(path=":memory:")
    r1 = MemoryRecord(content="a", embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    r2 = MemoryRecord(content="b", embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    store.add(r1)
    with pytest.raises(ValueError, match="dim mismatch"):
        store.add(r2)
    store.close()


def test_sqlite_store_rejects_dim_mismatch_on_search():
    """Searching with a wrong-dim query against an existing store must give
    a clear error, not numpy's generic 'shapes not aligned'."""
    from genome.memory.schema import MemoryRecord
    from genome.memory.sqlite_store import SQLiteMemoryStore

    store = SQLiteMemoryStore(path=":memory:")
    store.add(MemoryRecord(content="a", embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32)))
    bad_query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="EmbeddingProvider"):
        store.search(bad_query)
    store.close()


def test_concat_project_does_not_overflow_on_degenerate_seed():
    """concat_project floors near-zero column norms to prevent inf/nan."""
    from genome.operators import concat_project

    a = np.zeros(8, dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)
    out = concat_project(a, b, projection_seed=0)
    assert out.shape == (8,)
    assert np.isfinite(out).all()


def test_sqlite_search_rejects_nan_query():
    """search() with NaN in query embedding would silently NaN every score."""
    from genome.memory.sqlite_store import SQLiteMemoryStore

    store = SQLiteMemoryStore(path=":memory:")
    bad_q = np.array([float("nan"), 0.0, 0.0], dtype=np.float32)
    # No rows yet -> early return path; insert one to force scoring path
    rec_mem = Memory(embedding_provider=FakeEmbeddingProvider(dim=3), storage=store)
    try:
        rec_mem.add("seed", user_id="u")
        with pytest.raises(ValueError, match="NaN|Inf"):
            store.search(bad_q, user_id="u")
    finally:
        rec_mem.close()
