import pytest

from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider

langchain_core = pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from genome.adapters.langchain import (  # noqa: E402
    GenomeChatMessageHistory,
    GenomeRetrieverMemory,
)


@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=16))
    yield m
    m.close()


# ---------- GenomeChatMessageHistory ----------

def test_chat_history_add_user_and_ai_messages(mem):
    h = GenomeChatMessageHistory(memory=mem, session_id="s1")
    h.add_user_message("hello bot")
    h.add_ai_message("hello user")

    msgs = h.messages
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[0].content == "hello bot"
    assert msgs[1].content == "hello user"


def test_chat_history_add_message_preserves_role(mem):
    h = GenomeChatMessageHistory(memory=mem, session_id="s1")
    h.add_message(HumanMessage(content="ping"))
    h.add_message(AIMessage(content="pong"))
    roles = [type(m).__name__ for m in h.messages]
    assert roles == ["HumanMessage", "AIMessage"]


def test_chat_history_isolated_by_session(mem):
    a = GenomeChatMessageHistory(memory=mem, session_id="alice")
    b = GenomeChatMessageHistory(memory=mem, session_id="bob")
    a.add_user_message("alice here")
    b.add_user_message("bob here")
    assert len(a.messages) == 1
    assert len(b.messages) == 1
    assert a.messages[0].content == "alice here"
    assert b.messages[0].content == "bob here"


def test_chat_history_clear(mem):
    h = GenomeChatMessageHistory(memory=mem, session_id="s1")
    h.add_user_message("to be cleared")
    assert len(h.messages) == 1
    h.clear()
    assert h.messages == []


def test_chat_history_ordering_by_created_at(mem):
    h = GenomeChatMessageHistory(memory=mem, session_id="s1")
    h.add_user_message("first")
    h.add_ai_message("second")
    h.add_user_message("third")
    contents = [m.content for m in h.messages]
    assert contents == ["first", "second", "third"]


# ---------- GenomeRetrieverMemory ----------

def test_retriever_memory_variables(mem):
    r = GenomeRetrieverMemory(memory=mem, user_id="alice", memory_key="history")
    assert r.memory_variables == ["history"]


def test_retriever_load_with_empty_store(mem):
    r = GenomeRetrieverMemory(memory=mem, user_id="alice")
    out = r.load_memory_variables({"input": "anything"})
    assert out == {"history": "(no relevant memories)"}


def test_retriever_load_returns_bulletlist(mem):
    mem.add("user loves coffee", user_id="alice")
    mem.add("user moved to Tokyo", user_id="alice")
    r = GenomeRetrieverMemory(memory=mem, user_id="alice", top_k=5)
    out = r.load_memory_variables({"input": "what about coffee?"})
    assert "user loves coffee" in out["history"]
    assert out["history"].startswith("-")


def test_retriever_save_context_persists_both_turns(mem):
    r = GenomeRetrieverMemory(memory=mem, user_id="alice")
    r.save_context(
        {"input": "what's the weather?"},
        {"output": "sunny and 75"},
    )
    recs = mem.list_all(user_id="alice")
    contents = {r.content for r in recs}
    assert "what's the weather?" in contents
    assert "sunny and 75" in contents
    roles = {r.metadata.get("role") for r in recs}
    assert roles == {"human", "ai"}


def test_retriever_scoped_by_user(mem):
    mem.add("alice memory", user_id="alice")
    mem.add("bob memory", user_id="bob")
    r_alice = GenomeRetrieverMemory(memory=mem, user_id="alice")
    r_bob = GenomeRetrieverMemory(memory=mem, user_id="bob")
    assert "alice memory" in r_alice.load_memory_variables({"input": "anything"})["history"]
    assert "bob memory" in r_bob.load_memory_variables({"input": "anything"})["history"]
    assert "alice memory" not in r_bob.load_memory_variables({"input": "anything"})["history"]


def test_retriever_clear(mem):
    mem.add("some fact", user_id="alice")
    r = GenomeRetrieverMemory(memory=mem, user_id="alice")
    r.clear()
    assert mem.count(user_id="alice") == 0
