import pytest

from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider

llama = pytest.importorskip("llama_index.core")
from llama_index.core.llms import ChatMessage, MessageRole  # noqa: E402

from genome.adapters.llamaindex import GenomeChatMemory  # noqa: E402


@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=16))
    yield m
    m.close()


def test_put_and_get_preserves_order(mem):
    cm = GenomeChatMemory(memory=mem, session_id="s1")
    cm.put(ChatMessage(role=MessageRole.USER, content="hello"))
    cm.put(ChatMessage(role=MessageRole.ASSISTANT, content="hi there"))
    cm.put(ChatMessage(role=MessageRole.USER, content="what's up?"))

    msgs = cm.get()
    assert len(msgs) == 3
    assert [m.content for m in msgs] == ["hello", "hi there", "what's up?"]
    roles = [m.role for m in msgs]
    assert roles == [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER]


def test_get_all_alias(mem):
    cm = GenomeChatMemory(memory=mem, session_id="s1")
    cm.put(ChatMessage(role=MessageRole.USER, content="x"))
    assert len(cm.get_all()) == 1


def test_set_replaces_history(mem):
    cm = GenomeChatMemory(memory=mem, session_id="s1")
    cm.put(ChatMessage(role=MessageRole.USER, content="old"))
    cm.set([
        ChatMessage(role=MessageRole.USER, content="new1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="new2"),
    ])
    contents = [m.content for m in cm.get()]
    assert contents == ["new1", "new2"]


def test_reset_clears(mem):
    cm = GenomeChatMemory(memory=mem, session_id="s1")
    cm.put(ChatMessage(role=MessageRole.USER, content="to be cleared"))
    cm.reset()
    assert cm.get() == []


def test_get_relevant_searches(mem):
    cm = GenomeChatMemory(memory=mem, session_id="s1")
    cm.put(ChatMessage(role=MessageRole.USER, content="I love coffee"))
    cm.put(ChatMessage(role=MessageRole.USER, content="I love pizza"))
    cm.put(ChatMessage(role=MessageRole.USER, content="totally unrelated quantum physics"))
    relevant = cm.get_relevant("food preferences", top_k=3)
    assert len(relevant) == 3
    # All returned should come from this session's memories
    contents = {m.content for m in relevant}
    assert contents <= {"I love coffee", "I love pizza", "totally unrelated quantum physics"}


def test_sessions_isolated(mem):
    alice = GenomeChatMemory(memory=mem, session_id="alice")
    bob = GenomeChatMemory(memory=mem, session_id="bob")
    alice.put(ChatMessage(role=MessageRole.USER, content="alice msg"))
    bob.put(ChatMessage(role=MessageRole.USER, content="bob msg"))
    assert [m.content for m in alice.get()] == ["alice msg"]
    assert [m.content for m in bob.get()] == ["bob msg"]
