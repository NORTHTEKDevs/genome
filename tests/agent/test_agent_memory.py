"""Tests for the Letta-style agent memory runtime."""
import pytest

from genome.agent import AgentMemory, CoreBlock, tool_schemas
from genome.agent.memory import CORE_MEMORY_OPERATOR
from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider


@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=16))
    yield m
    m.close()


@pytest.fixture
def agent(mem):
    return AgentMemory(memory=mem, user_id="alice", session_id="s1")


# ---------- CoreBlock ----------

def test_core_block_append():
    b = CoreBlock(label="x", max_chars=100)
    b.append("first")
    b.append("second")
    assert b.value == "first\nsecond"


def test_core_block_append_over_cap_raises():
    b = CoreBlock(label="x", max_chars=10)
    b.append("12345")
    with pytest.raises(ValueError, match="exceed"):
        b.append("678910")


def test_core_block_replace():
    b = CoreBlock(label="x", value="I like NYC", max_chars=100)
    b.replace("NYC", "Tokyo")
    assert b.value == "I like Tokyo"


def test_core_block_replace_missing_raises():
    b = CoreBlock(label="x", value="I like NYC")
    with pytest.raises(ValueError, match="not found"):
        b.replace("London", "Tokyo")


# ---------- default blocks ----------

def test_default_core_blocks(agent):
    assert set(agent.core_blocks.keys()) == {"persona", "user", "scratch"}


def test_render_core_empty(agent):
    rendered = agent.render_core()
    assert "## persona" in rendered
    assert "## user" in rendered
    assert "## scratch" in rendered
    # Empty blocks render as "(empty)"
    assert rendered.count("(empty)") == 3


def test_render_core_with_values(agent):
    agent.core_blocks["user"].value = "name: Alice"
    agent.core_blocks["persona"].value = "helpful assistant"
    r = agent.render_core()
    assert "name: Alice" in r
    assert "helpful assistant" in r


# ---------- core tools ----------

def test_core_append_tool(agent):
    out = agent.core_append("user", "likes coffee")
    assert out["ok"] is True
    assert agent.core_blocks["user"].value == "likes coffee"


def test_core_append_unknown_label(agent):
    out = agent.core_append("nonexistent", "x")
    assert "error" in out


def test_core_replace_tool(agent):
    agent.core_blocks["user"].value = "lives in NYC"
    out = agent.core_replace("user", "NYC", "Tokyo")
    assert out["ok"] is True
    assert agent.core_blocks["user"].value == "lives in Tokyo"


def test_core_replace_missing_returns_error(agent):
    out = agent.core_replace("user", "London", "Tokyo")
    assert "error" in out


# ---------- persistence ----------

def test_core_persists_across_instances(mem):
    a1 = AgentMemory(memory=mem, user_id="alice", session_id="s1")
    a1.core_append("user", "remember this")

    # New AgentMemory for same user/session -> loads persisted core
    a2 = AgentMemory(memory=mem, user_id="alice", session_id="s1")
    assert a2.core_blocks["user"].value == "remember this"


def test_core_blocks_scoped_by_session(mem):
    a_s1 = AgentMemory(memory=mem, user_id="alice", session_id="s1")
    a_s2 = AgentMemory(memory=mem, user_id="alice", session_id="s2")
    a_s1.core_append("user", "s1 note")
    a_s2.core_append("user", "s2 note")

    a_s1_reload = AgentMemory(memory=mem, user_id="alice", session_id="s1")
    a_s2_reload = AgentMemory(memory=mem, user_id="alice", session_id="s2")
    assert a_s1_reload.core_blocks["user"].value == "s1 note"
    assert a_s2_reload.core_blocks["user"].value == "s2 note"


# ---------- archival tools ----------

def test_archival_insert_and_search(agent):
    out = agent.archival_insert("user loves pour-over coffee")
    assert out["ok"] is True
    assert len(out["ids"]) == 1

    results = agent.archival_search("what drinks?", limit=5)
    contents = [r["content"] for r in results["results"]]
    assert "user loves pour-over coffee" in contents


def test_archival_search_excludes_core_memory(agent):
    # Core blocks ARE stored in archival as CORE_MEMORY_OPERATOR records;
    # archival_search must filter them out.
    agent.core_append("user", "secret user info")
    agent.archival_insert("regular archival fact")

    results = agent.archival_search("any", limit=10)
    # No result should be a core_memory record
    for r in results["results"]:
        rec = agent.memory.store.get(r["id"])
        assert rec.operator != CORE_MEMORY_OPERATOR


def test_archival_delete(agent):
    out = agent.archival_insert("will be deleted")
    mem_id = out["ids"][0]
    delete_result = agent.archival_delete(mem_id)
    assert delete_result["ok"] is True


def test_archival_delete_cross_scope_refused(mem):
    """Alice can't delete Bob's memory via AgentMemory scoping."""
    alice = AgentMemory(memory=mem, user_id="alice", session_id="s1")
    bob = AgentMemory(memory=mem, user_id="bob", session_id="s1")

    bob_id = bob.archival_insert("bob's secret")["ids"][0]
    # alice tries to delete bob's memory
    result = alice.archival_delete(bob_id)
    assert result["ok"] is False


# ---------- synthesis ----------

def test_synthesize_memories(agent):
    a = agent.archival_insert("alpha fact")["ids"][0]
    b = agent.archival_insert("beta fact")["ids"][0]
    out = agent.synthesize_memories([a, b], operator="simple_average")
    assert out["ok"] is True
    assert "id" in out


def test_synthesize_single_parent_returns_error(agent):
    a = agent.archival_insert("only one")["ids"][0]
    out = agent.synthesize_memories([a])
    assert "error" in out


# ---------- tool dispatch ----------

def test_handle_tool_call_core_append(agent):
    out = agent.handle_tool_call("core_append", {"label": "user", "text": "x"})
    assert out["ok"] is True


def test_handle_tool_call_unknown_tool(agent):
    out = agent.handle_tool_call("nonexistent", {})
    assert "error" in out
    assert "unknown tool" in out["error"]


def test_handle_tool_call_missing_args(agent):
    out = agent.handle_tool_call("core_append", {"label": "user"})
    assert "error" in out
    assert "missing" in out["error"].lower()


def test_handle_tool_call_archival_insert(agent):
    out = agent.handle_tool_call(
        "archival_insert", {"content": "tool-inserted fact"},
    )
    assert out["ok"] is True


def test_handle_tool_call_archival_search(agent):
    agent.archival_insert("searchable fact")
    out = agent.handle_tool_call(
        "archival_search", {"query": "anything", "limit": 3},
    )
    assert "results" in out


# ---------- tool schemas ----------

def test_anthropic_tool_schemas():
    schemas = tool_schemas("anthropic")
    assert len(schemas) == 6
    names = {s["name"] for s in schemas}
    assert names == {
        "core_append", "core_replace", "archival_insert",
        "archival_search", "archival_delete", "synthesize_memories",
    }
    for s in schemas:
        assert "description" in s
        assert "input_schema" in s
        assert s["input_schema"]["type"] == "object"


def test_openai_tool_schemas():
    schemas = tool_schemas("openai")
    assert len(schemas) == 6
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "parameters" in s["function"]


def test_tool_schemas_invalid_format():
    with pytest.raises(ValueError, match="unknown format"):
        tool_schemas("mystery")


# ---------- archival_count ----------

def test_archival_count_excludes_core(agent):
    agent.archival_insert("fact 1")
    agent.archival_insert("fact 2")
    agent.core_append("user", "core stuff")
    # core doesn't count
    assert agent.archival_count() == 2
