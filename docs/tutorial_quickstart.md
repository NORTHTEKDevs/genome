# Quickstart: 5 minutes from zero to a working memory layer

## Install

GENOME is open source under the Apache License 2.0. Install from PyPI:

```bash
pip install genome-memory
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/NORTHTEKDevs/genome.git && cd genome
pip install -e .
# or with optional backends:
pip install -e ".[postgres,fastapi]"
```

## Hello memory

```python
from genome import Memory

m = Memory()  # in-memory SQLite
m.add("I love pour-over coffee", user_id="alice")
m.add("I moved to Tokyo last month", user_id="alice")
m.add("I work as a data scientist", user_id="alice")

results = m.search("what drinks does the user like?", user_id="alice", limit=3)
for r in results:
    print(f"{r.score:.3f}  {r.content}")
```

Expected output:
```
0.452  I love pour-over coffee
0.219  I moved to Tokyo last month
0.161  I work as a data scientist
```

## Persist to a file

```python
m = Memory(storage="memories.db")
```

Everything you add persists; reopen the same file in a later run and it's all there.

## Multi-user isolation

```python
m.add("secret A", user_id="alice")
m.add("secret B", user_id="bob")

# Alice's search only returns Alice's memories
m.search("secret", user_id="alice")  # -> ["secret A"]
m.search("secret", user_id="bob")    # -> ["secret B"]
```

## LLM-extracted atomic facts

```python
import os
from anthropic import Anthropic

client = Anthropic()

def claude(prompt: str) -> str:
    return client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text

m = Memory(llm_call=claude)
m.add(
    "I just moved to Tokyo, love pour-over coffee, and I work as a data scientist",
    user_id="alice",
)
# Adds 3 memories: "user lives in Tokyo", "user likes pour-over coffee",
#                  "user works as a data scientist"
```

## Synthesize a hybrid memory (the unique feature)

```python
# Find related memories
ids = [r.id for r in m.search("user lifestyle", user_id="alice", limit=3)]

# Recombine their embeddings into a new memory
hybrid = m.synthesize(
    memory_ids=ids,
    user_id="alice",
    operator="uniform_crossover",   # or "frequency_crossover", "simple_average", ...
)

# The hybrid has its own id, embedding, and provenance
print(hybrid.id)        # "mem_..."
print(hybrid.parents)   # [id1, id2, id3]
print(hybrid.operator)  # "uniform_crossover"

# Subsequent searches automatically hide the parents (so the hybrid surfaces)
m.search("lifestyle summary", user_id="alice", limit=3)
```

## Typed graph relations

```python
from genome import SUPERSEDES, CONTRADICTS

new_fact = m.add("I now drink tea, not coffee", user_id="alice")[0]
old_fact = m.search("coffee", user_id="alice", limit=1)[0].record

m.link(new_fact.id, old_fact.id, relation=SUPERSEDES, weight=0.9)

# Later: find all facts this one supersedes
for stale in m.related(new_fact.id, relation=SUPERSEDES):
    print("Outdated:", stale.content)
```

## Consolidation (fitness-based pruning + hybridization)

```python
# After many conversations, prune to top-500 by fitness (access count + recency)
result = m.consolidate(
    user_id="alice",
    max_memories=500,
    synthesize_before_prune=True,  # combine pairs before deleting
)
print(f"Kept {result.kept}, pruned {result.pruned}, hybridized {result.synthesized}")
```

## RAPTOR hierarchical summaries

```python
# Build a tree: clusters memories, summarizes each cluster, recurses
m.build_raptor_tree(user_id="alice", branching_factor=4, max_levels=3, llm_call=claude)

# Retrieve at any level
atomic_hits = m.search_at_level("did the user go to Paris?", user_id="alice", level=0)
summary_hits = m.search_at_level("what's the user like?", user_id="alice", level=2)
```

## Entity extraction + graph (GraphRAG-style)

```python
rec = m.add("Alice works at OpenAI in San Francisco", user_id="company")[0]
m.extract_entities(rec.id, llm_call=claude)

# Entity records are stored with metadata
for ent in m.list_entities(user_id="company", entity_type="PERSON"):
    print(ent.content, ent.metadata)

# Which memories mention each entity
alice_ent = m.list_entities(user_id="company", entity_type="PERSON")[0]
for mem in m.memories_mentioning(alice_ent.id):
    print(mem.content)
```

## Async API

```python
from genome import AsyncMemory

async def main():
    async with AsyncMemory(storage="mem.db") as m:
        await m.add("hello", user_id="alice")
        results = await m.search("hello", user_id="alice")
```

## LangChain integration

```python
from genome import Memory
from genome.adapters.langchain import GenomeChatMessageHistory

mem = Memory(storage="chat.db")
history = GenomeChatMessageHistory(memory=mem, session_id="session_1")
# Use as any BaseChatMessageHistory in LangChain chains
```

## LlamaIndex integration

```python
from genome.adapters.llamaindex import GenomeChatMemory
from llama_index.core.llms import ChatMessage, MessageRole

chat_memory = GenomeChatMemory(memory=mem, session_id="alice")
chat_memory.put(ChatMessage(role=MessageRole.USER, content="hello"))
relevant = chat_memory.get_relevant("what did I say?", top_k=3)
```

## REST API

The server is safe by default: it binds `127.0.0.1` and won't serve without auth.
For **local development** without a key, opt in explicitly:

```bash
pip install "genome-memory[fastapi]"
GENOME_ALLOW_NO_AUTH=1 python -m genome.server   # local dev only, loopback
```

For anything exposed, set a key instead (required to bind beyond localhost):

```bash
GENOME_API_KEY=$(openssl rand -hex 32) GENOME_HOST=0.0.0.0 python -m genome.server
```

Then (add `-H "X-API-Key: <key>"` when a key is set):
```bash
curl -X POST http://localhost:8080/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"text": "I love coffee", "user_id": "alice"}'

curl -X POST http://localhost:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "drinks", "user_id": "alice", "limit": 5}'
```

Interactive docs at `http://localhost:8080/docs`.

## Docker (with Postgres)

```bash
docker-compose up
# genome at http://localhost:8080, Postgres at 5432
```

## Next

- [Architecture](architecture.md) -- how genome works internally
- [API reference](api_reference.md) -- every method and its options
- [Changelog](../CHANGELOG.md)
