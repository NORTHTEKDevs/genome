# Writing an adapter

The protocol (`benchmarks/neutral/adapters.py`):

```python
class YourSystem:
    name = "yoursystem"

    def ingest(self, conversation) -> None:
        # conversation: genome.evals.locomo.LocomoConversation
        # turns have .speaker, .text, .session, .dia_id
        ...

    def answer(self, question) -> tuple[str, list[str], float]:
        # return (answer_text, retrieved_snippets, latency_seconds)
        # answer_text is produced by calling the SHARED responder you were
        # constructed with - never your own model choice.
        ...

    def close(self) -> None: ...
```

Register it in `make_system()` and it runs under identical rules to every
bundled system. `Mem0Baseline` in `genome/evals/baselines.py` is the reference
implementation (~100 lines including version detection).

## Zep

Run a Zep server locally (their docker compose), then map:
- `ingest`: `memory.add(session_id=conversation_id, messages=[...])` per turn,
  letting Graphiti build its graph (this spends THEIR extraction LLM calls -
  budget for it and report it in the disclosure block).
- `answer`: `memory.search(session_id, question, limit=top_k)` -> feed the
  returned facts/messages to the shared responder with the same answer prompt
  the baselines use.
- Fairness note: Zep's graph construction model should be the run's shared
  model where configurable; report whatever it actually was.

## Letta

Letta is an agent runtime, not a passive store, so a fair mapping needs an
explicit decision rather than a silent one. Proposed mapping:
- `ingest`: create one agent per conversation; feed turns as user messages with
  memory tools enabled and the responder model pinned to the shared model.
- `answer`: ask the question in a fresh session of the same agent; cap tool
  steps (e.g. 5) and report the cap in the disclosure block.
- Report tokens spent by self-editing memory operations - it is part of the
  system's real cost, exactly like Mem0's extraction calls.
