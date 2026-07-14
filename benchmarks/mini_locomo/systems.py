"""System adapters for the mini-LoCoMo benchmark.

Two systems compared:
- naive: just stores raw turn text and does dense cosine search at query time
- genome_full: GENOME with all 5 LoCoMo-readiness fixes enabled

Both share the same answer-generation prompt so that retrieval quality is the
isolated variable.
"""

from __future__ import annotations

from typing import Protocol

from benchmarks.mini_locomo.data import Turn
from genome import Memory
from genome.memory.extraction import IdentityExtractor

ANSWER_PROMPT = """\
You are answering a question based ONLY on the retrieved memory snippets below.

Retrieved memories (most relevant first):
{context}

Question: {question}

Rules:
- Use only information present in the retrieved memories.
- If the answer cannot be determined from the memories, say "Unknown".
- Be concise: 1-3 short sentences.
- For yes/no questions, start with "Yes" or "No".
"""


class System(Protocol):
    name: str

    def ingest(self, turns: list[Turn], user_id: str) -> None: ...
    def answer(self, question: str, user_id: str, llm) -> str: ...


def _build_context(results) -> str:
    if not results:
        return "(no memories retrieved)"
    return "\n".join(f"- {r.content}" for r in results)


def _format_turn(turn: Turn) -> str:
    """Stamp each turn with its session timestamp so temporal queries can
    succeed even without a dedicated temporal feature."""
    return f"[{turn.timestamp_iso}] {turn.text}"


class NaiveSystem:
    """Baseline: store every turn as one memory, dense cosine search at query.

    No extraction, no conflict resolution, no temporal KG, no recombination.
    Equivalent to the simplest "RAG over conversation history" pattern.
    """

    name = "naive"

    def __init__(self) -> None:
        self.mem = Memory(extractor=IdentityExtractor())

    def ingest(self, turns: list[Turn], user_id: str) -> None:
        for t in turns:
            self.mem.add(_format_turn(t), user_id=user_id)

    def answer(self, question: str, user_id: str, llm) -> str:
        results = self.mem.search(
            question, user_id=user_id, limit=8, mode="dense"
        )
        prompt = ANSWER_PROMPT.format(
            context=_build_context(results), question=question
        )
        return llm(prompt).strip()

    def close(self) -> None:
        self.mem.close()


class GenomeFullSystem:
    """All 5 LoCoMo-readiness fixes enabled.

    Uses the same Claude wrapper as the answer LLM, so the comparison is
    isolated to (architecture + retrieval) rather than (architecture + LLM).
    """

    name = "genome_full"

    def __init__(self, llm) -> None:
        # Reuse the same llm callable for extraction, conflict resolution,
        # entity extraction, and fact detection.
        self.mem = Memory(
            extractor=IdentityExtractor(),  # turns are already atomic-shaped
            llm_call=llm,
            resolve_conflicts=True,
            conflict_llm=llm,
            auto_extract_entities=True,
            # auto_consolidate_threshold off for tiny dataset
        )

    def ingest(self, turns: list[Turn], user_id: str) -> None:
        for t in turns:
            self.mem.add(_format_turn(t), user_id=user_id)

    def answer(self, question: str, user_id: str, llm) -> str:
        results = self.mem.search(
            question, user_id=user_id, limit=8, mode="hybrid"
        )
        prompt = ANSWER_PROMPT.format(
            context=_build_context(results), question=question
        )
        return llm(prompt).strip()

    def close(self) -> None:
        self.mem.close()
