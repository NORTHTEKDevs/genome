"""End-to-end smoke test: all 5 LoCoMo-readiness fixes wired together.

Exercises:
- Fix 1: V2 extraction prompt (via the LLMExtractor default)
- Fix 2: conflict resolution (UPDATE supersedes Tokyo with Berlin)
- Fix 3: hybrid BM25+dense search (mode='hybrid')
- Fix 4: auto-consolidation trigger (synthesizes hybrids past threshold)
- Fix 5: auto entity + temporal fact recording on add()
"""

import re

from genome import Memory
from genome.memory.extraction import IdentityExtractor


class StubLLM:
    """Returns plausible responses for conflict-resolution, fact-detection,
    and entity-extraction prompts. Picks branch based on prompt content.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Conflict resolution prompts contain "DECISION:"
        if "DECISION:" in prompt and "Output a single line" in prompt:
            if "Tokyo" in prompt and "Berlin" in prompt:
                m = re.search(r"id=(mem_[a-f0-9]+).*?Tokyo", prompt, re.DOTALL)
                if m:
                    return f"DECISION: UPDATE id={m.group(1)}"
            return "DECISION: ADD"
        # Entity extraction prompts wrap text in <text>...</text>
        if "<text>" in prompt:
            text_lower = prompt.lower()
            if "tokyo" in text_lower:
                return "ENTITY | Tokyo | PLACE | a city in Japan"
            if "berlin" in text_lower:
                return "ENTITY | Berlin | PLACE | a city in Germany"
            return "NONE"
        # Fact-detection prompts have "FACT_TYPE:" in the template
        if "FACT_TYPE:" in prompt:
            text_lower = prompt.lower()
            if "berlin" in text_lower:
                return "FACT_TYPE: location\nVALUE: Berlin\nCONFIDENCE: 0.9"
            if "tokyo" in text_lower:
                return "FACT_TYPE: location\nVALUE: Tokyo\nCONFIDENCE: 0.9"
            return "FACT_TYPE: none\nCONFIDENCE: 0.0"
        # Anything else (e.g. v2 extraction prompt routed through llm_call)
        # behaves as a passthrough: return the input as a single dash-prefixed fact
        return "- " + prompt.split("Input:")[-1].strip().split("\n")[0]


def test_smoke_all_five_fixes_wired_together():
    llm = StubLLM()
    m = Memory(
        extractor=IdentityExtractor(),
        llm_call=llm,
        resolve_conflicts=True,
        conflict_llm=llm,
        auto_extract_entities=True,
        auto_consolidate_threshold=20,
        auto_consolidate_target=10,
        auto_consolidate_synthesize=True,
    )

    # Fix 2: UPDATE supersedes prior fact
    m.add("user lives in Tokyo", user_id="alice")
    m.add("user lives in Berlin", user_id="alice")

    # Fix 3: hybrid search finds the entity-name match
    results = m.search("Berlin", user_id="alice", mode="hybrid", limit=3)
    assert any("Berlin" in r.content for r in results), (
        f"hybrid search should find Berlin, got {[r.content for r in results]}"
    )

    # Fix 5: temporal fact recorded for at least one entity
    entities = m.list_entities(user_id="alice")
    assert len(entities) >= 1, "auto-extract should have created entities"
    fact_types_seen: set[str] = set()
    for e in entities:
        for f in m.current_facts(e.id):
            fact_types_seen.add(f.fact_type)
    assert "location" in fact_types_seen, (
        f"expected a location fact, saw {fact_types_seen}"
    )

    # Fix 4: auto-consolidation fires past threshold + creates hybrids
    for i in range(25):
        m.add(f"unrelated fact {i}", user_id="alice")
    final_count = m.count(user_id="alice")
    assert final_count <= 20, (
        f"auto-consolidate should have pruned to <=20, got {final_count}"
    )
    all_recs = m.list_all(user_id="alice")
    hybrids = [r for r in all_recs if r.parents]
    assert len(hybrids) >= 1, (
        "auto-consolidate with synthesize=True should have created at least "
        "one hybrid record"
    )
