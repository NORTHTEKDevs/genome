"""Tests for auto-extraction of entities + temporal facts on Memory.add()."""

import pytest

from genome import Memory
from genome.memory.extraction import IdentityExtractor


class FactTypeLLM:
    """Returns plausible LLM responses for both entity-extraction and
    fact-detection prompts. Picks based on prompt content.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Entity extraction prompt is identifiable by "<text>" wrapping
        if "<text>" in prompt:
            text_lower = prompt.lower()
            if "tokyo" in text_lower:
                return "ENTITY | Tokyo | PLACE | a city in Japan"
            if "berlin" in text_lower:
                return "ENTITY | Berlin | PLACE | a city in Germany"
            if "google" in text_lower:
                return "ENTITY | Google | ORG | a tech company"
            return "NONE"
        # Otherwise treat as the fact-detection prompt
        text_lower = prompt.lower()
        if "tokyo" in text_lower:
            return "FACT_TYPE: location\nVALUE: Tokyo\nCONFIDENCE: 0.9"
        if "berlin" in text_lower:
            return "FACT_TYPE: location\nVALUE: Berlin\nCONFIDENCE: 0.9"
        if "google" in text_lower:
            return "FACT_TYPE: employer\nVALUE: Google\nCONFIDENCE: 0.85"
        return "FACT_TYPE: none\nCONFIDENCE: 0.0"


def test_auto_extract_off_by_default():
    """Without auto_extract_entities, add() does NOT create entities."""
    m = Memory(extractor=IdentityExtractor())
    m.add("user lives in Tokyo", user_id="alice")
    entities = m.list_entities(user_id="alice")
    assert entities == []


def test_auto_extract_requires_llm():
    """auto_extract_entities=True without any LLM raises at construction."""
    with pytest.raises(ValueError, match="llm"):
        Memory(extractor=IdentityExtractor(), auto_extract_entities=True)


def test_auto_extract_creates_entity_record_on_add():
    """When enabled, add() persists at least one entity."""
    m = Memory(
        extractor=IdentityExtractor(),
        llm_call=FactTypeLLM(),
        auto_extract_entities=True,
    )
    m.add("user lives in Tokyo", user_id="alice")
    entities = m.list_entities(user_id="alice")
    assert len(entities) >= 1
    names = [e.metadata.get("entity_name", "") for e in entities]
    assert "Tokyo" in names


def test_auto_extract_records_high_confidence_location_fact():
    """High-confidence fact-detection produces an EntityFact record."""
    m = Memory(
        extractor=IdentityExtractor(),
        llm_call=FactTypeLLM(),
        auto_extract_entities=True,
    )
    m.add("user lives in Tokyo", user_id="alice")
    entities = m.list_entities(user_id="alice")
    assert len(entities) >= 1
    found_location_fact = False
    for e in entities:
        for f in m.current_facts(e.id):
            if f.fact_type == "location":
                found_location_fact = True
                break
    assert found_location_fact, "no location fact recorded for any entity"


class LowConfidenceLLM:
    """Always returns confidence below 0.7 for fact detection."""

    def __call__(self, prompt: str) -> str:
        if "<text>" in prompt:
            return "ENTITY | Mystery | OTHER | unknown thing"
        return "FACT_TYPE: location\nVALUE: somewhere\nCONFIDENCE: 0.3"


def test_auto_extract_skips_low_confidence_facts():
    """Confidence below 0.7 must NOT produce a fact record."""
    m = Memory(
        extractor=IdentityExtractor(),
        llm_call=LowConfidenceLLM(),
        auto_extract_entities=True,
    )
    m.add("something vague happened", user_id="alice")
    entities = m.list_entities(user_id="alice")
    # Entity may be created, but no facts attached to it
    for e in entities:
        assert m.current_facts(e.id) == []


def test_auto_extract_swallows_llm_error_keeps_memory():
    """If the entity-extraction LLM raises, add() must still complete and
    the memory must be stored. Auto-extract is best-effort, not blocking."""
    class FailingLLM:
        def __call__(self, prompt: str) -> str:
            raise RuntimeError("simulated entity-extract API failure")

    m = Memory(
        extractor=IdentityExtractor(),
        llm_call=FailingLLM(),
        auto_extract_entities=True,
    )
    recs = m.add("user lives in Tokyo", user_id="alice")
    # Memory was stored despite the LLM error
    assert len(recs) == 1
    assert recs[0].content == "user lives in Tokyo"
    assert m.count(user_id="alice") == 1


def test_auto_extract_swallows_fact_detection_error():
    """If entity extraction succeeds but fact detection raises, the entity
    must still be persisted; only the temporal fact is dropped."""
    class PartialFailLLM:
        def __init__(self):
            self.call_count = 0

        def __call__(self, prompt: str) -> str:
            self.call_count += 1
            if "<text>" in prompt:
                return "ENTITY | Tokyo | PLACE | a city in Japan"
            # Fact-detection LLM call: raise
            raise RuntimeError("simulated fact-detection API failure")

    m = Memory(
        extractor=IdentityExtractor(),
        llm_call=PartialFailLLM(),
        auto_extract_entities=True,
    )
    recs = m.add("user lives in Tokyo", user_id="alice")
    assert len(recs) == 1
    # Entity should have been recorded; fact should not.
    entities = m.list_entities(user_id="alice")
    if entities:
        for e in entities:
            assert m.current_facts(e.id) == []
