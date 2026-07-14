"""Tests for FACT_EXTRACTION_PROMPT_V2 with categories + few-shot examples."""

from genome.memory.extraction import (
    FACT_EXTRACTION_PROMPT_V2,
    LLMExtractor,
)


class FakeLLM:
    """Predictable LLM for testing prompt -> response shape."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        for keyword, response in self.responses.items():
            if keyword in prompt:
                return response
        return "NO_FACTS"


def test_v2_prompt_contains_categories():
    """V2 prompt names the 5 fact categories."""
    for cat in ["preference", "plan", "relationship", "professional", "temporal"]:
        assert cat in FACT_EXTRACTION_PROMPT_V2.lower()


def test_v2_prompt_contains_few_shot_examples():
    """V2 prompt has at least 5 input/output examples."""
    assert FACT_EXTRACTION_PROMPT_V2.lower().count("input:") >= 5
    assert FACT_EXTRACTION_PROMPT_V2.lower().count("facts:") >= 5


def test_v2_pronoun_resolution_rule():
    """V2 prompt instructs to resolve pronouns within input."""
    assert "pronoun" in FACT_EXTRACTION_PROMPT_V2.lower()


def test_v2_temporal_cue_rule():
    """V2 prompt instructs to preserve temporal cues."""
    text = FACT_EXTRACTION_PROMPT_V2.lower()
    assert any(cue in text for cue in ["temporal", "yesterday", "date"])


def test_extractor_uses_v2_prompt_by_default():
    """LLMExtractor with prompt_version='v2' calls LLM with V2 template."""
    fake = FakeLLM({"Input:": "- user likes coffee"})
    extractor = LLMExtractor(fake, prompt_version="v2")
    facts = extractor.extract("I love coffee")
    assert facts == ["user likes coffee"]
    assert "preference" in fake.calls[0].lower()  # V2 marker
