"""LLM-as-judge utilities for conversational memory evaluation.

Accepts any `LLMCallFn` so it works with Claude, OpenAI, or any callable. The
prompts used here mirror the Mem0 LOCOMO methodology (semantic-equivalence
judging with the gold answer + prediction in the same prompt).
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import re
from dataclasses import dataclass

from genome.memory.extraction import LLMCallFn

JUDGE_PROMPT = """\
You are evaluating whether a predicted answer correctly answers a question \
given the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Does the predicted answer convey the same information as the gold answer? \
Minor phrasing differences, paraphrasing, or extra context do not count as wrong \
as long as the core factual claim matches.

Output exactly one of:
CORRECT
INCORRECT
PARTIAL

Then on a new line, a brief one-sentence reason.
"""

# Binary variant for headline "J" scores. The de-facto industry protocol
# (Mem0 arXiv:2504.19413 and everyone who compares against it) uses a
# binary correct/incorrect judge; PARTIAL does not exist there. Keep the
# graded prompt above for internal diagnostics; publish with this one.
JUDGE_PROMPT_BINARY = """\
You are evaluating whether a predicted answer correctly answers a question \
given the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Does the predicted answer convey the same information as the gold answer? \
Minor phrasing differences, paraphrasing, or extra surrounding context do \
not count as wrong as long as the core factual claim matches. An answer \
that misses the core fact, is only vaguely related, or contradicts the \
gold answer is wrong.

Output exactly one of:
CORRECT
INCORRECT

Then on a new line, a brief one-sentence reason.
"""

# The judge prompt from Mem0's published LoCoMo harness
# (github.com/mem0ai/memory-benchmarks, benchmarks/locomo/prompts.py,
# no-evidence variant), reproduced VERBATIM including its system line.
# This is the most-reused judge in the ecosystem; running with it removes
# the "homemade judge" attack surface entirely. Its leniency rules
# (partial credit, 14-day date tolerance, paraphrase acceptance) apply
# equally to every system in a within-harness comparison.
JUDGE_SYSTEM_PROMPT_MEM0 = (
    "You are evaluating conversational AI memory recall. "
    "Return JSON only with the format requested."
)

JUDGE_PROMPT_MEM0 = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {gold}
Generated answer: {predicted}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


def judge_prompt_for_mode(mode: str) -> str:
    """The exact judge prompt text used for a given judge mode.

    Single source of truth so callers (the judge itself, and the methodology
    recorder that publishes the prompt) can never disagree -- a mismatch there
    reads as "claimed the ecosystem judge, shipped a homemade one".
    """
    if mode == "mem0":
        return JUDGE_SYSTEM_PROMPT_MEM0 + "\n\n" + JUDGE_PROMPT_MEM0
    if mode == "binary":
        return JUDGE_PROMPT_BINARY
    return JUDGE_PROMPT


def preprocess_gold_mem0(category: str, gold: str) -> str:
    """Mem0-harness gold preprocessing: open-domain (category 3) gold
    answers keep only the part before the first semicolon. Reproduced from
    their preprocess_answer() so mem0-mode runs match their protocol."""
    if category == "open-domain" and ";" in gold:
        return gold.split(";")[0].strip()
    return gold


@dataclass
class JudgeVerdict:
    label: str           # "CORRECT" | "INCORRECT" | "PARTIAL"
    reason: str
    raw: str

    @property
    def is_correct(self) -> bool:
        return self.label == "CORRECT"

    @property
    def is_partial(self) -> bool:
        return self.label == "PARTIAL"

    @property
    def score(self) -> float:
        """Numeric score: CORRECT=1, PARTIAL=0.5, INCORRECT=0."""
        if self.label == "CORRECT":
            return 1.0
        if self.label == "PARTIAL":
            return 0.5
        return 0.0


def judge_answer(
    llm_call: LLMCallFn,
    question: str,
    gold: str,
    predicted: str,
    mode: str = "graded",
) -> JudgeVerdict:
    """Call an LLM judge to score a predicted answer against the gold.

    mode="mem0" uses the judge prompt from Mem0's published LoCoMo harness
    verbatim (CORRECT/WRONG, JSON output) -- the ecosystem-standard judge;
    use this for publishable runs.
    mode="binary" forces CORRECT/INCORRECT with our own prompt.
    mode="graded" allows CORRECT/PARTIAL/INCORRECT (internal diagnostics).
    A stray PARTIAL in the binary modes is mapped to INCORRECT
    (conservative -- never inflates our own score).
    """
    if mode not in ("graded", "binary", "mem0"):
        raise ValueError(
            f"judge mode must be 'graded', 'binary' or 'mem0', got {mode!r}"
        )
    if mode == "mem0":
        prompt = judge_prompt_for_mode("mem0").format(
            question=question.strip(),
            gold=gold.strip(),
            predicted=(predicted or "").strip() or "(no answer)",
        )
        response = llm_call(prompt)
        return _parse_verdict_mem0(response)
    template = JUDGE_PROMPT_BINARY if mode == "binary" else JUDGE_PROMPT
    prompt = template.format(
        question=question.strip(),
        gold=gold.strip(),
        predicted=(predicted or "").strip() or "(no answer)",
    )
    response = llm_call(prompt)
    verdict = _parse_verdict(response)
    if mode == "binary" and verdict.label == "PARTIAL":
        verdict = JudgeVerdict(
            label="INCORRECT",
            reason=f"(binary mode: PARTIAL demoted) {verdict.reason}",
            raw=verdict.raw,
        )
    return verdict


def _parse_verdict_mem0(response: str) -> JudgeVerdict:
    """Parse the mem0-judge JSON output ({"reasoning": ..., "label":
    "CORRECT"|"WRONG"}). WRONG maps to INCORRECT. Tolerates code fences
    and falls back to a label regex; unparseable output defaults to
    INCORRECT so judge failures never inflate scores."""
    import json as _json

    text = (response or "").strip()
    body = text
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", body).strip()
    label = None
    reason = ""
    try:
        obj = _json.loads(body)
        if isinstance(obj, dict):
            label = str(obj.get("label", "")).strip().upper() or None
            reason = str(obj.get("reasoning", "")).strip()
    except (ValueError, TypeError):
        pass
    if label is None:
        m = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', text, re.IGNORECASE)
        if m:
            label = m.group(1).upper()
        else:
            m = re.search(r"\b(CORRECT|WRONG)\b", text, re.IGNORECASE)
            label = m.group(1).upper() if m else None
    if label == "WRONG":
        label = "INCORRECT"
    if label not in ("CORRECT", "INCORRECT"):
        label = "INCORRECT"
        reason = reason or f"unparseable judge output: {text[:150]}"
    return JudgeVerdict(label=label, reason=reason or text[:200], raw=text)


def _parse_verdict(response: str) -> JudgeVerdict:
    """Parse the judge's response into a structured verdict.

    Looks for the labels CORRECT / INCORRECT / PARTIAL as a standalone word.
    First match wins. If none found, defaults to INCORRECT with the full
    response as the reason (so we surface bad judge output rather than
    silently counting as correct).
    """
    text = response.strip()
    # Prefer line-anchored matches; fall back to any standalone occurrence.
    # Track whether we found via the line-anchored path so we use the same
    # criterion to extract the reason (avoid the label/reason mismatch where
    # "The answer is CORRECT because..." finds the label inline but reason
    # extraction looked for line-start matches and missed it).
    label = None
    label_anchored = False
    for candidate in ("CORRECT", "INCORRECT", "PARTIAL"):
        if re.search(rf"^\s*{candidate}\b", text, flags=re.MULTILINE):
            label = candidate
            label_anchored = True
            break
    if label is None:
        for candidate in ("INCORRECT", "PARTIAL", "CORRECT"):
            if re.search(rf"\b{candidate}\b", text, flags=re.IGNORECASE):
                label = candidate
                break
    if label is None:
        label = "INCORRECT"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    reason = ""
    if label_anchored:
        for i, line in enumerate(lines):
            if line.upper().startswith(label):
                if i + 1 < len(lines):
                    reason = lines[i + 1]
                break
    else:
        # Inline-found label: use the line that contains it as the reason.
        for line in lines:
            if re.search(rf"\b{label}\b", line, flags=re.IGNORECASE):
                reason = line
                break
    if not reason:
        reason = text[:200]
    return JudgeVerdict(label=label, reason=reason, raw=text)


# ---------- optional: sync + async adapters for common SDKs ----------

def anthropic_judge(
    client,  # type: ignore[no-untyped-def]
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 256,
) -> LLMCallFn:
    """Return an LLMCallFn backed by an Anthropic client for judging.

    Usage:
        from anthropic import Anthropic
        judge = anthropic_judge(Anthropic(), model="claude-haiku-4-5-20251001")
    """
    def call(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Defensive bounds check: malformed/empty API responses shouldn't crash
        # the eval. Return empty string and let parse_verdict handle it (which
        # safely defaults to INCORRECT).
        if not getattr(resp, "content", None):
            return ""
        first = resp.content[0]
        return getattr(first, "text", "") or ""
    return call


def openai_judge(
    client,  # type: ignore[no-untyped-def]
    model: str = "gpt-4o-mini",
    max_tokens: int = 256,
) -> LLMCallFn:
    """Return an LLMCallFn backed by an OpenAI client for judging.

    Usage:
        from openai import OpenAI
        judge = openai_judge(OpenAI(), model="gpt-4o-mini")
    """
    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if not getattr(resp, "choices", None):
            return ""
        msg = getattr(resp.choices[0], "message", None)
        if msg is None:
            return ""
        return getattr(msg, "content", "") or ""
    return call


__all__ = [
    "JUDGE_PROMPT",
    "JUDGE_PROMPT_BINARY",
    "JUDGE_PROMPT_MEM0",
    "JUDGE_SYSTEM_PROMPT_MEM0",
    "judge_prompt_for_mode",
    "preprocess_gold_mem0",
    "JudgeVerdict",
    "judge_answer",
    "anthropic_judge",
    "openai_judge",
]
