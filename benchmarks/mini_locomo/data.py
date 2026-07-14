"""Synthetic mini-LoCoMo dataset: one Alice conversation that evolves over time.

Designed to exercise GENOME's 5 fixes:
- Fact extraction from conversational input (Fix 1)
- Conflict resolution as Alice moves Tokyo -> Berlin (Fix 2)
- Hybrid keyword search on city names (Fix 3)
- Auto-temporal extraction of location/employer facts (Fix 5)
- (Auto-consolidation needs >threshold memories - covered by unit tests)

Questions span the 5 LoCoMo categories: single-hop, multi-hop, open-domain,
temporal, adversarial.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    speaker: str  # "user" or "assistant"
    text: str
    session: int  # 1-indexed session number
    timestamp_iso: str  # ISO date for temporal grounding


@dataclass(frozen=True)
class Question:
    qid: str
    category: str  # single_hop / multi_hop / open_domain / temporal / adversarial
    text: str
    gold_answer: str
    gold_keywords: list[str]  # acceptable answer tokens for F1 / EM


# Conversation: Alice tells the assistant about her life. Facts evolve.
CONVERSATION: list[Turn] = [
    # Session 1 - January 2026: Alice introduces herself
    Turn(
        speaker="user",
        text="Hi! I'm Alice. I live in Tokyo and work at Google as a software engineer.",
        session=1,
        timestamp_iso="2026-01-15",
    ),
    Turn(
        speaker="user",
        text="My sister Maya also lives in Tokyo. She's a graphic designer at a startup called Pixel.",
        session=1,
        timestamp_iso="2026-01-15",
    ),
    Turn(
        speaker="user",
        text="I love pour-over coffee, especially Ethiopian beans. I drink it every morning.",
        session=1,
        timestamp_iso="2026-01-15",
    ),
    # Session 2 - March 2026: The big move
    Turn(
        speaker="user",
        text="Big news: I just moved to Berlin last week for a new role at Spotify. Started as a senior engineer.",
        session=2,
        timestamp_iso="2026-03-22",
    ),
    Turn(
        speaker="user",
        text="The Berlin apartment is in Kreuzberg. Maya is still in Tokyo, we video-call weekly.",
        session=2,
        timestamp_iso="2026-03-22",
    ),
    Turn(
        speaker="user",
        text="I met a new friend Klaus at a coffee shop. He's a chef from Munich who moved here last year.",
        session=2,
        timestamp_iso="2026-03-22",
    ),
]


QUESTIONS: list[Question] = [
    Question(
        qid="q1_single",
        category="single_hop",
        text="Where does Alice currently live?",
        gold_answer="Berlin",
        gold_keywords=["berlin"],
    ),
    Question(
        qid="q2_multi",
        category="multi_hop",
        text="What city does Alice's sister live in, and what is her sister's job?",
        gold_answer="Tokyo, graphic designer",
        gold_keywords=["tokyo", "graphic", "designer"],
    ),
    Question(
        qid="q3_open",
        category="open_domain",
        text="Tell me about Klaus.",
        gold_answer="Klaus is a chef from Munich, Alice's new friend in Berlin",
        gold_keywords=["chef", "munich", "friend"],
    ),
    Question(
        qid="q4_temporal",
        category="temporal",
        text="Where was Alice living in February 2026?",
        gold_answer="Tokyo",
        gold_keywords=["tokyo"],
    ),
    Question(
        qid="q5_adversarial",
        category="adversarial",
        text="Does Alice work at Microsoft?",
        gold_answer="No, Alice works at Spotify (and previously at Google), not Microsoft",
        gold_keywords=["no", "spotify", "google"],
    ),
]
