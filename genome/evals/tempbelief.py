"""TempBelief: a temporal-contradiction benchmark where facts CHANGE over time and
the agent must answer with the current truth, a point-in-time (as-of) truth, or the
full history.

Deterministic, seeded, template-authored (NOT LLM-authored) so gold labels are
exactly known and reproducible. The ground-truth FactEvent log is the SOLE source of
truth for both the rendered NL turns and every gold answer -- never derived from what
any system extracts (no circularity, no oracle).

The dataset is designed to separate belief-state modeling from retrieval/capacity:
  - Facts update over time (Boston -> Seattle -> Austin), so overwrite-based memory
    (Mem0) structurally loses the history and older values.
  - ~1/3 of fact assertions are revealed OUT OF CHRONOLOGICAL ORDER (a later session
    retro-narrates an earlier-dated fact), so wall-clock ordering is wrong and only
    domain-time validity answers as-of correctly.
  - Distractors: tentative plans and one-off temporary events that must NOT change
    the persistent attribute (false-positive resistance).

Turns are LocomoTurn-compatible (speaker/text/turn_id/dia_id/session/session_datetime)
so the existing FullContextBaseline / Mem0Baseline / dense retrieval run unmodified.

Splits: current-value, as-of (in-range), history; plus an as-of-abstention diagnostic.
"""
from __future__ import annotations

import json
from calendar import timegm
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataclasses (field names mirror LocomoTurn / LocomoQuestion for harness reuse)
# ---------------------------------------------------------------------------
@dataclass
class TBTurn:
    speaker: str
    text: str
    turn_id: str
    dia_id: str
    session: int
    session_datetime: str


@dataclass
class TBQuestion:
    question: str
    answer: str
    category: str          # split name: current-value | as-of | history | as-of-abstention
    question_id: str
    entity: str
    attribute: str


@dataclass
class TBConversation:
    conversation_id: str
    turns: list[TBTurn]
    questions: list[TBQuestion]
    speaker_a: str = "user"
    speaker_b: str = "friend"
    speakers: list[str] = field(default_factory=lambda: ["user", "friend"])


# ---------------------------------------------------------------------------
# Vocabulary -- deliberately MIXES GENOME's 6-type enum with out-of-enum
# attributes (car, favorite_food, gym) to test open-vocabulary generalization.
# ---------------------------------------------------------------------------
_NAMES = ["Jordan", "Maya", "Priya", "Diego", "Lena", "Omar", "Nina", "Theo",
          "Sana", "Marcus", "Yuki", "Ravi", "Clara", "Ivan", "Rosa", "Kofi"]

# attribute -> (question_noun, value pool)
_ATTRS = {
    "location":            ("city",            ["Boston", "Seattle", "Austin", "Denver", "Chicago", "Portland", "Miami", "Atlanta"]),
    "employer":            ("employer",        ["Google", "Stripe", "a startup", "Amazon", "IBM", "a nonprofit", "Netflix", "a bank"]),
    "occupation":          ("job",             ["nurse", "teacher", "designer", "analyst", "chef", "pilot", "lawyer", "barista"]),
    "relationship_status": ("relationship status", ["single", "dating someone", "engaged", "married", "divorced"]),
    "car":                 ("car",             ["a Civic", "a Tesla", "a pickup truck", "a Prius", "a Jeep", "a minivan"]),
    "favorite_food":       ("favorite food",   ["sushi", "tacos", "pho", "pizza", "curry", "ramen", "barbecue"]),
    "gym":                 ("gym",             ["Planet Fitness", "CrossFit", "a yoga studio", "Equinox", "the YMCA"]),
}
_ATTR_KEYS = list(_ATTRS)

# domain-time grid: month indices 0..29 -> (year, month) from 2023-01
def _grid_date(idx: int) -> tuple[int, int]:
    y = 2023 + idx // 12
    m = 1 + idx % 12
    return y, m

def _epoch(idx: int) -> float:
    y, m = _grid_date(idx)
    return float(timegm((y, m, 1, 0, 0, 0, 0, 0, 0)))

_MONTHNAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

def _month_year(idx: int) -> str:
    y, m = _grid_date(idx)
    return f"{_MONTHNAMES[m - 1]} {y}"


@dataclass
class FactEvent:
    entity: str
    attribute: str
    value: str
    from_idx: int          # domain-time month index when it became true
    until_idx: int | None  # exclusive; None = still current


# ---------------------------------------------------------------------------
# Assertion phrasing (templated, varied; domain date embedded in text)
# ---------------------------------------------------------------------------
def _relative_phrase(fact_idx: int, narr_idx: int) -> str:
    """Phrase a fact's domain date RELATIVE to the narration month (no verbatim date).
    The LLM resolves these against the message's own timestamp (which the harness
    prefixes to every turn), so this tests real-language relative-time handling."""
    d = max(0, narr_idx - fact_idx)  # months between when-true and when-said
    # Resolvable relative phrasing (precise arithmetic from the message date),
    # not vague ("about two years ago") -- the eval measures whether the system
    # resolves relative dates, not whether it guesses inherently-coarse ones.
    if d == 0:
        return "just now"
    if d == 1:
        return "last month"
    if d < 24:
        return f"{d} months ago"
    y, m = d // 12, d % 12
    return f"{y} years ago" if m == 0 else f"{y} years and {m} months ago"


def _rel_assert_text(ev: FactEvent, narr_idx: int) -> str:
    phrase = _relative_phrase(ev.from_idx, narr_idx)
    e, v = ev.entity, ev.value
    verb = {
        "location": f"moved to {v}", "employer": f"started at {v}",
        "occupation": f"became a {v}", "relationship_status": f"became {v}",
        "car": f"got {v}", "favorite_food": f"got into {v}", "gym": f"joined {v}",
    }[ev.attribute]
    return f"{e} {verb} {phrase}."


def _assert_text(ev: FactEvent, forward: bool) -> str:
    noun, _ = _ATTRS[ev.attribute]
    when = _month_year(ev.from_idx)
    e, v = ev.entity, ev.value
    if ev.attribute == "location":
        base = f"{e} moved to {v}" if forward else f"back in {when}, {e} was actually living in {v}"
    elif ev.attribute == "employer":
        base = f"{e} started working at {v}" if forward else f"back in {when}, {e} was at {v}"
    elif ev.attribute == "occupation":
        base = f"{e} became a {v}" if forward else f"back in {when}, {e} was working as a {v}"
    elif ev.attribute == "relationship_status":
        base = f"{e} is now {v}" if forward else f"back in {when}, {e} was {v}"
    elif ev.attribute == "car":
        base = f"{e} got {v}" if forward else f"back in {when}, {e} was driving {v}"
    elif ev.attribute == "favorite_food":
        base = f"{e}'s favorite food these days is {v}" if forward else f"back in {when}, {e}'s favorite food was {v}"
    else:  # gym
        base = f"{e} joined {v}" if forward else f"back in {when}, {e}'s gym was {v}"
    if forward:
        return f"In {when}, {base}."
    return base[0].upper() + base[1:] + "."


def _distractor_tentative(ev: FactEvent, pool: list[str]) -> str:
    alt = next((x for x in pool if x != ev.value), ev.value)
    noun, _ = _ATTRS[ev.attribute]
    return f"{ev.entity} has been talking about maybe switching {noun} to {alt} at some point, but nothing's decided."

def _distractor_temporary(ev: FactEvent) -> str:
    if ev.attribute == "location":
        return f"{ev.entity} flew to Denver for a work conference last week, just a short trip."
    return f"{ev.entity} tried something different for a day but went right back to their usual."

_FILLER = [
    "The weather has been all over the place lately.",
    "We caught up over coffee for a couple hours.",
    "I've been meaning to call more often, honestly.",
    "Work has been busy but good.",
    "We laughed about old times for a while.",
    "Nothing too exciting on my end this week.",
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def _build_conversation(cid: str, seed: int, relative: bool = False) -> TBConversation:
    # deterministic pseudo-choices from seed (no RNG module -> fully reproducible)
    def pick(lst, k):
        return lst[(seed * 7 + k * 13) % len(lst)]

    n_entities = 3
    entities = []
    used = set()
    for i in range(n_entities):
        nm = _NAMES[(seed * 3 + i * 5) % len(_NAMES)]
        j = 0
        while nm in used:
            j += 1
            nm = _NAMES[(seed * 3 + i * 5 + j) % len(_NAMES)]
        used.add(nm)
        entities.append(nm)

    events: list[FactEvent] = []
    for i, ent in enumerate(entities):
        # 2 distinct attributes per entity
        a1 = _ATTR_KEYS[(seed + i * 2) % len(_ATTR_KEYS)]
        a2 = _ATTR_KEYS[(seed + i * 2 + 3) % len(_ATTR_KEYS)]
        for a in dict.fromkeys([a1, a2]):
            _, pool = _ATTRS[a]
            # 3 values, 3 domain dates strictly increasing across the 30-month grid
            base = (seed + i * 4 + hash(a) % 5) % 3           # start offset
            idxs = [3 + base + i, 12 + base + i, 22 + base + i]  # spread across grid
            vals = [pool[(seed + i + a_off) % len(pool)] for a_off in (1, 4, 7)]
            # ensure distinct consecutive values
            if vals[1] == vals[0]:
                vals[1] = pool[(pool.index(vals[1]) + 1) % len(pool)]
            if vals[2] in (vals[0], vals[1]):
                vals[2] = pool[(pool.index(vals[2]) + 2) % len(pool)]
            for k, (v, fi) in enumerate(zip(vals, idxs)):
                until = idxs[k + 1] if k + 1 < len(idxs) else None
                events.append(FactEvent(ent, a, v, fi, until))

    # ---- render turns across 8 sessions ----
    # session s narrates at wall-clock month index 2 + s*3 (so narration order is
    # chronological by SESSION, but ~1/3 of facts are retro-narrated out of order).
    turns: list[TBTurn] = []
    tid = 0
    n_sessions = 8

    def add(session, text, speaker="user"):
        nonlocal tid
        y, m = _grid_date(2 + session * 3)
        turns.append(TBTurn(
            speaker=speaker, text=text,
            turn_id=f"{cid}:t{tid}", dia_id=f"D{session+1}:{tid}",
            session=session + 1,
            session_datetime=f"{_MONTHNAMES[m-1]} {y}"))
        tid += 1

    # assign each event to a narration session. Forward assertions go in the
    # session covering their domain date; ~1/3 are pushed to a LATER session
    # (out-of-order retro-narration) to defeat wall-clock ordering.
    for ei, ev in enumerate(events):
        # session whose narration month >= domain month (forward)
        fwd_session = min(n_sessions - 1, max(0, (ev.from_idx - 2) // 3))
        out_of_order = (ei % 3 == 0) and fwd_session < n_sessions - 1
        sess = (n_sessions - 1) if out_of_order else fwd_session
        if relative:
            add(sess, _rel_assert_text(ev, 2 + sess * 3))   # relative phrasing
        else:
            add(sess, _assert_text(ev, forward=not out_of_order))
        # sprinkle distractors near some events
        if ei % 4 == 1:
            _, pool = _ATTRS[ev.attribute]
            add(min(n_sessions - 1, fwd_session), _distractor_tentative(ev, pool))
        if ei % 5 == 2 and ev.attribute == "location":
            add(min(n_sessions - 1, fwd_session), _distractor_temporary(ev))

    # filler chit-chat interleaved (~1:1 with fact turns), deterministic
    fact_turns = len(turns)
    for f in range(fact_turns):
        s = f % n_sessions
        add(s, _FILLER[(seed + f) % len(_FILLER)], speaker="friend")

    # stable sort by session preserves insertion order within a session
    turns.sort(key=lambda t: t.session)

    # ---- questions + gold from the event log ----
    questions: list[TBQuestion] = []
    qid = 0
    # index events by (entity, attribute) sorted by from_idx
    slots: dict[tuple[str, str], list[FactEvent]] = {}
    for ev in events:
        slots.setdefault((ev.entity, ev.attribute), []).append(ev)
    for seq in slots.values():
        seq.sort(key=lambda e: e.from_idx)

    for (ent, attr), seq in slots.items():
        noun, _ = _ATTRS[attr]
        # current-value
        questions.append(TBQuestion(
            question=f"What is {ent}'s current {noun}?",
            answer=seq[-1].value, category="current-value",
            question_id=f"{cid}:q{qid}", entity=ent, attribute=attr)); qid += 1
        # as-of: one per completed interval (query at midpoint of the interval)
        for k in range(len(seq)):
            start = seq[k].from_idx
            end = seq[k].until_idx if seq[k].until_idx is not None else start + 6
            mid = (start + end) // 2
            if mid <= start:
                mid = start
            questions.append(TBQuestion(
                question=f"What was {ent}'s {noun} in {_month_year(mid)}?",
                answer=seq[k].value, category="as-of",
                question_id=f"{cid}:q{qid}", entity=ent, attribute=attr)); qid += 1
        # history
        hist = "; ".join(f"{e.value} (from {_month_year(e.from_idx)})" for e in seq)
        questions.append(TBQuestion(
            question=f"List every {noun} {ent} has had over time, in order.",
            answer=hist, category="history",
            question_id=f"{cid}:q{qid}", entity=ent, attribute=attr)); qid += 1
        # as-of-abstention: a date strictly BEFORE the first known fact
        before = seq[0].from_idx - 6
        if before >= 0:
            questions.append(TBQuestion(
                question=f"What was {ent}'s {noun} in {_month_year(before)}?",
                answer="no information available", category="as-of-abstention",
                question_id=f"{cid}:q{qid}", entity=ent, attribute=attr)); qid += 1

    return TBConversation(conversation_id=cid, turns=turns, questions=questions)


def generate(n_conversations: int = 12, relative: bool = False) -> list[TBConversation]:
    return [_build_conversation(f"tb-{i:02d}", seed=i + 1, relative=relative)
            for i in range(n_conversations)]


def to_json(convs: list[TBConversation]) -> list[dict]:
    return [asdict(c) for c in convs]


def load(path: str | Path) -> list[TBConversation]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for c in data:
        turns = [TBTurn(**t) for t in c["turns"]]
        qs = [TBQuestion(**q) for q in c["questions"]]
        out.append(TBConversation(conversation_id=c["conversation_id"], turns=turns,
                                  questions=qs, speaker_a=c.get("speaker_a", "user"),
                                  speaker_b=c.get("speaker_b", "friend"),
                                  speakers=c.get("speakers", ["user", "friend"])))
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rel = "--relative" in sys.argv
    convs = generate(n, relative=rel)
    outp = Path("benchmarks/data/tempbelief_rel.json" if rel else "benchmarks/data/tempbelief.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(to_json(convs), indent=1), encoding="utf-8")
    nt = sum(len(c.turns) for c in convs)
    nq = sum(len(c.questions) for c in convs)
    from collections import Counter
    cats = Counter(q.category for c in convs for q in c.questions)
    print(f"wrote {outp}: {len(convs)} convs, {nt} turns, {nq} questions")
    print("splits:", dict(cats))
