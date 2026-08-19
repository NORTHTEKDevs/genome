# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Multi-agent belief attribution: who believes what, since when.

Agents sharing a store must not clobber each other's beliefs: agent B recording a
different value must not close agent A's fact. Conflicts between believers are
surfaced, not silently resolved.
"""

from __future__ import annotations

import numpy as np
import pytest

from genome import Memory
from genome.memory.entities import ENTITY_OPERATOR
from genome.memory.schema import MemoryRecord
from genome.memory.temporal import belief_conflicts, facts_believed_by


@pytest.fixture
def mem_entity():
    m = Memory(storage=":memory:")
    ent = MemoryRecord(
        content="Jordan",
        embedding=np.asarray(m.embed.encode("Jordan"), dtype=np.float32),
        user_id="u1",
        operator=ENTITY_OPERATOR,
        metadata={"entity_type": "PERSON", "entity_name": "Jordan"},
    )
    m.store.add(ent)
    yield m, ent.id
    m.close()


def test_believed_by_is_recorded(mem_entity):
    m, eid = mem_entity
    fact = m.record_fact(
        eid, "location", "Seattle", valid_from=1700000000.0, believed_by="agent-a"
    )
    assert fact.believed_by == "agent-a"


def test_different_believers_do_not_clobber_each_other(mem_entity):
    m, eid = mem_entity
    m.record_fact(eid, "location", "Seattle", valid_from=1700000000.0,
                  believed_by="agent-a")
    m.record_fact(eid, "location", "Austin", valid_from=1700000100.0,
                  believed_by="agent-b")

    current = [f for f in m.current_facts(eid, user_id="u1")
               if f.fact_type == "location"]
    assert len(current) == 2, "each believer keeps its own current fact"

    a_view = facts_believed_by(m, eid, "agent-a")
    assert [f.value for f in a_view] == ["Seattle"]


def test_same_believer_supersedes_its_own_belief(mem_entity):
    m, eid = mem_entity
    m.record_fact(eid, "location", "Seattle", valid_from=1700000000.0,
                  believed_by="agent-a")
    m.record_fact(eid, "location", "Austin", valid_from=1735000000.0,
                  believed_by="agent-a")
    a_current = [f for f in facts_believed_by(m, eid, "agent-a") if f.is_current]
    assert [f.value for f in a_current] == ["Austin"]


def test_unattributed_facts_stay_isolated_from_believers(mem_entity):
    m, eid = mem_entity
    m.record_fact(eid, "location", "Anchorage", valid_from=1700000000.0)
    m.record_fact(eid, "location", "Austin", valid_from=1735000000.0,
                  believed_by="agent-a")
    unattributed = [
        f for f in m.current_facts(eid, user_id="u1")
        if f.fact_type == "location" and f.believed_by is None
    ]
    assert [f.value for f in unattributed] == ["Anchorage"], (
        "an agent's belief must not close the shared unattributed fact"
    )


def test_belief_conflicts_surfaces_disagreement(mem_entity):
    m, eid = mem_entity
    m.record_fact(eid, "location", "Seattle", valid_from=1700000000.0,
                  believed_by="agent-a")
    m.record_fact(eid, "location", "Austin", valid_from=1700000100.0,
                  believed_by="agent-b")
    m.record_fact(eid, "employer", "Northtek", valid_from=1700000000.0,
                  believed_by="agent-a")
    m.record_fact(eid, "employer", "Northtek", valid_from=1700000100.0,
                  believed_by="agent-b")

    conflicts = belief_conflicts(m, eid)
    assert len(conflicts) == 1, "agreement on employer is not a conflict"
    conflict = conflicts[0]
    assert conflict.fact_type == "location"
    assert {(v.believed_by, v.value) for v in conflict.positions} == {
        ("agent-a", "Seattle"),
        ("agent-b", "Austin"),
    }
