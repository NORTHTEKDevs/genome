# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the round-6 adversarial findings.

Both were cross-tenant leaks in the open core, found by attacking subsystems
(the response cache, consolidation) that earlier rounds never touched.
"""

import genome
from genome.memory.cache import ResponseCache


class TestCacheKeyCollision:
    """A "|"-joined cache key let one tenant read another's cached results."""

    def test_delimiter_in_user_id_does_not_collide(self):
        c = ResponseCache()
        assert c._key("q", "a|b", "c", 5, False, 0) != c._key("q", "a", "b|c", 5, False, 0)

    def test_delimiter_in_query_does_not_collide(self):
        c = ResponseCache()
        assert c._key("x|y", None, "c", 5, False, 0) != c._key("x", "y", "c", 5, False, 0)

    def test_none_scope_distinct_from_literal_none_string(self):
        c = ResponseCache()
        assert c._key("q", None, "c", 5, False, 0) != c._key("q", "None", "c", 5, False, 0)

    def test_mode_boundary_does_not_collide(self):
        c = ResponseCache()
        assert c._key("q", "u", "a|dense", 5, False, 0, "x") != c._key(
            "q", "u", "a", 5, False, 0, "dense|x"
        )

    def test_cached_results_do_not_cross_tenants(self):
        """End-to-end: the real facade must not serve tenant A's hit to tenant B."""
        mem = genome.Memory(storage=":memory:", enable_cache=True)
        mem.add("VICTIM SECRET: the password is hunter2", user_id="a|b", agent_id="c")
        victim = [r.content for r in mem.search("secret password", user_id="a|b", agent_id="c")]
        assert any("VICTIM SECRET" in c for c in victim)

        mem.add("attacker's own unrelated note", user_id="a", agent_id="b|c")
        attacker = [
            r.content for r in mem.search("secret password", user_id="a", agent_id="b|c")
        ]
        assert not any("VICTIM SECRET" in c for c in attacker)

    def test_cache_still_hits_for_a_repeated_query(self):
        """Negative control: the fix must not simply disable caching."""
        mem = genome.Memory(storage=":memory:", enable_cache=True)
        mem.add("the sky is blue", user_id="u", agent_id="a")
        mem.search("sky", user_id="u", agent_id="a")
        before = mem._cache.stats.hits
        mem.search("sky", user_id="u", agent_id="a")
        assert mem._cache.stats.hits == before + 1


class TestConsolidateScopeIsolation:
    """Unscoped consolidate() spliced two tenants' plaintext into one hybrid."""

    def _mixed_tenants(self):
        mem = genome.Memory(storage=":memory:")
        for i in range(6):
            mem.add(f"tenantA filler memory number {i}", user_id="tenant-a", agent_id="agentA")
        mem.add(
            "tenantB SECRET: quarterly revenue is $4.2M",
            user_id="tenant-b",
            agent_id="agentB",
        )
        return mem

    def test_hybrid_never_merges_two_tenants(self):
        mem = self._mixed_tenants()
        mem.consolidate(
            max_memories=3,
            synthesize_before_prune=True,
            synthesis_operator="uniform_crossover",
        )
        for r in mem.list_all():
            if r.metadata.get("consolidation"):
                assert not ("tenantA" in r.content and "tenantB" in r.content)

    def test_hybrid_stays_in_its_own_scope(self):
        """An unscoped sweep must not relocate tenant content to the null scope."""
        mem = self._mixed_tenants()
        mem.consolidate(
            max_memories=3,
            synthesize_before_prune=True,
            synthesis_operator="uniform_crossover",
        )
        for r in mem.list_all():
            if r.metadata.get("consolidation"):
                assert (r.user_id, r.agent_id) in {
                    ("tenant-a", "agentA"),
                    ("tenant-b", "agentB"),
                }

    def test_synthesis_still_happens_within_one_scope(self):
        """Negative control: grouping must not stop same-scope synthesis."""
        mem = genome.Memory(storage=":memory:")
        for i in range(6):
            mem.add(f"filler memory number {i}", user_id="solo", agent_id="agent")
        result = mem.consolidate(
            max_memories=2,
            user_id="solo",
            agent_id="agent",
            synthesize_before_prune=True,
            synthesis_operator="uniform_crossover",
        )
        assert result.synthesized >= 1
