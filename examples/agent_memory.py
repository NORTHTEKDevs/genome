"""Agent memory example.

Shows a long-running agent that:
1. Remembers user preferences across sessions
2. Uses graph relations to mark superseded facts (beliefs change over time)
3. Consolidates memories periodically so the store doesn't grow unbounded
4. Synthesizes hybrid memories that capture general patterns

Run:
    python examples/agent_memory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genome import Memory, SUPERSEDES  # noqa: E402


def session_1(mem: Memory, user: str) -> None:
    """Initial session: user tells agent preferences."""
    print("\n=== Session 1: User introduces themselves ===")
    facts = [
        "I'm learning Go programming",
        "I usually run at 6 AM on weekdays",
        "I love pour-over coffee",
        "I work at a fintech startup",
        "I live in Brooklyn",
    ]
    for f in facts:
        rec = mem.add(f, user_id=user)[0]
        print(f"  + {rec.content}")


def session_2(mem: Memory, user: str) -> None:
    """Later: user's Go interest has waned, new language. Mark the update."""
    print("\n=== Session 2: User's learning changes ===")
    old = mem.search("programming language user is learning", user_id=user, limit=1)[0].record

    new_rec = mem.add("I'm now learning Rust programming", user_id=user)[0]
    print(f"  + {new_rec.content}")

    # Mark the new belief supersedes the old one
    mem.link(new_rec.id, old.id, relation=SUPERSEDES)
    print(f"  (marked {new_rec.id} SUPERSEDES {old.id})")


def session_3(mem: Memory, user: str) -> None:
    """User adds more context."""
    print("\n=== Session 3: Additional context ===")
    facts = [
        "My morning runs are typically 5km",
        "I prefer Ethiopian single-origin beans",
        "I use a V60 pour-over dripper",
        "My team manages payment infrastructure",
        "I bike to work on Thursdays",
    ]
    for f in facts:
        mem.add(f, user_id=user)
        print(f"  + {f}")


def synthesize_lifestyle_summary(mem: Memory, user: str) -> None:
    """Recombine a few related memories into a high-level 'lifestyle' hybrid."""
    print("\n=== Synthesis: extract user's lifestyle pattern ===")
    relevant = mem.search("user's routine and habits", user_id=user, limit=3)
    parent_ids = [r.id for r in relevant]
    print("  Parents:")
    for r in relevant:
        print(f"    - {r.content}")

    hybrid = mem.synthesize(
        memory_ids=parent_ids,
        operator="uniform_crossover_with_mutation",
        user_id=user,
        content="user's lifestyle: early-morning athletic coffee enthusiast in tech",
        seed=42,
    )
    print(f"  Hybrid: {hybrid.content}")
    print(f"  (id={hybrid.id}, operator={hybrid.operator})")


def demonstrate_retrieval_with_supersedes(mem: Memory, user: str) -> None:
    """Show how superseded facts can be filtered out at query time."""
    print("\n=== Retrieval: what language is the user learning? ===")
    # Naive search: might return both old and new
    all_results = mem.search("programming language", user_id=user, limit=5)
    print("  All matches:")
    for r in all_results:
        print(f"    - {r.content}  (score={r.score:.3f})")

    # Filter out superseded ones
    superseded_ids = set()
    for r in all_results:
        for edge in mem.edges_of(r.id, relation=SUPERSEDES, direction="in"):
            superseded_ids.add(edge.to_id)
    active_results = [r for r in all_results if r.id not in superseded_ids]
    print("\n  Active beliefs (superseded filtered out):")
    for r in active_results:
        print(f"    - {r.content}")


def consolidate(mem: Memory, user: str) -> None:
    """Prune the store, preserving high-fitness memories, hybridizing low-fitness pairs."""
    print("\n=== Consolidation ===")
    before = mem.count(user_id=user)
    result = mem.consolidate(
        user_id=user,
        max_memories=5,
        synthesize_before_prune=True,
    )
    print(f"  Before: {before}  After kept: {result.kept}")
    print(f"  Pruned: {result.pruned}  Synthesized hybrids: {result.synthesized}")
    print("\n  Final memories:")
    for r in mem.list_all(user_id=user):
        marker = " (hybrid)" if r.is_synthesized else ""
        print(f"    - {r.content}{marker}")


def main() -> int:
    print("=" * 60)
    print("GENOME Agent Memory Example")
    print("=" * 60)

    with Memory() as mem:
        user = "alice"
        session_1(mem, user)
        session_2(mem, user)
        session_3(mem, user)
        synthesize_lifestyle_summary(mem, user)
        demonstrate_retrieval_with_supersedes(mem, user)
        consolidate(mem, user)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
