"""Multi-user memory example.

Shows how to operate genome as a shared-infrastructure memory layer where each
user has a fully isolated memory scope. Useful for SaaS / chatbot products where
memory must never leak between customers.

Run:
    python examples/multi_user.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genome import Memory  # noqa: E402


USERS = {
    "alice": [
        "Alice loves hiking in the Pacific Northwest",
        "Alice works as a civil engineer in Seattle",
        "Alice drinks tea, not coffee",
    ],
    "bob": [
        "Bob runs marathons",
        "Bob works as a neurosurgeon in Boston",
        "Bob is allergic to tree nuts",
    ],
    "carlos": [
        "Carlos is a sommelier in Mexico City",
        "Carlos owns a natural wine bar",
        "Carlos speaks five languages",
    ],
}


def main() -> int:
    print("=" * 60)
    print("GENOME Multi-User Memory Example")
    print("=" * 60)

    with Memory() as mem:
        # Ingest for all users
        for user, facts in USERS.items():
            print(f"\nIngesting {len(facts)} facts for {user}...")
            for f in facts:
                mem.add(f, user_id=user)

        # Same query, different users -> different answers
        query = "What does the user do for work?"
        print(f"\n\nQuery across all users: {query!r}")
        for user in USERS:
            print(f"\n  -- {user}'s memories --")
            results = mem.search(query, user_id=user, limit=2)
            for r in results:
                print(f"    {r.content}  (score={r.score:.3f})")

        # Verify isolation: Alice's memories are never leaked to Bob
        print("\n\nIsolation check: searching 'hiking' in each user's scope")
        for user in USERS:
            results = mem.search("hiking", user_id=user, limit=1)
            if results:
                print(f"  {user}: {results[0].content!r}")
            else:
                print(f"  {user}: (no relevant memories)")

        # Total size check
        total = mem.count()
        per_user = {u: mem.count(user_id=u) for u in USERS}
        print("\n\nStorage summary:")
        print(f"  total memories: {total}")
        for u, c in per_user.items():
            print(f"  {u}: {c}")

        # Reset one user only
        print("\nResetting Carlos's memories...")
        deleted = mem.reset(user_id="carlos")
        print(f"  deleted {deleted}")
        print(f"  alice still has {mem.count(user_id='alice')} memories")
        print(f"  bob still has {mem.count(user_id='bob')} memories")
        print(f"  carlos: {mem.count(user_id='carlos')} memories")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
