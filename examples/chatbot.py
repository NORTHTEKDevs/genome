"""End-to-end chatbot example using genome's memory layer.

Run with:
    python examples/chatbot.py                     # uses identity extractor, no LLM required
    python examples/chatbot.py --llm anthropic     # uses Claude for fact extraction (needs ANTHROPIC_API_KEY)

What it demonstrates:

    1. Adding facts to memory (atomic or LLM-extracted).
    2. Retrieving relevant memories for a query, with automatic parent filtering.
    3. Synthesizing a hybrid memory from multiple related memories.
    4. Persistence across runs via SQLite.
    5. Consolidation with synthesis-before-prune.

The chatbot is intentionally simple: it echoes what it remembers about you. Real
agents would feed the retrieved memories into their system prompt or context.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python examples/chatbot.py` without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genome import Memory  # noqa: E402


def make_claude_llm():
    """Return an LLMCallFn that calls Claude. Requires anthropic SDK + ANTHROPIC_API_KEY."""
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError(
            "Anthropic SDK not installed. Run: pip install anthropic"
        ) from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable.")
    client = Anthropic()

    def call(prompt: str) -> str:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text  # type: ignore[no-any-return]

    return call


def print_header(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def demo(mem: Memory, user_id: str) -> None:
    print_header("1. Adding facts")
    samples = [
        "I love pour-over coffee, especially Ethiopian Yirgacheffe.",
        "I just moved to Tokyo last month.",
        "I work as a data scientist at a fintech startup.",
        "I'm learning to play Go.",
        "I prefer Japanese jazz from the 70s.",
    ]
    for text in samples:
        added = mem.add(text, user_id=user_id)
        for a in added:
            print(f"  + {a.content}  (id={a.id})")

    print_header(f"2. Search ({mem.count(user_id=user_id)} memories stored)")
    queries = [
        "what drinks does the user like?",
        "where does the user live?",
        "what does the user do for work?",
    ]
    for q in queries:
        print(f"\n  Q: {q}")
        for r in mem.search(q, user_id=user_id, limit=2):
            print(f"    -> {r.content!r}  (score={r.score:.3f})")

    print_header("3. Synthesize a hybrid memory")
    print("  Picking the top-3 most similar to 'user lifestyle':")
    ids = [r.id for r in mem.search("user lifestyle", user_id=user_id, limit=3)]
    for i in ids:
        rec = mem.get(i)
        if rec:
            print(f"    parent: {rec.content!r}")
    hybrid = mem.synthesize(
        memory_ids=ids,
        user_id=user_id,
        operator="uniform_crossover_with_mutation",
        seed=42,
    )
    print(f"\n  Hybrid created: {hybrid.id}")
    print(f"    content: {hybrid.content}")
    print(f"    operator: {hybrid.operator}")
    print(f"    parents: {hybrid.parents}")

    print_header("4. Search with parent filtering (default ON)")
    print("  Parents are automatically excluded -- the hybrid surfaces:")
    results = mem.search("user lifestyle summary", user_id=user_id, limit=3)
    for r in results:
        marker = " <- synthesized" if r.record.is_synthesized else ""
        print(f"    -> {r.content!r}{marker}")

    print_header("5. Consolidate with synthesis-before-prune")
    print(f"  Before: {mem.count(user_id=user_id)} memories")
    result = mem.consolidate(
        user_id=user_id, max_memories=3, synthesize_before_prune=True
    )
    print(
        f"  After: kept={result.kept}, pruned={result.pruned}, "
        f"synthesized={result.synthesized}"
    )
    print("  Surviving memories:")
    for r in mem.list_all(user_id=user_id):
        marker = " <- synthesized" if r.is_synthesized else ""
        print(f"    {r.content!r}{marker}")


def main() -> int:
    parser = argparse.ArgumentParser(description="GENOME memory layer demo chatbot")
    parser.add_argument(
        "--llm",
        choices=["none", "anthropic"],
        default="none",
        help="LLM to use for fact extraction. 'none' = identity (each input = 1 fact).",
    )
    parser.add_argument(
        "--storage",
        default=":memory:",
        help="SQLite path. Default is in-memory (fresh per run).",
    )
    parser.add_argument("--user-id", default="alice")
    args = parser.parse_args()

    llm_call = make_claude_llm() if args.llm == "anthropic" else None

    print_header("GENOME Memory Layer Demo")
    print(f"  storage: {args.storage}")
    print(f"  llm: {args.llm}")
    print(f"  user_id: {args.user_id}")

    with Memory(storage=args.storage, llm_call=llm_call) as mem:
        demo(mem, user_id=args.user_id)

    print("\nDone. Re-run with --storage path/to/memory.db to persist across runs.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
