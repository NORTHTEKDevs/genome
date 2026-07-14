"""RAG pipeline example.

Shows genome used as the memory backbone of a retrieval-augmented generation
pipeline. Documents are ingested as facts, queries retrieve relevant context,
and a (mocked) LLM answers using the context.

Run:
    python examples/rag_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genome import Memory  # noqa: E402


DOCS = {
    "handbook": """
        Our company observes federal holidays plus two floating days per year.
        Parental leave is 16 weeks paid for primary caregivers, 4 weeks for secondary.
        Health insurance is provided by Aetna and becomes effective on day one.
        401k matches 100% of contributions up to 6% of salary.
        Annual performance reviews happen in March and September.
    """,
    "engineering_guide": """
        All production services must have structured logging via structlog.
        Deployments go through GitHub Actions and require two approving reviews.
        On-call rotation is one week per engineer, rotating Mondays.
        Critical incidents pager threshold: 5-minute response.
        We use Terraform for all infrastructure changes.
    """,
    "security_policy": """
        Multi-factor authentication is required for all company accounts.
        Customer data must never leave the production VPC.
        Secrets are stored in AWS Secrets Manager, never in code.
        Security incidents must be reported within 24 hours.
        Laptop encryption is mandatory for all engineers.
    """,
}


def ingest_docs(memory: Memory) -> None:
    """Split each doc into sentences (one per memory) scoped by doc name as agent_id."""
    for doc_name, text in DOCS.items():
        for sentence in text.strip().splitlines():
            s = sentence.strip()
            if s:
                memory.add(
                    s, user_id="company", agent_id=doc_name,
                    metadata={"source": doc_name},
                )


def retrieve_context(memory: Memory, query: str, k: int = 3) -> list[str]:
    results = memory.search(query, user_id="company", limit=k)
    return [f"[{r.record.agent_id}] {r.content}" for r in results]


def mock_llm(prompt: str) -> str:
    """Placeholder for the answerer LLM. Replace with your Claude/OpenAI call."""
    return f"(answer would be generated from the prompt below)\n---\n{prompt[:500]}..."


def answer(query: str, memory: Memory) -> str:
    context = retrieve_context(memory, query, k=4)
    prompt = (
        f"Answer the user's question using ONLY the context below.\n"
        f"If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n" + "\n".join(context) + f"\n\nQuestion: {query}\n\nAnswer:"
    )
    return mock_llm(prompt)


def main() -> int:
    print("=" * 60)
    print("GENOME RAG Pipeline Example")
    print("=" * 60)

    with Memory() as mem:
        print(f"\nIngesting {len(DOCS)} documents...")
        ingest_docs(mem)
        print(f"Stored {mem.count(user_id='company')} facts across {len(DOCS)} docs")

        queries = [
            "How many weeks of parental leave do we offer?",
            "What's the process for deploying to production?",
            "How do we handle secrets?",
            "What happens in September?",
        ]

        for q in queries:
            print(f"\nQuery: {q}")
            print("-" * 60)
            retrieved = retrieve_context(mem, q, k=3)
            for r in retrieved:
                print(f"  - {r}")
            print(f"\n  Answer: {answer(q, mem)[:200]}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
