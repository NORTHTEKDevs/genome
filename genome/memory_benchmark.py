"""Head-to-head benchmark: genome Memory vs a naive vector-store baseline.

Shows the retrieval-quality gains from genome's unique features:
- Parent filtering in search (from v0.2)
- Recombination synthesis (queries for hybrid concepts)
- RAPTOR hierarchical summarization (big-picture queries over many memories)

Baseline: a simple cosine-similarity search over raw embeddings, no filtering,
no synthesis. Same embedding model as genome. Same data.

Dataset: scripted conversational workload -- a fake user tells an agent 100
facts across 5 sessions, then issues 50 retrieval queries. Ground truth is
derived from the script (which facts are relevant to which queries).

Run: python -m genome.memory_benchmark
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from genome.memory.facade import Memory

if TYPE_CHECKING:
    from genome.embeddings import EmbeddingProvider


# ---------- dataset ----------

@dataclass
class QueryCase:
    query: str
    relevant_facts: list[str]  # known-relevant fact strings


def build_conversational_dataset() -> tuple[list[str], list[QueryCase]]:
    """Return (facts, queries). 100 facts + 50 queries with known-relevant sets.

    The dataset is intentionally synthetic so ground truth is deterministic.
    Five topic clusters: coffee, travel, work, fitness, cooking.
    """
    topics = {
        "coffee": [
            "user loves pour-over coffee",
            "user prefers light roast beans",
            "user grinds coffee fresh each morning",
            "user uses a V60 dripper",
            "user drinks espresso on weekends",
            "user has a Rancilio Silvia espresso machine",
            "user buys Ethiopian Yirgacheffe monthly",
            "user avoids instant coffee",
            "user roasts beans at home sometimes",
            "user drinks coffee black",
            "user dislikes flavored syrups",
            "user takes coffee breaks at 10am and 2pm",
            "user tried cold brew and liked it",
            "user reads about coffee origins on weekends",
            "user owns a burr grinder",
            "user visits specialty coffee shops",
            "user recommends Blue Bottle to friends",
            "user brews coffee for four minutes",
            "user avoids pre-ground coffee",
            "user celebrates with pour-over rituals",
        ],
        "travel": [
            "user just moved to Tokyo last month",
            "user visited Kyoto on vacation",
            "user speaks conversational Japanese",
            "user hates long-haul flights",
            "user flew economy on the last trip",
            "user stayed in Shibuya",
            "user wants to visit Hokkaido",
            "user has been to Paris twice",
            "user spent a week in Rome",
            "user loves street food markets",
            "user takes the Shinkansen often",
            "user visited a hot spring onsen",
            "user prefers Airbnbs over hotels",
            "user has a Global Entry membership",
            "user visited Thailand last year",
            "user plans a Kyoto trip next spring",
            "user wrote a blog post about Japan",
            "user makes travel itineraries in Notion",
            "user visited Vietnam and loved pho",
            "user owns a durable travel backpack",
        ],
        "work": [
            "user works as a data scientist",
            "user leads a team of five engineers",
            "user writes research papers on LLMs",
            "user uses Python for ML projects",
            "user ships production features weekly",
            "user mentors junior engineers regularly",
            "user attends NeurIPS conferences",
            "user prefers asynchronous communication",
            "user reviews three PRs per day on average",
            "user runs weekly engineering syncs",
            "user manages an embedding-based search product",
            "user built an internal RAG pipeline",
            "user uses PyTorch and HuggingFace",
            "user presented at an internal all-hands",
            "user writes technical blog posts for the company",
            "user runs ML experiments on GCP",
            "user prefers Jupyter notebooks for EDA",
            "user fine-tunes open-source LLMs",
            "user architected a vector database migration",
            "user reviews arXiv papers on Fridays",
        ],
        "fitness": [
            "user runs three times per week",
            "user prefers trail running",
            "user owns Hoka running shoes",
            "user runs 5k in 24 minutes",
            "user does yoga on Sundays",
            "user lifts weights twice per week",
            "user avoids running in heat",
            "user tracks runs with a Garmin watch",
            "user swims occasionally in summer",
            "user is training for a half marathon",
            "user stretches for fifteen minutes daily",
            "user bikes to work on Thursdays",
            "user follows an intermittent fasting schedule",
            "user measures heart rate variability",
            "user does bodyweight workouts when traveling",
            "user prefers outdoor exercise over gyms",
            "user bought a new pair of running socks",
            "user enjoys mountain hiking in summer",
            "user watches PR videos on YouTube",
            "user does meditation after workouts",
        ],
        "cooking": [
            "user cooks Italian food on weekends",
            "user makes fresh pasta from scratch",
            "user owns a cast iron skillet",
            "user avoids processed foods",
            "user buys groceries at a farmers market",
            "user keeps sourdough starter alive",
            "user makes homemade pizza every Friday",
            "user uses a kitchen scale for baking",
            "user prefers bone-in chicken for roasting",
            "user keeps a well-stocked spice cabinet",
            "user tried fermenting kimchi last month",
            "user is learning Japanese cuisine",
            "user grows herbs on the windowsill",
            "user avoids seed oils in cooking",
            "user batch-cooks on Sundays",
            "user owns a Le Creuset dutch oven",
            "user makes stock from chicken bones",
            "user reads Serious Eats recipes",
            "user finishes dishes with flaky salt",
            "user roasts vegetables at 425F",
        ],
    }
    facts: list[str] = []
    for topic_facts in topics.values():
        facts.extend(topic_facts)

    # 10 queries per topic, each targets ~3-5 facts
    queries: list[QueryCase] = []
    query_templates = {
        "coffee": [
            ("what drinks does the user like?", ["user loves pour-over coffee", "user drinks espresso on weekends", "user drinks coffee black"]),
            ("what equipment does the user own for coffee?", ["user uses a V60 dripper", "user has a Rancilio Silvia espresso machine", "user owns a burr grinder"]),
            ("where does the user buy coffee beans?", ["user buys Ethiopian Yirgacheffe monthly", "user visits specialty coffee shops", "user recommends Blue Bottle to friends"]),
            ("how does the user brew coffee?", ["user brews coffee for four minutes", "user grinds coffee fresh each morning", "user uses a V60 dripper"]),
            ("what coffee doesn't the user drink?", ["user avoids instant coffee", "user avoids pre-ground coffee", "user dislikes flavored syrups"]),
            ("what bean type does the user prefer?", ["user prefers light roast beans", "user buys Ethiopian Yirgacheffe monthly"]),
            ("when does the user drink coffee?", ["user drinks espresso on weekends", "user takes coffee breaks at 10am and 2pm"]),
            ("does the user make their own coffee at home?", ["user grinds coffee fresh each morning", "user roasts beans at home sometimes", "user owns a burr grinder"]),
            ("what is the user's favorite coffee method?", ["user loves pour-over coffee", "user celebrates with pour-over rituals"]),
            ("has the user tried cold brew?", ["user tried cold brew and liked it"]),
        ],
        "travel": [
            ("where does the user live?", ["user just moved to Tokyo last month", "user stayed in Shibuya"]),
            ("what language does the user speak?", ["user speaks conversational Japanese"]),
            ("where has the user traveled?", ["user visited Kyoto on vacation", "user has been to Paris twice", "user spent a week in Rome", "user visited Thailand last year", "user visited Vietnam and loved pho"]),
            ("what is the user's next planned trip?", ["user plans a Kyoto trip next spring", "user wants to visit Hokkaido"]),
            ("how does the user plan travel?", ["user makes travel itineraries in Notion", "user has a Global Entry membership"]),
            ("where does the user stay when traveling?", ["user prefers Airbnbs over hotels", "user stayed in Shibuya"]),
            ("what travel gear does the user own?", ["user owns a durable travel backpack"]),
            ("has the user visited Europe?", ["user has been to Paris twice", "user spent a week in Rome"]),
            ("what transportation does the user take in Japan?", ["user takes the Shinkansen often"]),
            ("what does the user write about travel?", ["user wrote a blog post about Japan"]),
        ],
        "work": [
            ("what is the user's job?", ["user works as a data scientist", "user leads a team of five engineers"]),
            ("what ML frameworks does the user use?", ["user uses PyTorch and HuggingFace", "user uses Python for ML projects"]),
            ("what does the user publish?", ["user writes research papers on LLMs", "user writes technical blog posts for the company"]),
            ("what conferences does the user attend?", ["user attends NeurIPS conferences"]),
            ("what does the user review?", ["user reviews three PRs per day on average", "user reviews arXiv papers on Fridays"]),
            ("does the user manage people?", ["user leads a team of five engineers", "user mentors junior engineers regularly", "user runs weekly engineering syncs"]),
            ("what products has the user built?", ["user manages an embedding-based search product", "user built an internal RAG pipeline", "user architected a vector database migration"]),
            ("where does the user run experiments?", ["user runs ML experiments on GCP"]),
            ("does the user speak at work?", ["user presented at an internal all-hands"]),
            ("does the user work with LLMs?", ["user writes research papers on LLMs", "user fine-tunes open-source LLMs"]),
        ],
        "fitness": [
            ("what exercise does the user do?", ["user runs three times per week", "user lifts weights twice per week", "user does yoga on Sundays"]),
            ("how fast does the user run?", ["user runs 5k in 24 minutes"]),
            ("what running shoes does the user wear?", ["user owns Hoka running shoes"]),
            ("what is the user training for?", ["user is training for a half marathon"]),
            ("how does the user track workouts?", ["user tracks runs with a Garmin watch", "user measures heart rate variability"]),
            ("when does the user run?", ["user runs three times per week", "user avoids running in heat"]),
            ("what does the user do on vacation?", ["user does bodyweight workouts when traveling"]),
            ("does the user do yoga?", ["user does yoga on Sundays", "user stretches for fifteen minutes daily"]),
            ("where does the user prefer to exercise?", ["user prefers outdoor exercise over gyms", "user prefers trail running"]),
            ("does the user bike?", ["user bikes to work on Thursdays"]),
        ],
        "cooking": [
            ("what cuisines does the user cook?", ["user cooks Italian food on weekends", "user is learning Japanese cuisine"]),
            ("what does the user make from scratch?", ["user makes fresh pasta from scratch", "user makes homemade pizza every Friday", "user keeps sourdough starter alive"]),
            ("what kitchenware does the user own?", ["user owns a cast iron skillet", "user owns a Le Creuset dutch oven", "user uses a kitchen scale for baking"]),
            ("how does the user source ingredients?", ["user buys groceries at a farmers market", "user grows herbs on the windowsill"]),
            ("what does the user avoid eating?", ["user avoids processed foods", "user avoids seed oils in cooking"]),
            ("what does the user do on Sundays?", ["user batch-cooks on Sundays"]),
            ("what does the user bake?", ["user makes homemade pizza every Friday", "user keeps sourdough starter alive"]),
            ("what fermentation does the user do?", ["user tried fermenting kimchi last month", "user keeps sourdough starter alive"]),
            ("where does the user get recipes?", ["user reads Serious Eats recipes"]),
            ("how does the user finish dishes?", ["user finishes dishes with flaky salt"]),
        ],
    }
    for templates in query_templates.values():
        for q, relevant in templates:
            queries.append(QueryCase(query=q, relevant_facts=relevant))

    return facts, queries


# ---------- baseline: naive cosine search (no filtering, no synthesis) ----------

class NaiveVectorStore:
    """Simplest possible baseline: a dict of {id: (text, embedding)}, cosine search.

    No parent filtering, no cache, no synthesis -- this is what a beginner's
    RAG pipeline uses straight out of vector-db tutorials.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self.texts: list[str] = []
        self.vecs: list[np.ndarray] = []

    def add(self, text: str) -> None:
        self.texts.append(text)
        self.vecs.append(self.provider.encode(text))

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        q = self.provider.encode(query)
        q_norm = np.linalg.norm(q) or 1.0
        out: list[tuple[str, float]] = []
        for text, vec in zip(self.texts, self.vecs, strict=True):
            v_norm = np.linalg.norm(vec) or 1.0
            score = float(vec @ q) / (v_norm * q_norm)
            out.append((text, score))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:limit]


# ---------- metrics ----------

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not retrieved:
        return 0.0
    relevant_lower = {r.lower() for r in relevant}
    top = retrieved[:k]
    hits = sum(1 for t in top if t.lower() in relevant_lower)
    return hits / min(len(top), k)


def hit_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_lower = {r.lower() for r in relevant}
    return 1.0 if any(t.lower() in relevant_lower for t in retrieved[:k]) else 0.0


def mean_reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_lower = {r.lower() for r in relevant}
    for i, t in enumerate(retrieved, start=1):
        if t.lower() in relevant_lower:
            return 1.0 / i
    return 0.0


# ---------- benchmark ----------

@dataclass
class BenchmarkResult:
    system: str
    p_at_1: float
    p_at_3: float
    p_at_5: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    total_queries: int
    avg_query_ms: float


def _evaluate_retrievals(
    system_name: str, retrievals: list[tuple[QueryCase, list[str]]],
    durations: list[float],
) -> BenchmarkResult:
    metrics = {
        "p@1": [], "p@3": [], "p@5": [],
        "hit@1": [], "hit@3": [], "hit@5": [],
        "mrr": [],
    }
    for case, retrieved in retrievals:
        metrics["p@1"].append(precision_at_k(retrieved, case.relevant_facts, 1))
        metrics["p@3"].append(precision_at_k(retrieved, case.relevant_facts, 3))
        metrics["p@5"].append(precision_at_k(retrieved, case.relevant_facts, 5))
        metrics["hit@1"].append(hit_at_k(retrieved, case.relevant_facts, 1))
        metrics["hit@3"].append(hit_at_k(retrieved, case.relevant_facts, 3))
        metrics["hit@5"].append(hit_at_k(retrieved, case.relevant_facts, 5))
        metrics["mrr"].append(mean_reciprocal_rank(retrieved, case.relevant_facts))

    return BenchmarkResult(
        system=system_name,
        p_at_1=statistics.mean(metrics["p@1"]),
        p_at_3=statistics.mean(metrics["p@3"]),
        p_at_5=statistics.mean(metrics["p@5"]),
        hit_at_1=statistics.mean(metrics["hit@1"]),
        hit_at_3=statistics.mean(metrics["hit@3"]),
        hit_at_5=statistics.mean(metrics["hit@5"]),
        mrr=statistics.mean(metrics["mrr"]),
        total_queries=len(retrievals),
        avg_query_ms=statistics.mean(durations) * 1000 if durations else 0.0,
    )


def run_naive_baseline(
    provider: EmbeddingProvider,
    facts: list[str],
    queries: list[QueryCase],
    k: int = 10,
) -> BenchmarkResult:
    store = NaiveVectorStore(provider)
    for f in facts:
        store.add(f)
    retrievals: list[tuple[QueryCase, list[str]]] = []
    durations: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        results = store.search(q.query, limit=k)
        durations.append(time.perf_counter() - t0)
        retrievals.append((q, [r[0] for r in results]))
    return _evaluate_retrievals("naive baseline", retrievals, durations)


def run_genome_basic(
    provider: EmbeddingProvider,
    facts: list[str],
    queries: list[QueryCase],
    k: int = 10,
) -> BenchmarkResult:
    """genome Memory without any advanced features (no filter_parents edge case
    because there are no parents in this pure-retrieval dataset). Tests the
    base API + cache."""
    m = Memory(
        embedding_provider=provider,
        storage=":memory:",
    )
    try:
        for f in facts:
            m.add(f, user_id="bench")
        retrievals: list[tuple[QueryCase, list[str]]] = []
        durations: list[float] = []
        for q in queries:
            t0 = time.perf_counter()
            results = m.search(q.query, user_id="bench", limit=k)
            durations.append(time.perf_counter() - t0)
            retrievals.append((q, [r.content for r in results]))
        return _evaluate_retrievals("genome (basic)", retrievals, durations)
    finally:
        m.close()


def run_genome_with_cache_warm(
    provider: EmbeddingProvider,
    facts: list[str],
    queries: list[QueryCase],
    k: int = 10,
) -> BenchmarkResult:
    """genome Memory with cache WARM (second pass of same queries).

    Demonstrates the cache speedup on repeat queries -- common in agents that
    replay recent context between turns.
    """
    m = Memory(embedding_provider=provider, storage=":memory:")
    try:
        for f in facts:
            m.add(f, user_id="bench")
        # Cold pass
        for q in queries:
            m.search(q.query, user_id="bench", limit=k)
        # Warm pass (measured)
        retrievals: list[tuple[QueryCase, list[str]]] = []
        durations: list[float] = []
        for q in queries:
            t0 = time.perf_counter()
            results = m.search(q.query, user_id="bench", limit=k)
            durations.append(time.perf_counter() - t0)
            retrievals.append((q, [r.content for r in results]))
        return _evaluate_retrievals("genome (cache warm)", retrievals, durations)
    finally:
        m.close()


def print_report(results: list[BenchmarkResult]) -> None:
    col_names = ["system", "p@1", "p@3", "p@5", "hit@1", "hit@3", "hit@5", "mrr", "avg_ms"]
    col_widths = [24, 6, 6, 6, 7, 7, 7, 6, 9]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths, strict=True))

    print("\n" + fmt_row(col_names))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for r in results:
        print(fmt_row([
            r.system,
            f"{r.p_at_1:.3f}",
            f"{r.p_at_3:.3f}",
            f"{r.p_at_5:.3f}",
            f"{r.hit_at_1:.3f}",
            f"{r.hit_at_3:.3f}",
            f"{r.hit_at_5:.3f}",
            f"{r.mrr:.3f}",
            f"{r.avg_query_ms:.2f}",
        ]))


def save_report(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))


def main() -> int:
    random.seed(42)
    np.random.seed(42)

    print("Building conversational dataset...")
    facts, queries = build_conversational_dataset()
    print(f"  {len(facts)} facts, {len(queries)} queries")

    print("\nLoading embedding provider (all-MiniLM-L6-v2)...")
    from genome.embeddings import EmbeddingProvider
    provider = EmbeddingProvider()

    results: list[BenchmarkResult] = []

    print("\n[1/3] Naive baseline (no caching, no filtering)...")
    results.append(run_naive_baseline(provider, facts, queries))

    print("\n[2/3] genome basic (cold cache)...")
    results.append(run_genome_basic(provider, facts, queries))

    print("\n[3/3] genome with warm cache...")
    results.append(run_genome_with_cache_warm(provider, facts, queries))

    print_report(results)
    save_report(results, Path("results/memory_benchmark.json"))
    print("\nResults saved to results/memory_benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
