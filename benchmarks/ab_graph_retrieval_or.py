"""Cheap retrieval-only A/B of dense vs graph mode, via OpenRouter + local embedder.

Same probe as ab_graph_retrieval.py with two deliberate deltas:
  - entity extraction (the only paid part) runs through OpenRouter
    (anthropic/claude-haiku-4.5, same model as the original protocol);
  - embeddings use the library's local default (all-MiniLM-L6-v2) for
    every mode, matching GENOME's shipped configuration and the n=205
    LongMemEval corroboration run.

Env: OPENROUTER_API_KEY required.
Run: .venv/Scripts/python.exe benchmarks/ab_graph_retrieval_or.py [--n-convs 2]
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

from openai import OpenAI

from genome.embeddings import EmbeddingProvider
from genome.evals.locomo import LocomoConfig, load_locomo, replay_conversation
from genome.memory.facade import Memory

OR_BASE = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-haiku-4.5"
_client: OpenAI | None = None


def haiku(prompt: str) -> str:
    global _client
    if _client is None:
        _client = OpenAI(base_url=OR_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    for attempt in (1, 2):
        try:
            r = _client.chat.completions.create(
                model=MODEL, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}])
            return (r.choices[0].message.content or "") if r.choices else ""
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)
    return ""


def hit_rate(hits, evidence) -> float:
    ev = {str(e) for e in evidence}
    if not ev:
        return None
    got = {str(h.record.metadata.get("dia_id", "")) for h in hits} - {""}
    return len(ev & got) / len(ev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-convs", type=int, default=2)
    args = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("need OPENROUTER_API_KEY"); return 1

    convs = load_locomo("benchmarks/data/locomo10.json")[: args.n_convs]
    cfg = LocomoConfig(name="probe", top_k=30, filter_parents=True,
                       search_mode="graph", auto_extract_entities=True)
    embed = EmbeddingProvider()  # local default: all-MiniLM-L6-v2

    dense_hr = defaultdict(list)
    graph_hr = defaultdict(list)
    graph_fired = defaultdict(int)
    counts = defaultdict(int)

    for ci, conv in enumerate(convs):
        mem = Memory(storage=":memory:", embedding_provider=embed,
                     llm_call=haiku, auto_extract_entities=True)
        print(f"ingesting conv {conv.conversation_id} "
              f"({len(conv.turns)} turns) with entity extraction...", flush=True)
        replay_conversation(mem, conv, user_id="probe", config=cfg)
        n_ent = len(mem.list_all(user_id="probe", agent_id=conv.conversation_id))
        print(f"  ingested; {n_ent} records in scope", flush=True)
        for q in conv.questions:
            if q.category == "adversarial" or not q.evidence:
                continue
            d = mem.search(q.question, user_id="probe",
                           agent_id=conv.conversation_id, limit=30,
                           filter_parents=True, mode="dense")
            g = mem.search(q.question, user_id="probe",
                           agent_id=conv.conversation_id, limit=30,
                           filter_parents=True, mode="graph")
            hd, hg = hit_rate(d, q.evidence), hit_rate(g, q.evidence)
            if hd is None:
                continue
            dense_hr[q.category].append(hd)
            graph_hr[q.category].append(hg)
            counts[q.category] += 1
            if {r.record.id for r in d} != {r.record.id for r in g}:
                graph_fired[q.category] += 1
        mem.close()

    print("\n=== RETRIEVAL hit_rate: dense vs graph (rebuilt) ===")
    print(f"{'category':12} {'n':>4} {'dense':>7} {'graph':>7} {'delta':>7} {'fired':>6}")
    all_d, all_g = [], []
    for cat in sorted(counts):
        d = sum(dense_hr[cat]) / len(dense_hr[cat])
        g = sum(graph_hr[cat]) / len(graph_hr[cat])
        all_d += dense_hr[cat]; all_g += graph_hr[cat]
        print(f"{cat:12} {counts[cat]:>4} {d:>7.3f} {g:>7.3f} {g-d:>+7.3f} "
              f"{graph_fired[cat]:>6}")
    md = sum(all_d) / len(all_d); mg = sum(all_g) / len(all_g)
    print(f"{'ALL':12} {len(all_d):>4} {md:>7.3f} {mg:>7.3f} {mg-md:>+7.3f}")
    print("\nREAD: graph must be >= dense on multi-hop and NOT below dense "
          "elsewhere. If multi-hop delta > 0 with no regression, the feature works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
