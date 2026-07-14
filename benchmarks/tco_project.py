"""Deployment TCO projection: GENOME vs Mem0, memory-ingest cost + write latency.

Deterministic -- no API keys, no spend. INPUTS ARE MEASURED (not assumed): they come
from the instrumented 80-turn run logged in Finding 03 (benchmarks/ingest_cost.py ->
results/ingest_cost.log). This script only does the arithmetic of projecting those
measured per-message figures to a stated deployment, under multiple extraction-model
prices so the gap can't be dismissed as expensive-model cherry-picking.

The load-bearing fact is architectural and price-independent: Mem0 makes >=1 LLM call
per message in the write path; GENOME makes 0. Everything below follows from that.

Run: .venv/Scripts/python.exe benchmarks/tco_project.py [--users 10000] [--msgs-per-day 50]
"""
from __future__ import annotations

import argparse

# ---- MEASURED (Finding 03, 80-turn instrumented slice) ----
N = 80
MEM0_IN, MEM0_OUT = 687_192, 4_940          # tokens over 80 msgs (instrumented)
MEM0_CALLS = 80                              # 1.00 LLM call / msg
MEM0_WALL = 164.4                            # s over 80 msgs
GEN_EMBED_TOK = 4_221                        # tokens over 80 msgs
GEN_CALLS = 0                                # 0 LLM calls
GEN_WALL = 19.2                              # s over 80 msgs

# ---- prices ($/Mtok) ----
PRICES = {
    "Haiku 4.5":   (1.00, 5.00),
    "gpt-4o-mini": (0.15, 0.60),            # Mem0's common cheap default
    "gpt-4.1-nano": (0.10, 0.40),           # about as cheap as hosted extraction gets
}
EMBED_PRICE = 0.02                           # text-embedding-3-small


def per_msg(pin, pout):
    m_in, m_out = MEM0_IN / N, MEM0_OUT / N
    mem0 = (m_in * pin + m_out * pout) / 1e6
    gen = (GEN_EMBED_TOK / N) * EMBED_PRICE / 1e6
    return mem0, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=10_000)
    ap.add_argument("--msgs-per-day", type=int, default=50)
    args = ap.parse_args()

    monthly = args.users * args.msgs_per_day * 30
    lat_mem0, lat_gen = MEM0_WALL / N, GEN_WALL / N

    print("=== MEASURED (Finding 03, 80-turn instrumented slice) ===")
    print(f"  Mem0  : {MEM0_CALLS/N:.2f} LLM calls/msg, {lat_mem0*1000:.0f} ms/msg write latency")
    print(f"  GENOME: {GEN_CALLS} LLM calls/msg, {lat_gen*1000:.0f} ms/msg "
          f"(incl. OpenAI embed round-trip; a local embedder is ~single-digit ms)")
    print(f"  -> {lat_mem0/lat_gen:.1f}x lower write latency, and ZERO LLM dependency in the write path\n")

    print(f"=== DEPLOYMENT: {args.users:,} users x {args.msgs_per_day} msgs/day "
          f"= {monthly:,} messages/month ===\n")
    print(f"GENOME memory-ingest LLM calls/month:  0")
    print(f"Mem0   memory-ingest LLM calls/month:  {monthly:,}\n")

    _, gen_msg = per_msg(1, 1)
    gen_month = gen_msg * monthly
    print(f"{'extraction model':14} {'Mem0 $/mo':>14} {'Mem0 $/yr':>14} {'GENOME $/mo':>12} {'ratio':>8}")
    for name, (pin, pout) in PRICES.items():
        mem0_msg, _ = per_msg(pin, pout)
        mem0_month = mem0_msg * monthly
        print(f"{name:14} {mem0_month:>14,.0f} {mem0_month*12:>14,.0f} "
              f"{gen_month:>12,.2f} {mem0_msg/gen_msg:>7,.0f}x")

    print(f"\nEven at the cheapest hosted extraction model, Mem0's memory-ingest bill for a "
          f"{args.users:,}-user\ndeployment is ~${per_msg(*PRICES['gpt-4.1-nano'])[0]*monthly*12:,.0f}/year; "
          f"GENOME's is ~${gen_month*12:,.0f}/year. The gap is the 0-vs-1\nLLM-call-per-message "
          f"architecture, not a pricing choice -- and it WIDENS as Mem0's store fills\n"
          f"(each write ships existing memories back to the LLM; this linear projection under-states it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
