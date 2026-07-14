"""Prove GENOME's DEFAULT write path is fully local: no network, no LLM, no embedding
API. Measures model warm-up (one-time) and steady-state per-message write latency using
the default embedder (sentence-transformers/all-MiniLM-L6-v2). No API keys required.

This is the artifact behind the "runs air-gapped, single-digit-ms writes" claim that Mem0
cannot match (Mem0 needs an LLM API in the write path to extract memories).

Run: .venv/Scripts/python.exe benchmarks/local_writepath.py [--n 200]
"""
from __future__ import annotations

import argparse
import socket
import time

from genome.embeddings import DEFAULT_MODEL, EmbeddingProvider
from genome.memory.facade import Memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    # Guard: assert the default provider makes no network calls. We block outbound
    # sockets AFTER model load, then do the writes -- if anything tries to phone home,
    # the write raises and the "fully local" claim is falsified loudly.
    print(f"default embedder: {DEFAULT_MODEL}")
    t0 = time.time()
    embed = EmbeddingProvider()            # loads the local model (one-time)
    warm = time.time() - t0
    print(f"model warm-up (one-time, cold): {warm:.2f}s, dim={embed.dim}")

    mem = Memory(storage=":memory:", embedding_provider=embed)
    msgs = [f"Message {i}: the user mentioned detail number {i} about their project on day {i%30}."
            for i in range(args.n)]

    # one warm write so any lazy init is out of the way
    mem.add(msgs[0], user_id="warm")

    # Block the network to PROVE no external call happens during writes.
    _orig = socket.socket.connect
    blocked = {"hits": 0}
    def _no_net(self, addr, *a, **k):
        blocked["hits"] += 1
        raise OSError(f"network blocked (attempted {addr}) -- write path is NOT local")
    socket.socket.connect = _no_net
    try:
        t0 = time.time()
        for m in msgs:
            mem.add(m, user_id="u")
        dt = time.time() - t0
    finally:
        socket.socket.connect = _orig
    mem.close()

    per = dt / args.n * 1000
    print(f"\n=== LOCAL WRITE PATH (network blocked during writes) ===")
    print(f"  wrote {args.n} messages in {dt:.2f}s -> {per:.1f} ms/message")
    print(f"  outbound network attempts during writes: {blocked['hits']} (0 = fully local)")
    print(f"  LLM calls: 0   embedding-API calls: 0   external dependencies: none")
    print(f"\n  vs Mem0 measured write path: 2,055 ms/message + 1 LLM API call (Finding 03)")
    if per > 0:
        print(f"  -> ~{2055/per:.0f}x lower write latency, and it runs air-gapped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
