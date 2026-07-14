"""Threaded repro for the latent sqlite3.InterfaceError under concurrent use.

An always-on agent calls add/search from multiple threads (heartbeat,
scheduled jobs, UI). The Memory facade must tolerate that.
"""
import threading

import numpy as np

from genome import Memory


class _FakeEmbedder:
    dim = 32

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            vec[hash(token) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode_batch(self, texts):
        return np.stack([self.encode(t) for t in texts])


def test_concurrent_add_and_search_is_thread_safe():
    m = Memory(embedding_provider=_FakeEmbedder())
    errors = []

    def worker(i):
        try:
            for j in range(20):
                m.add(f"fact {i}-{j} about topic {j % 5}", user_id="u")
                m.search(f"topic {j % 5}", user_id="u")
        except Exception as e:  # noqa: BLE001 - we want ANY concurrency error
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"concurrency errors: {[repr(e) for e in errors[:3]]}"
