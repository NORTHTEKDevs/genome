"""Query cache for Memory.search.

Keyed by (query, scope, limit, filter_parents) + an O(1) per-scope epoch that
increments on every mutation. A cache miss was previously O(n) (full scope
scan to compute fingerprint); now it's O(1).

Thread-safe: the `AsyncMemory` facade dispatches to sync methods through
`asyncio.to_thread`, which means multiple worker threads may call into the
cache concurrently. A process-wide lock protects the underlying OrderedDict
from concurrent mutation.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class ResponseCache:
    """Thread-safe LRU cache for Memory.search results.

    Parameters
    ----------
    capacity : int
        Max number of entries to keep. LRU-evicted beyond this.
    """

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: OrderedDict[str, list] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def _key(
        self,
        query: str,
        user_id: str | None,
        agent_id: str | None,
        limit: int,
        filter_parents: bool,
        epoch: int,
        mode: str = "dense",
    ) -> str:
        raw = (
            f"{query.strip().lower()}|{user_id}|{agent_id}|"
            f"{limit}|{filter_parents}|{epoch}|{mode}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        query: str,
        user_id: str | None,
        agent_id: str | None,
        limit: int,
        filter_parents: bool,
        epoch: int,
        mode: str = "dense",
    ) -> list | None:
        k = self._key(query, user_id, agent_id, limit, filter_parents, epoch, mode)
        with self._lock:
            if k in self._data:
                self._data.move_to_end(k)
                self.stats.hits += 1
                return list(self._data[k])
            self.stats.misses += 1
            return None

    def put(
        self,
        query: str,
        user_id: str | None,
        agent_id: str | None,
        limit: int,
        filter_parents: bool,
        epoch: int,
        results: list,
        mode: str = "dense",
    ) -> None:
        k = self._key(query, user_id, agent_id, limit, filter_parents, epoch, mode)
        with self._lock:
            self._data[k] = list(results)
            self._data.move_to_end(k)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)
            self.stats.size = len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.stats.invalidations += 1
            self.stats.size = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ScopeEpochs:
    """O(1) per-scope mutation counter.

    Every mutation to a scope bumps its counter; cache keys embed the current
    counter so invalidation is automatic without scanning records.
    """

    _GLOBAL = (None, None)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epochs: dict[tuple[str | None, str | None], int] = {}
        self._global_epoch = 0

    def _key(self, user_id: str | None, agent_id: str | None) -> tuple:
        return (user_id, agent_id)

    def bump(self, user_id: str | None, agent_id: str | None) -> None:
        """Call on every mutation. Also bumps the global epoch so unscoped
        queries invalidate correctly."""
        with self._lock:
            k = self._key(user_id, agent_id)
            self._epochs[k] = self._epochs.get(k, 0) + 1
            self._global_epoch += 1

    def current(self, user_id: str | None, agent_id: str | None) -> int:
        """Current epoch for a scope. Combines scope-local + global to also
        invalidate when a parent scope (None, None) mutates."""
        with self._lock:
            k = self._key(user_id, agent_id)
            return self._epochs.get(k, 0) + self._global_epoch

    def reset_all(self) -> None:
        with self._lock:
            self._epochs.clear()
            self._global_epoch += 1


__all__ = ["ResponseCache", "CacheStats", "ScopeEpochs"]
