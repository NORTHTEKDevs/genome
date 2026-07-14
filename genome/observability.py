"""Structured logging and metrics hooks for genome.

Keeps zero runtime dependencies: uses Python's stdlib `logging` with a JSON
formatter, and provides a simple in-process metrics registry that can be
scraped (or forwarded to Prometheus/OTel via a user-provided sink).

Usage:

    from genome.observability import configure_logging, get_metrics

    configure_logging(level="INFO", json_output=True)
    metrics = get_metrics()
    metrics.counter("memory.add.count", tags={"user_id": "alice"}).inc()

To forward to Prometheus/OTel, attach a sink:

    metrics.set_sink(my_otel_sink)  # sink receives (metric_name, value, tags)
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------- logging ----------

_CONFIGURED = False


class JSONFormatter(logging.Formatter):
    """Structured JSON log records. One record per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra fields the caller passed via logger.info(..., extra={...})
        for k, v in record.__dict__.items():
            if k in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    stream=None,
) -> None:
    """Set up logging once at application startup.

    Parameters
    ----------
    level : str
        "DEBUG", "INFO", "WARNING", "ERROR".
    json_output : bool
        If True, emit JSON records. If False, emit human-readable.
    stream : file-like
        Destination stream. Defaults to stderr.
    """
    global _CONFIGURED
    handler = logging.StreamHandler(stream=stream or sys.stderr)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
    root = logging.getLogger("genome")
    # Idempotent: remove old handlers on reconfigure
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'genome' namespace."""
    if not name.startswith("genome"):
        name = f"genome.{name}"
    return logging.getLogger(name)


# ---------- metrics ----------

MetricSink = Callable[[str, float, dict[str, str]], None]
"""A sink receives (metric_name, value, tags). Use to bridge to OTel, Prometheus, etc."""


@dataclass
class CounterHandle:
    """Handle for a counter metric. Methods are thread-safe."""
    name: str
    tags: dict[str, str] = field(default_factory=dict)
    _registry: MetricsRegistry | None = None

    def inc(self, amount: float = 1.0) -> None:
        if self._registry is not None:
            self._registry._counter_inc(self.name, self.tags, amount)


@dataclass
class HistogramHandle:
    """Handle for a histogram metric. Records latencies or sizes."""
    name: str
    tags: dict[str, str] = field(default_factory=dict)
    _registry: MetricsRegistry | None = None

    def observe(self, value: float) -> None:
        if self._registry is not None:
            self._registry._histogram_observe(self.name, self.tags, value)

    def time(self) -> _Timer:
        """Context manager: records the elapsed time on __exit__."""
        return _Timer(self)


class _Timer:
    def __init__(self, h: HistogramHandle) -> None:
        self._h = h
        self._start = 0.0

    def __enter__(self) -> _Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._h.observe(time.perf_counter() - self._start)


class MetricsRegistry:
    """Thread-safe in-process metrics registry.

    - Counters: cumulative integer/float values with tag combinations
    - Histograms: sequences of observations -- we keep counts + sums + recent window

    An optional sink forwards every recorded value to an external system.
    """

    _HIST_WINDOW = 1024

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, frozenset], float] = defaultdict(float)
        self._hist_counts: dict[tuple[str, frozenset], int] = defaultdict(int)
        self._hist_sums: dict[tuple[str, frozenset], float] = defaultdict(float)
        self._hist_recent: dict[tuple[str, frozenset], list[float]] = defaultdict(list)
        self._sink: MetricSink | None = None

    def set_sink(self, sink: MetricSink | None) -> None:
        with self._lock:
            self._sink = sink

    def counter(
        self, name: str, tags: dict[str, str] | None = None
    ) -> CounterHandle:
        return CounterHandle(name=name, tags=dict(tags or {}), _registry=self)

    def histogram(
        self, name: str, tags: dict[str, str] | None = None
    ) -> HistogramHandle:
        return HistogramHandle(name=name, tags=dict(tags or {}), _registry=self)

    def _counter_inc(
        self, name: str, tags: dict[str, str], amount: float
    ) -> None:
        key = (name, frozenset(tags.items()))
        with self._lock:
            self._counters[key] += amount
            sink = self._sink
        if sink is not None:
            try:
                sink(name, amount, tags)
            except Exception:  # never let a sink break the caller
                pass

    def _histogram_observe(
        self, name: str, tags: dict[str, str], value: float
    ) -> None:
        key = (name, frozenset(tags.items()))
        with self._lock:
            self._hist_counts[key] += 1
            self._hist_sums[key] += value
            buf = self._hist_recent[key]
            buf.append(value)
            if len(buf) > self._HIST_WINDOW:
                del buf[: len(buf) - self._HIST_WINDOW]
            sink = self._sink
        if sink is not None:
            try:
                sink(name, value, tags)
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        """Return a plain-dict snapshot of current metrics (for /metrics or debug)."""
        with self._lock:
            out: dict[str, Any] = {"counters": {}, "histograms": {}}
            for (name, tagset), val in self._counters.items():
                tags = dict(tagset)
                bucket = out["counters"].setdefault(name, [])
                bucket.append({"tags": tags, "value": val})
            for (name, tagset), count in self._hist_counts.items():
                tags = dict(tagset)
                total = self._hist_sums[(name, tagset)]
                recent = list(self._hist_recent[(name, tagset)])
                mean_recent = sum(recent) / len(recent) if recent else 0.0
                bucket = out["histograms"].setdefault(name, [])
                bucket.append({
                    "tags": tags, "count": count,
                    "sum": total,
                    "mean": total / count if count else 0.0,
                    "mean_recent": mean_recent,
                    "max_recent": max(recent) if recent else 0.0,
                })
            return out

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._hist_counts.clear()
            self._hist_sums.clear()
            self._hist_recent.clear()


_REGISTRY = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    """Return the process-global metrics registry."""
    return _REGISTRY


# ---------- error capture ----------

ErrorSink = Callable[["CapturedError"], None]
"""A sink receives each CapturedError. Use to forward to Sentry/Datadog/etc."""


@dataclass
class CapturedError:
    """A snapshot of one captured exception.

    Stable enough to dedupe by `fingerprint` (sha256 of exception type + the
    top non-genome-internal traceback frames). Tags hold any scope hints the
    capturer wanted to attach (user_id, agent_id, operation, request_path).
    """
    timestamp: float
    error_type: str
    message: str
    fingerprint: str
    stack: str
    tags: dict[str, str] = field(default_factory=dict)


class ErrorCapture:
    """Genome-native error capture. A Sentry-equivalent that runs in-process.

    - Bounded ring buffer of recent errors (default 1024).
    - Group counter keyed on `fingerprint` for at-a-glance "what's blowing up
      most".
    - Pluggable `sink` for forwarding to external systems (Sentry SDK,
      Datadog, file, HTTP webhook). Sink errors never bubble back to the
      caller.

    All operations thread-safe via an internal Lock.
    """

    _STACK_LIMIT = 4096        # max chars of formatted traceback per entry
    _FRAME_FINGERPRINT = 5     # frames considered for the fingerprint

    def __init__(self, *, capacity: int = 1024, sink: ErrorSink | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._buf: list[CapturedError] = []
        self._counts: dict[str, int] = defaultdict(int)
        self._sink: ErrorSink | None = sink

    def set_sink(self, sink: ErrorSink | None) -> None:
        with self._lock:
            self._sink = sink

    def capture(
        self, exc: BaseException, *, tags: dict[str, str] | None = None,
    ) -> CapturedError:
        """Record an exception. Returns the CapturedError so callers can
        also log/inspect it. Always succeeds; never raises."""
        ce = self._build(exc, tags or {})
        with self._lock:
            self._buf.append(ce)
            if len(self._buf) > self._capacity:
                # Drop oldest
                del self._buf[: len(self._buf) - self._capacity]
            self._counts[ce.fingerprint] += 1
            sink = self._sink
        if sink is not None:
            try:
                sink(ce)
            except Exception:  # never let a sink break the caller
                pass
        return ce

    def recent(self, limit: int = 50) -> list[CapturedError]:
        """Return the N most recent captures, newest first."""
        with self._lock:
            return list(reversed(self._buf[-max(0, limit):]))

    def grouped(self) -> list[dict[str, Any]]:
        """Return [{fingerprint, count, error_type, message, last_seen}, ...]
        sorted by count descending. Cheap to compute; safe to expose as a
        REST endpoint."""
        with self._lock:
            seen: dict[str, CapturedError] = {}
            for ce in self._buf:
                seen[ce.fingerprint] = ce  # last wins -> most-recent sample
            counts = dict(self._counts)
        rows = [
            {
                "fingerprint": fp,
                "count": counts.get(fp, 0),
                "error_type": seen[fp].error_type,
                "message": seen[fp].message,
                "last_seen": seen[fp].timestamp,
                "tags": dict(seen[fp].tags),
            }
            for fp in seen
        ]
        rows.sort(key=lambda r: r["count"], reverse=True)
        return rows

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()
            self._counts.clear()

    @staticmethod
    def _build(exc: BaseException, tags: dict[str, str]) -> CapturedError:
        import hashlib
        import traceback
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if len(tb) > ErrorCapture._STACK_LIMIT:
            tb = tb[: ErrorCapture._STACK_LIMIT] + "...[truncated]"
        # Fingerprint: error type + top N frame signatures (file:line). Excludes
        # message text so similar errors with different content group together.
        frames = traceback.extract_tb(exc.__traceback__)
        sig_frames = frames[-ErrorCapture._FRAME_FINGERPRINT:] if frames else []
        sig_text = type(exc).__name__ + "|" + "|".join(
            f"{f.filename}:{f.lineno}" for f in sig_frames
        )
        fingerprint = hashlib.sha256(sig_text.encode("utf-8")).hexdigest()[:16]
        return CapturedError(
            timestamp=time.time(),
            error_type=type(exc).__name__,
            message=str(exc)[:500],
            fingerprint=fingerprint,
            stack=tb,
            tags=dict(tags),
        )


_ERROR_CAPTURE = ErrorCapture()


def get_error_capture() -> ErrorCapture:
    """Return the process-global error capture."""
    return _ERROR_CAPTURE


__all__ = [
    "configure_logging",
    "get_logger",
    "JSONFormatter",
    "MetricsRegistry",
    "get_metrics",
    "CounterHandle",
    "HistogramHandle",
    "MetricSink",
    "ErrorCapture",
    "CapturedError",
    "ErrorSink",
    "get_error_capture",
]
