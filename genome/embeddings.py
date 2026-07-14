"""Sentence embedding provider wrapping sentence-transformers (default) or OpenAI.

A single `EmbeddingProvider(model_name=...)` constructor handles both:
  - "sentence-transformers/all-MiniLM-L6-v2"  (default, 384-d, local)
  - "BAAI/bge-small-en-v1.5"                  (384-d, local)
  - "BAAI/bge-large-en-v1.5"                  (1024-d, local)
  - "openai:text-embedding-3-small"           (1536-d, API)
  - "openai:text-embedding-3-large"           (3072-d, API)

Routing is by the `openai:` prefix. Local models load via SentenceTransformer.
OpenAI embeddings require the `openai` package and an OPENAI_API_KEY env var.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingProvider:
    """Returns numpy float32 embeddings. Polymorphic on `model_name` prefix.

    `openai:<model>` routes to the OpenAI embeddings API. Anything else
    routes to sentence-transformers (loaded once, cached). Both backends
    share the same `encode(text) -> ndarray` and `encode_batch(texts) ->
    ndarray` interface so the rest of the library is backend-agnostic.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        if model_name.startswith("openai:"):
            self._backend = _OpenAIBackend(model_name.removeprefix("openai:"))
        else:
            self._backend = _SentenceTransformerBackend(model_name)
        self.dim = self._backend.dim

    def encode(self, text: str) -> np.ndarray:
        return self._backend.encode(text).astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self._backend.encode_batch(texts).astype(np.float32)


class _SentenceTransformerBackend:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True)


class _OpenAIBackend:
    """Thin wrapper over the OpenAI embeddings API.

    The `dim` is sniffed from a single test call at construction so the rest
    of the library can size the pgvector column correctly. Errors here mean
    the caller's OPENAI_API_KEY is missing or wrong; surface them loud.
    """

    def __init__(self, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI embeddings require the `openai` package. "
                "Install with: pip install openai"
            ) from e
        self._client = OpenAI()
        self._model = model
        # Sniff dimension with one cheap call -- cached for the lifetime of this provider.
        probe = self._client.embeddings.create(model=model, input="dim probe")
        self.dim = len(probe.data[0].embedding)

    # OpenAI's embedding API documents 2048 inputs per request. We use 2048
    # as the chunk size; smaller would waste API budget on long benchmarks.
    _CHUNK = 2048
    # Conservative retry on transient API errors (rate limit, 5xx).
    _MAX_RETRIES = 3  # retained for API compatibility (unused by retry loop)
    _TRANSIENT_RETRIES = 30  # ~20 min of capped backoff before giving up
    _RETRY_BASE_S = 1.0

    def encode(self, text: str) -> np.ndarray:
        if not text:
            # OpenAI returns an error on empty input. Don't bill the user
            # for a no-op; return a zero vector consistent with encode_batch.
            return np.zeros(self.dim, dtype=np.float32)
        resp = self._call_with_retry(input_payload=text)
        return np.asarray(resp.data[0].embedding, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self._CHUNK):
            chunk = texts[i : i + self._CHUNK]
            # Replace empty strings with a single space so the OpenAI API
            # doesn't reject the whole batch over one bad item.
            safe_chunk = [t if t else " " for t in chunk]
            resp = self._call_with_retry(input_payload=safe_chunk)
            out.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
        return np.stack(out)

    def _call_with_retry(self, *, input_payload):
        """OpenAI embeddings call with capped exponential-backoff retry on
        transient failures. At benchmark scale (200k calls on the default
        embedder) a single 429 or network blip must not stop a multi-hour
        sweep -- so transient faults get a patient budget (~20 min), matching
        the responder/judge LLM path's hardening. Rate limits and network
        resets are weather on a long run, not failure; a genuinely dead
        network still aborts."""
        import time as _time
        last_exc: Exception | None = None
        for attempt in range(self._TRANSIENT_RETRIES):
            try:
                return self._client.embeddings.create(
                    model=self._model, input=input_payload,
                )
            except Exception as e:  # noqa: BLE001 -- intentional broad catch
                last_exc = e
                name = type(e).__name__.lower()
                msg = str(e).lower()
                # Retry on rate-limit, timeout, connection reset, or HTTP 5xx.
                # Auth (401/403), bad-request (400), not-found (404) never retry.
                try:
                    status = int(getattr(e, "status_code", 0))
                except (TypeError, ValueError):
                    status = 0
                is_5xx = 500 <= status < 600
                transient = (
                    "rate" in msg or "429" in msg
                    or "timeout" in name or "timed out" in msg
                    or "connection" in name or "connecterror" in name
                    or is_5xx
                )
                if transient and attempt < self._TRANSIENT_RETRIES - 1:
                    # Capped backoff so late attempts don't sleep for hours.
                    _time.sleep(min(30.0, self._RETRY_BASE_S * (2 ** attempt)))
                    continue
                raise
        # exhausted retries
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenAI embeddings call failed without exception")
