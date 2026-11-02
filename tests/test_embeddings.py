import subprocess
import sys
import types

import numpy as np
import pytest

from genome.embeddings import EmbeddingProvider


def test_embedding_provider_returns_correct_shape():
    provider = EmbeddingProvider()
    vec = provider.encode("machine learning engineer")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_embedding_provider_batch():
    provider = EmbeddingProvider()
    vecs = provider.encode_batch(["a", "b", "c"])
    assert vecs.shape == (3, 384)


def test_embedding_provider_deterministic():
    provider = EmbeddingProvider()
    v1 = provider.encode("test string")
    v2 = provider.encode("test string")
    assert np.allclose(v1, v2)


# -- the model loads from the local cache, not the network -------------------
#
# GENOME is sold as local-first and air-gapped, but SentenceTransformer(name)
# asks the Hugging Face Hub for metadata on every load, even when the model is
# already on disk. With the network down that made startup fail outright
# (unreachable proxy: ProxyError after ~76s) or stall for ~5 minutes of retries
# (no route) before falling back to the cache - past any MCP client's startup
# timeout. A cached model must load with zero network calls; only a model that
# is not on disk yet may go to the network.


def _fake_sentence_transformers(monkeypatch, *, cached, download_error=None):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only") and not cached:
                raise OSError(f"{model_name} is not in the local cache")
            if not kwargs.get("local_files_only") and download_error is not None:
                raise download_error

        def get_sentence_embedding_dimension(self):
            return 8

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


def test_a_cached_model_loads_without_asking_the_hub(monkeypatch):
    calls = _fake_sentence_transformers(monkeypatch, cached=True)

    provider = EmbeddingProvider("some-org/some-model")

    assert provider.dim == 8
    assert calls == [{"local_files_only": True}], "a cached model went to the network"


def test_a_model_not_on_disk_is_downloaded(monkeypatch):
    calls = _fake_sentence_transformers(monkeypatch, cached=False)

    provider = EmbeddingProvider("some-org/some-model")

    assert provider.dim == 8
    assert calls == [{"local_files_only": True}, {}], "a cold cache must fall back to a download"


def test_a_failed_download_reports_the_download_error(monkeypatch):
    """With no cache and no network the user needs the network error, not the
    cache miss that preceded it."""
    _fake_sentence_transformers(
        monkeypatch, cached=False, download_error=ConnectionError("huggingface.co unreachable")
    )

    with pytest.raises(ConnectionError, match="unreachable"):
        EmbeddingProvider("some-org/some-model")


_BLOCKED_STARTUP = r"""
import socket

attempts = []

def tripwire(*args, **kwargs):
    attempts.append(repr(args)[:200])
    raise RuntimeError("network blocked")

socket.socket.connect = tripwire
socket.getaddrinfo = tripwire

from genome.embeddings import EmbeddingProvider

provider = EmbeddingProvider()
assert provider.encode("air-gapped").shape == (384,)
print("NETWORK_ATTEMPTS", len(attempts), attempts[:3])
"""


def test_the_cached_default_model_loads_with_the_network_blocked():
    """The real thing, end to end: once the default model is on disk, a fresh
    process builds the provider with no outbound connection or DNS lookup at
    all. A fresh process, because an earlier load in this one leaves a pooled
    keep-alive connection to the Hub that would hide the calls."""
    EmbeddingProvider()  # make sure the model is cached (downloads it on a cold cache)

    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_STARTUP],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "NETWORK_ATTEMPTS 0 []" in result.stdout, result.stdout
