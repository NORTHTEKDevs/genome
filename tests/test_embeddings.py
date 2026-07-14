import numpy as np

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
