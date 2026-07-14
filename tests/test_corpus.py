import numpy as np

from genome.corpus import build_default_corpus


def test_build_default_corpus_has_minimum_size():
    corpus = build_default_corpus()
    assert len(corpus) >= 100


def test_corpus_has_embeddings_and_texts():
    corpus = build_default_corpus()
    assert corpus.texts[0] is not None
    assert corpus.embeddings.shape[0] == len(corpus.texts)
    assert corpus.embeddings.dtype == np.float32


def test_corpus_search_returns_top_k():
    corpus = build_default_corpus()
    query_vec = corpus.embeddings[0]
    results = corpus.search(query_vec, k=5)
    assert len(results) == 5
    # The first result should be itself (highest cosine sim)
    assert results[0].text == corpus.texts[0]
