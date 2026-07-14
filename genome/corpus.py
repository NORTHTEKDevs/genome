"""Retrieval corpus for GENOME validation experiments.

The default corpus includes items covering the concepts in the parent
pair dataset plus distractors, so retrieval tests can find expected
hybrids meaningfully.

The corpus is built from:
1. All expected_hybrids from the parent-pair dataset (the targets to retrieve)
2. All parent strings themselves (realism: parents exist in the world)
3. A fixed list of semantic-noise distractors
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genome.dataset import load_parent_pairs
from genome.embeddings import EmbeddingProvider

# Semantic-noise distractors - unrelated concepts to add retrieval difficulty.
DISTRACTORS: list[str] = [
    "quantum physics",
    "French cuisine",
    "civil engineering",
    "medieval history",
    "astronomy",
    "jazz music",
    "forestry",
    "ceramics",
    "civil law",
    "geology",
    "accounting",
    "plumbing",
    "botany",
    "fashion design",
    "theology",
    "architecture",
    "marine biology",
    "cryptography",
    "meteorology",
    "philosophy",
    "psychology",
    "anthropology",
    "economics",
    "sociology",
    "literature",
    "poetry",
    "sculpture",
    "dance",
    "gardening",
    "woodworking",
    "welding",
    "knitting",
    "pottery",
    "glass blowing",
    "blacksmithing",
    "taxidermy",
    "origami",
    "numismatics",
    "philately",
    "birdwatching",
    "cave exploration",
    "mountaineering",
    "kite flying",
    "paragliding",
    "stamp collecting",
    "coin collecting",
    "rock climbing",
    "genealogy",
    "astrology",
    "lapidary",
]


def _build_default_texts() -> list[str]:
    """Gather unique texts from dataset hybrids, parents, and distractors."""
    pairs = load_parent_pairs()
    texts: dict[str, None] = {}  # ordered set
    for p in pairs:
        for h in p.expected_hybrids:
            texts[h] = None
    for p in pairs:
        texts[p.parent_a] = None
        texts[p.parent_b] = None
    for d in DISTRACTORS:
        texts[d] = None
    return list(texts.keys())


# Lazy-computed on first import from the dataset
DEFAULT_TEXTS: list[str] = _build_default_texts()


@dataclass
class RetrievalResult:
    text: str
    score: float
    index: int


class RetrievalCorpus:
    """In-memory corpus over which we run cosine-similarity search."""

    def __init__(self, texts: list[str], embeddings: np.ndarray) -> None:
        assert len(texts) == embeddings.shape[0]
        self.texts = list(texts)
        self.embeddings = embeddings.astype(np.float32)
        self._norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self._norms[self._norms == 0] = 1.0

    def __len__(self) -> int:
        return len(self.texts)

    def search(self, query: np.ndarray, k: int = 10) -> list[RetrievalResult]:
        q = query.astype(np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        scores = (self.embeddings @ q) / (self._norms.flatten() * q_norm)
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(text=self.texts[i], score=float(scores[i]), index=int(i))
            for i in top_idx
        ]


def build_default_corpus(
    provider: EmbeddingProvider | None = None,
) -> RetrievalCorpus:
    """Embed the DEFAULT_TEXTS and return a RetrievalCorpus."""
    provider = provider or EmbeddingProvider()
    vecs = provider.encode_batch(DEFAULT_TEXTS)
    return RetrievalCorpus(DEFAULT_TEXTS, vecs)
