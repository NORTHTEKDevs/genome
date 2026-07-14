"""Scoring for mini-LoCoMo answers.

- token_f1: token-level F1 between predicted and gold answer (SQuAD style)
- keyword_recall: fraction of gold_keywords present in prediction (case-insensitive)
- contains_all_keywords: 1.0 if every gold_keyword appears in prediction, else 0.0

The keyword metrics are more stable than F1 on short adversarial answers where
the model may phrase the answer differently but still surface the right facts.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)
_ARTICLES = {"a", "an", "the"}


def _normalize(s: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, drop articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return [tok for tok in s.split() if tok and tok not in _ARTICLES]


def token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred)
    gold_toks = _normalize(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(pred: str, gold_keywords: list[str]) -> float:
    if not gold_keywords:
        return 1.0
    pred_lower = pred.lower()
    hits = sum(
        1 for kw in gold_keywords if re.search(r"\b" + re.escape(kw.lower()) + r"\b", pred_lower)
    )
    return hits / len(gold_keywords)


def contains_all_keywords(pred: str, gold_keywords: list[str]) -> float:
    if not gold_keywords:
        return 1.0
    return 1.0 if keyword_recall(pred, gold_keywords) == 1.0 else 0.0
