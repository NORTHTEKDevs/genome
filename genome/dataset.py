"""Parent pair dataset loader for GENOME validation."""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET = Path(__file__).parent.parent / "benchmarks" / "parent_pairs.json"


@dataclass(frozen=True)
class ParentPair:
    id: str
    parent_a: str
    parent_b: str
    expected_hybrids: list[str]


def load_parent_pairs(path: Path | str | None = None) -> list[ParentPair]:
    """Load curated parent pairs from JSON."""
    p = Path(path) if path else DEFAULT_DATASET
    data = json.loads(p.read_text(encoding="utf-8"))
    return [
        ParentPair(
            id=row["id"],
            parent_a=row["parent_a"],
            parent_b=row["parent_b"],
            expected_hybrids=list(row["expected_hybrids"]),
        )
        for row in data["pairs"]
    ]
