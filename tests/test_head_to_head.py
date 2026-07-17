# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""The GENOME-vs-Mem0 reproducer is a public credibility artifact -- a broken one
is a landmine. This runs its offline `--smoke` mode end-to-end (no key, no network,
no Mem0) and asserts the whole pipeline (ingest -> answer -> judge -> pair ->
McNemar -> verdict) completes. The real key-spending run is verified out-of-band."""

import subprocess
import sys
from pathlib import Path


def test_head_to_head_smoke_runs_offline():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "benchmarks" / "head_to_head.py"), "--smoke"],
        capture_output=True, text=True, timeout=300, cwd=str(root),
    )
    assert proc.returncode == 0, proc.stderr
    assert "SMOKE OK" in proc.stdout
    assert "McNemar" in proc.stdout
    assert "VERDICT:" in proc.stdout
