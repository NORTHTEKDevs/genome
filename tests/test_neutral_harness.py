# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""The neutral harness is a public credibility artifact; a broken one is worse
than none. Its offline --smoke mode must run the full N-system pipeline."""

import subprocess
import sys
from pathlib import Path


def test_neutral_smoke_runs_offline():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(  # noqa: S603 - fixed argv, our own script
        [sys.executable, str(root / "benchmarks" / "neutral" / "run.py"), "--smoke"],
        capture_output=True, text=True, timeout=300, cwd=str(root),
    )
    assert proc.returncode == 0, proc.stderr
    assert "SMOKE OK" in proc.stdout
    assert "Pairwise McNemar" in proc.stdout
    assert "genome vs stub-a" in proc.stdout
    assert "Disclosure:" in proc.stdout
