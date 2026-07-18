# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""The `python -m genome.verify` receipt must actually pass on a clean machine --
it is the repo's "don't trust me, run it" credibility artifact, so a broken one is
worse than none. This runs it small and asserts every core claim reproduces."""

import sys

import genome.verify as v


def test_verify_reproduces_core_claims(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["genome-verify", "--n", "5"])
    rc = v.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "REPRODUCED LOCALLY" in out
    assert "Air-gapped write path" in out
    assert "0 network attempts" in out  # the socket block saw no phone-home
    assert "[FAIL]" not in out
