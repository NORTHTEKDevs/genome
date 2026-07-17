# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Tests that build the REST app exercise functionality, not deployment auth, so
they opt into the unauthenticated local-dev mode. The default-deny behaviour
itself is covered explicitly in tests/server/test_bind_safety.py (which clears
this var). No-op for tests that never touch the REST server."""

import pytest


@pytest.fixture(autouse=True)
def _allow_no_auth(monkeypatch):
    monkeypatch.setenv("GENOME_ALLOW_NO_AUTH", "1")
