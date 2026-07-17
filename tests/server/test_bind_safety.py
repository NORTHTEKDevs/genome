# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""The server must be safe by default: it defaults to loopback and refuses to
bind a non-loopback interface without an API key (the API is destructive)."""

import genome.server.__main__ as srv


def _run(monkeypatch, env):
    calls = {}

    def fake_run(app, host, port, reload):
        calls["host"] = host
        calls["port"] = port

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    for k in ("GENOME_HOST", "GENOME_API_KEY", "GENOME_PORT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return srv.main(), calls


def test_default_binds_loopback(monkeypatch):
    rc, calls = _run(monkeypatch, {})
    assert rc == 0
    assert calls["host"] == "127.0.0.1"


def test_public_bind_without_key_is_refused(monkeypatch):
    rc, calls = _run(monkeypatch, {"GENOME_HOST": "0.0.0.0"})
    assert rc == 2
    assert "host" not in calls  # uvicorn.run must never be reached


def test_public_bind_with_key_is_allowed(monkeypatch):
    rc, calls = _run(
        monkeypatch, {"GENOME_HOST": "0.0.0.0", "GENOME_API_KEY": "secret"}
    )
    assert rc == 0
    assert calls["host"] == "0.0.0.0"


def test_localhost_without_key_is_allowed(monkeypatch):
    rc, calls = _run(monkeypatch, {"GENOME_HOST": "127.0.0.1"})
    assert rc == 0
    assert calls["host"] == "127.0.0.1"


# --- default-deny at the auth layer (closes the `uvicorn --host 0.0.0.0` bypass) ---

def _app_client(monkeypatch, env):
    from fastapi.testclient import TestClient

    from genome.memory.facade import Memory
    from genome.server.app import create_app
    for k in ("GENOME_API_KEY", "GENOME_ALLOW_NO_AUTH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return TestClient(create_app(memory=Memory(storage=":memory:")))


def test_no_key_no_optin_refuses_503(monkeypatch):
    # No API key and no explicit opt-in: every endpoint refuses, regardless of bind.
    c = _app_client(monkeypatch, {})
    assert c.get("/health").status_code == 503


def test_no_key_with_optin_serves(monkeypatch):
    c = _app_client(monkeypatch, {"GENOME_ALLOW_NO_AUTH": "1"})
    assert c.get("/health").status_code == 200


def test_key_set_requires_it(monkeypatch):
    c = _app_client(monkeypatch, {"GENOME_API_KEY": "secret"})
    assert c.get("/health").status_code == 401
    assert c.get("/health", headers={"X-API-Key": "secret"}).status_code == 200
