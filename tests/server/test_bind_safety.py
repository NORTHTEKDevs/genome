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


# --- the opt-in is confined to loopback peers, so it can't be abused when the
# --- process is bound to 0.0.0.0 by a launcher that skips the __main__ bind gate.

class _FakeReq:
    class _C:
        def __init__(self, host):
            self.host = host

    def __init__(self, host):
        self.client = self._C(host) if host is not None else None


def test_is_local_client_rejects_remote():
    from genome.server.app import _is_local_client
    assert _is_local_client(_FakeReq("8.8.8.8")) is False
    assert _is_local_client(_FakeReq("192.168.1.10")) is False
    assert _is_local_client(_FakeReq("2001:4860:4860::8888")) is False


def test_is_local_client_allows_loopback():
    from genome.server.app import _is_local_client
    assert _is_local_client(_FakeReq("127.0.0.1")) is True
    assert _is_local_client(_FakeReq("127.0.0.5")) is True
    assert _is_local_client(_FakeReq("::1")) is True


def test_is_local_client_no_client_is_refused():
    from genome.server.app import _is_local_client
    assert _is_local_client(_FakeReq(None)) is False


def test_optin_still_denies_remote_peer(monkeypatch):
    # Even with GENOME_ALLOW_NO_AUTH=1, a non-loopback peer is refused. A real ASGI
    # server fills request.client from the socket; TestClient can't set it per
    # request, so force the loopback check to fail to simulate a remote attacker.
    import genome.server.app as appmod

    monkeypatch.setattr(appmod, "_is_local_client", lambda request: False)
    c = _app_client(monkeypatch, {"GENOME_ALLOW_NO_AUTH": "1"})
    assert c.get("/health").status_code == 503
