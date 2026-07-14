"""Startup env-var validation tests."""
import pytest

from genome.errors import ConfigError
from genome.server.app import _build_memory_from_env, validate_env_config


def test_validate_env_clean_by_default(monkeypatch):
    # Wipe all GENOME_* env vars for a clean baseline
    for k in list(__import__("os").environ):
        if k.startswith("GENOME_"):
            monkeypatch.delenv(k, raising=False)
    assert validate_env_config() == []


def test_validate_env_bad_postgres_dsn(monkeypatch):
    monkeypatch.setenv("GENOME_STORAGE", "postgresql://missing-at-sign")
    issues = validate_env_config()
    assert any("malformed" in i for i in issues)


def test_validate_env_rejects_negative_cache_size(monkeypatch):
    monkeypatch.setenv("GENOME_CACHE_SIZE", "-1")
    issues = validate_env_config()
    assert any("GENOME_CACHE_SIZE" in i and "positive" in i for i in issues)


def test_validate_env_rejects_nonint_cache_size(monkeypatch):
    monkeypatch.setenv("GENOME_CACHE_SIZE", "biggie")
    issues = validate_env_config()
    assert any("GENOME_CACHE_SIZE" in i and "integer" in i for i in issues)


def test_validate_env_warns_on_public_bind_without_api_key(monkeypatch):
    monkeypatch.setenv("GENOME_HOST", "0.0.0.0")
    monkeypatch.delenv("GENOME_API_KEY", raising=False)
    issues = validate_env_config()
    assert any("GENOME_API_KEY" in i for i in issues)


def test_validate_env_ipv6_public_bind_also_warned(monkeypatch):
    monkeypatch.setenv("GENOME_HOST", "::")
    monkeypatch.delenv("GENOME_API_KEY", raising=False)
    issues = validate_env_config()
    assert any("GENOME_API_KEY" in i for i in issues)


def test_validate_env_localhost_no_warning(monkeypatch):
    monkeypatch.setenv("GENOME_HOST", "127.0.0.1")
    monkeypatch.delenv("GENOME_API_KEY", raising=False)
    issues = validate_env_config()
    assert not any("GENOME_API_KEY" in i for i in issues)


def test_validate_env_public_with_api_key_no_warning(monkeypatch):
    monkeypatch.setenv("GENOME_HOST", "0.0.0.0")
    monkeypatch.setenv("GENOME_API_KEY", "real-secret")
    issues = validate_env_config()
    assert not any("GENOME_API_KEY" in i for i in issues)


def test_build_memory_from_env_fails_fast(monkeypatch):
    """Bad config must raise ConfigError before expensive init."""
    monkeypatch.setenv("GENOME_STORAGE", "postgresql://broken")
    with pytest.raises(ConfigError) as ei:
        _build_memory_from_env()
    assert "invalid env configuration" in str(ei.value)
    assert "malformed" in str(ei.value)
