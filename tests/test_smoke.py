import re

import pytest


def test_import_genome():
    import genome

    assert isinstance(genome.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", genome.__version__) is not None


def test_version_matches_installed_metadata():
    """__version__ must agree with the installed distribution.

    Releases 1.0.4-1.0.6 shipped self-reporting "1.0.3" because the module
    string was bumped by hand and forgotten. The module now derives its version
    from package metadata; this test pins that invariant so a regression to a
    hardcoded string cannot silently desync again.
    """
    from importlib.metadata import version

    import genome

    assert genome.__version__ == version("genome-memory")


def _packaged_version():
    """The version in pyproject.toml, or a skip when run outside the checkout."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    project = root / "pyproject.toml"
    if not project.exists():
        pytest.skip("source checkout only: these files are not part of the wheel")
    with project.open("rb") as handle:
        return root, tomllib.load(handle)["project"]["version"]


def test_mcp_manifest_version_matches_pyproject():
    """`server.json` is one of two places the version is transcribed by hand.

    It sat at 1.0.6 while `pyproject.toml` moved to 1.1.0, so the MCP registry
    advertised an older release than the package and nothing failed anywhere -
    the same silent-desync class as the ``__version__`` bug above. Both the
    manifest's own version and every package entry inside it are pinned here.
    """
    import json

    root, expected = _packaged_version()
    manifest = root / "server.json"
    if not manifest.exists():
        pytest.skip("source checkout only: server.json is not part of the wheel")

    server = json.loads(manifest.read_text(encoding="utf-8"))
    assert server["version"] == expected
    for package in server.get("packages", []):
        assert package["version"] == expected


def test_citation_version_matches_pyproject():
    """`CITATION.cff` is the other one, and it is what GitHub and Zenodo read.

    A stale version here mis-cites the software in anyone's bibliography, which
    is worse than a stale number in a manifest: it ends up in someone's paper.
    """
    root, expected = _packaged_version()
    citation = root / "CITATION.cff"
    if not citation.exists():
        pytest.skip("source checkout only: CITATION.cff is not part of the wheel")

    # Regex rather than a YAML parser: pyyaml is not a dependency of this package
    # and a smoke test must not be the reason it becomes one.
    found = re.search(
        r"^version:\s*['\"]?([^'\"\s]+)", citation.read_text(encoding="utf-8"), re.M
    )
    assert found is not None, "CITATION.cff has no top-level version field"
    assert found.group(1) == expected
