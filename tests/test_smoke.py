import re


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
