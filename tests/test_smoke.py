import re


def test_import_genome():
    import genome
    # Sanity check: version exists and looks like a SemVer string. Avoids
    # hardcoding the literal so this test does not need to be touched on
    # every release.
    assert isinstance(genome.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", genome.__version__) is not None
