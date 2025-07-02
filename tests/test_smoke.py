"""Smoke test to check mlfsm module importability."""

import mlfsm


def test_import():
    """Check that mlfsm can be imported and has a __version__ attribute."""
    assert hasattr(mlfsm, "__version__")
