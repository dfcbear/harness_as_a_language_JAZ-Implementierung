"""Unit tests for dynamic scoping in JAZ."""

import pytest
from jaz import scope
from jaz.scope import get_current_depth, get_scoped_variables


def test_scope_variables_lifecycle():
    assert get_scoped_variables() == {}

    with scope(a=1, b="test"):
        vars1 = get_scoped_variables()
        assert vars1["a"] == 1
        assert vars1["b"] == "test"

        # Nested scope overrides
        with scope(b="overridden", c=True):
            vars2 = get_scoped_variables()
            assert vars2["a"] == 1
            assert vars2["b"] == "overridden"
            assert vars2["c"] is True

        # Unwinds correctly
        vars3 = get_scoped_variables()
        assert vars3["b"] == "test"
        assert "c" not in vars3

    # Cleaned up after outer scope
    assert get_scoped_variables() == {}
