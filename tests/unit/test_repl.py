"""Unit tests for the LocalPythonREPL."""

import pytest
from jaz.repl.local import LocalPythonREPL


def test_repl_state_persistence():
    repl = LocalPythonREPL()
    # Turn 1: define variable
    res1 = repl.execute("x = 42\nprint('x is defined')")
    assert "x is defined" in res1.stdout
    assert not res1.has_returned
    assert repl.get_variable("x") == 42

    # Turn 2: use variable from previous turn
    res2 = repl.execute("y = x * 2")
    assert repl.get_variable("y") == 84


def test_repl_top_level_return():
    repl = LocalPythonREPL()
    res = repl.execute("x = 10\nreturn x + 5")
    assert res.has_returned
    assert res.return_value == 15
    assert repl.get_variable("x") == 10


def test_repl_conditional_return():
    repl = LocalPythonREPL()
    code = """
items = [1, 2, 3, 4]
found = None
for item in items:
    if item == 3:
        return f"found_{item}"
"""
    res = repl.execute(code)
    assert res.has_returned
    assert res.return_value == "found_3"


def test_repl_error_capture():
    repl = LocalPythonREPL()
    res = repl.execute("1 / 0")
    assert not res.has_returned
    assert res.error is not None
    assert "ZeroDivisionError" in res.error


def test_repl_ensure_packages_availability():
    repl = LocalPythonREPL()
    # ensure_packages is callable in namespace
    res = repl.execute("ensure_packages('sys', 'os')\nreturn True")
    assert res.has_returned
    assert res.return_value is True

