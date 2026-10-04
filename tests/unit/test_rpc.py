"""Unit tests for the sandboxed SubprocessREPL and RPC tool proxying."""

import pytest
from jaz.repl.subprocess_repl import SubprocessREPL


def test_subprocess_basic_execution():
    repl = SubprocessREPL()
    try:
        res = repl.execute("print('HELLO FROM SANDBOX')")
        assert "HELLO FROM SANDBOX" in res.stdout
        assert not res.has_returned
    finally:
        repl.close()


def test_subprocess_state_and_return():
    repl = SubprocessREPL()
    try:
        # Turn 1
        res1 = repl.execute("val = 100")
        assert not res1.has_returned

        # Turn 2
        res2 = repl.execute("return val + 23")
        assert res2.has_returned
        assert res2.return_value == 123
    finally:
        repl.close()


def test_subprocess_host_tool_rpc_call():
    repl = SubprocessREPL()
    try:
        def host_multiplier(a: int, b: int) -> int:
            return a * b

        repl.set_variable("multiply", host_multiplier)
        res = repl.execute("result = multiply(6, 7)\nreturn result")
        assert res.has_returned
        assert res.return_value == 42
    finally:
        repl.close()


def test_host_tool_rpc_error_propagation():
    """Test that exceptions from host tools are propagated through RPC to the sandbox."""
    repl = SubprocessREPL()
    try:
        def failing_tool(msg: str) -> str:
            raise ValueError(f"Intentional tool failure: {msg}")

        repl.set_variable("failing_tool", failing_tool)
        res = repl.execute("result = failing_tool('boom')\nreturn result")
        # The error should be propagated - execution should not succeed
        assert not res.has_returned
        assert res.error is not None
        assert "Intentional tool failure: boom" in res.error
    finally:
        repl.close()


def test_host_tool_not_found_rpc_error():
    """Test that calling a non-existent host tool produces a clear error."""
    repl = SubprocessREPL()
    try:
        # Manually create a RemoteToolProxy for a tool that was never registered
        res = repl.execute(
            "from jaz.repl.rpc import RemoteToolProxy\n"
            "proxy = RemoteToolProxy('nonexistent_tool')\n"
            "result = proxy()\n"
            "return result"
        )
        assert not res.has_returned
        assert res.error is not None
        assert "not found" in res.error.lower()
    finally:
        repl.close()


def test_clean_shutdown_after_execution_error():
    """Test that SubprocessREPL shuts down cleanly after an execution error."""
    repl = SubprocessREPL()
    try:
        # Trigger a syntax error inside the sandbox
        res = repl.execute("this is not valid python !!!")
        assert res.error is not None
        assert "SyntaxError" in res.error
    finally:
        repl.close()

    # After close, the process should be cleaned up
    assert repl.process is None


def test_multiple_close_calls_safe():
    """Test that calling close() multiple times does not raise exceptions."""
    repl = SubprocessREPL()
    repl.close()
    # Second and third close should be safe no-ops
    repl.close()
    repl.close()
    assert repl.process is None


def test_subprocess_crash_handling():
    """Test that unexpected subprocess termination is handled gracefully."""
    repl = SubprocessREPL()
    try:
        # Simulate a crash by killing the subprocess
        repl.process.kill()
        repl.process.wait()

        # Attempting to execute after crash should return an error, not hang
        res = repl.execute("print('test')")
        assert res.error is not None
    finally:
        repl.close()
