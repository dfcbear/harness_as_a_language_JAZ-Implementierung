"""Types and data structures for REPL execution."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Return:
    """Sentinel wrapping an explicit return value from the REPL."""
    value: Any


@dataclass
class Raise:
    """Sentinel wrapping an exception raised during REPL execution."""
    exception: Exception
    traceback: str


@dataclass
class REPLResult:
    """Result of executing a code snippet in the REPL."""
    stdout: str
    stderr: str
    return_value: Any = None
    has_returned: bool = False
    error: Optional[str] = None
    exception: Optional[BaseException] = None
    execution_time: float = 0.0
