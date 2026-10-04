"""REPL execution event definitions."""

from dataclasses import dataclass, field
import time
from typing import Any, Optional


@dataclass
class REPLExecEnter:
    """Triggered before executing code in the REPL."""
    invoke_id: str
    code: str
    turn: int
    skip_execution: bool = False
    lint_error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class REPLExecExit:
    """Triggered after code finishes executing in the REPL."""
    invoke_id: str
    turn: int
    stdout: str
    stderr: str
    return_value: Any = None
    has_returned: bool = False
    error: Optional[str] = None
    exception: Optional[BaseException] = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)
