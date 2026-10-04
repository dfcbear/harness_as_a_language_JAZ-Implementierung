"""Hook event definitions."""

from dataclasses import dataclass, field
import time
from typing import Any

from .invoke import InvokeEnter, InvokeExit, InvokeSend
from .llm_query import LLMQueryEnter, LLMQueryExit, LLMQuerySend
from .repl_execution import REPLExecEnter, REPLExecExit


@dataclass
class Completed:
    """Event fired when an entire session or task completes."""
    invoke_id: str
    result: Any
    timestamp: float = field(default_factory=time.time)


__all__ = [
    "InvokeEnter",
    "InvokeSend",
    "InvokeExit",
    "LLMQueryEnter",
    "LLMQuerySend",
    "LLMQueryExit",
    "REPLExecEnter",
    "REPLExecExit",
    "Completed",
]
