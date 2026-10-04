"""Invoke event definitions."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional


@dataclass
class InvokeEnter:
    """Triggered when an invoke() call begins."""
    invoke_id: str
    parent_id: Optional[str]
    depth: int
    inputs: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class InvokeSend:
    """Triggered when a message is sent to/from the agent loop."""
    invoke_id: str
    message: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class InvokeExit:
    """Triggered when an invoke() call completes or raises."""
    invoke_id: str
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)
