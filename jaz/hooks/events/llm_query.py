"""LLM Query event definitions."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional


@dataclass
class LLMQueryEnter:
    """Triggered before querying the LLM."""
    invoke_id: str
    messages: List[Dict[str, Any]]
    model: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMQuerySend:
    """Triggered when LLM payload is dispatched."""
    invoke_id: str
    payload: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMQueryExit:
    """Triggered after LLM responds."""
    invoke_id: str
    response: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    duration: float = 0.0
    error: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
