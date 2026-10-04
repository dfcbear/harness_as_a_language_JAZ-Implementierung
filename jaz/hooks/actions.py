"""Actions that hooks can return or raise to modify loop state or abort."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class HookAction:
    """Base class for hook actions."""
    pass


@dataclass
class AddMessages(HookAction):
    """Action to append messages to the LLM conversation."""
    messages: List[Dict[str, Any]]

    def __init__(self, *messages: Any):
        if len(messages) == 1 and isinstance(messages[0], list):
            self.messages = messages[0]
        else:
            self.messages = list(messages)


@dataclass
class DropMessages(HookAction):
    """Action to drop messages by index from the LLM conversation."""
    indices: List[int]

    def __init__(self, *indices: int):
        if len(indices) == 1 and isinstance(indices[0], list):
            self.indices = indices[0]
        else:
            self.indices = list(indices)


@dataclass
class DropVariables(HookAction):
    """Action to drop variables from the REPL namespace."""
    var_names: List[str]

    def __init__(self, *names: str):
        if len(names) == 1 and isinstance(names[0], list):
            self.var_names = names[0]
        else:
            self.var_names = list(names)


@dataclass
class Abort(HookAction, Exception):
    """Action to abort the invoke loop immediately."""
    reason: str = "Execution aborted by hook"

    def __str__(self) -> str:
        return self.reason
