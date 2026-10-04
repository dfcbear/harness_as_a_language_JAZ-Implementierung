"""Hooks system for monitoring, controlling, and validating JAZ agents."""

from .actions import Abort, AddMessages, DropMessages, DropVariables, HookAction
from .dispatcher import Hook, HookDispatcher
from .events import (
    Completed,
    InvokeEnter,
    InvokeExit,
    InvokeSend,
    LLMQueryEnter,
    LLMQueryExit,
    LLMQuerySend,
    REPLExecEnter,
    REPLExecExit,
)
from .linter import PythonLinterHook
from .observability import (
    FileLogger,
    JsonlEventEmitter,
    PrintLogger,
    TrajectoryDirectoryRecorder,
    TrajectoryRecorder,
)
from .replay import TrajectoryReplay
from .resources import (
    BudgetForcing,
    BudgetPool,
    ContextWindowWarning,
    IterationLimit,
    RecursionLimit,
)
from .validation import ReturnType, ValidateREPLCode, ValidateReturn

__all__ = [
    # Base
    "Hook",
    "HookDispatcher",
    "HookAction",
    # Actions
    "Abort",
    "AddMessages",
    "DropMessages",
    "DropVariables",
    # Events
    "InvokeEnter",
    "InvokeSend",
    "InvokeExit",
    "LLMQueryEnter",
    "LLMQuerySend",
    "LLMQueryExit",
    "REPLExecEnter",
    "REPLExecExit",
    "Completed",
    # Resource control
    "BudgetPool",
    "IterationLimit",
    "RecursionLimit",
    "BudgetForcing",
    "ContextWindowWarning",
    # Validation
    "ReturnType",
    "ValidateReturn",
    "ValidateREPLCode",
    "PythonLinterHook",
    # Observability
    "PrintLogger",
    "FileLogger",
    "TrajectoryRecorder",
    "TrajectoryDirectoryRecorder",
    "JsonlEventEmitter",
    # Replay
    "TrajectoryReplay",
]
