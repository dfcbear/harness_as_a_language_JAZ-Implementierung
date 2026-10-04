"""Hook base class and dispatcher."""

from typing import Any, List, Optional
from .actions import Abort, AddMessages, DropMessages, DropVariables, HookAction
from .events import (
    InvokeEnter,
    InvokeExit,
    InvokeSend,
    LLMQueryEnter,
    LLMQueryExit,
    REPLExecEnter,
    REPLExecExit,
)


class Hook:
    """Base class for all JAZ hooks.
    
    Can be used locally (passed as positional argument to invoke)
    or scoped (used as context manager: with MyHook(): ...).
    """

    def on_invoke_enter(self, event: InvokeEnter) -> Optional[HookAction]:
        return None

    def on_invoke_send(self, event: InvokeSend) -> Optional[HookAction]:
        return None

    def on_invoke_exit(self, event: InvokeExit) -> None:
        pass

    def on_llm_query_enter(self, event: LLMQueryEnter) -> Optional[HookAction]:
        return None

    def on_llm_stream_chunk(self, invoke_id: str, chunk: str, total_chunks: int) -> None:
        """Called progressively as streaming tokens arrive from the LLM."""
        pass

    def on_llm_query_exit(self, event: LLMQueryExit) -> Optional[HookAction]:
        return None

    def on_repl_exec_enter(self, event: REPLExecEnter) -> Optional[HookAction]:
        return None

    def on_repl_exec_exit(self, event: REPLExecExit) -> Optional[HookAction]:
        return None

    # Context manager support for scoped activation
    def __enter__(self):
        from ..scope import _enter_scoped_hook
        self._scope_token = _enter_scoped_hook(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from ..scope import _exit_scoped_hook
        _exit_scoped_hook(self, self._scope_token)
        return False


class HookDispatcher:
    """Dispatches events to a combined set of local and scoped hooks.

    Optimization: bound methods are pre-resolved at construction time and
    hooks that only carry the base Hook no-op implementation are filtered
    out, so dispatch loops iterate exclusively over hooks that actually
    override a given callback.  This eliminates per-call getattr overhead
    and avoids calling no-op methods.
    """

    def __init__(self, hooks: List[Hook]):
        self.hooks = list(hooks)
        # Pre-resolve bound methods, keeping only hooks whose class
        # actually overrides the base Hook implementation.
        self._invoke_enter = [
            h.on_invoke_enter for h in hooks
            if type(h).on_invoke_enter is not Hook.on_invoke_enter
        ]
        self._invoke_send = [
            h.on_invoke_send for h in hooks
            if type(h).on_invoke_send is not Hook.on_invoke_send
        ]
        self._invoke_exit = [
            h.on_invoke_exit for h in hooks
            if type(h).on_invoke_exit is not Hook.on_invoke_exit
        ]
        self._llm_query_enter = [
            h.on_llm_query_enter for h in hooks
            if type(h).on_llm_query_enter is not Hook.on_llm_query_enter
        ]
        self._llm_stream_chunk = [
            h.on_llm_stream_chunk for h in hooks
            if type(h).on_llm_stream_chunk is not Hook.on_llm_stream_chunk
        ]
        self._llm_query_exit = [
            h.on_llm_query_exit for h in hooks
            if type(h).on_llm_query_exit is not Hook.on_llm_query_exit
        ]
        self._repl_exec_enter = [
            h.on_repl_exec_enter for h in hooks
            if type(h).on_repl_exec_enter is not Hook.on_repl_exec_enter
        ]
        self._repl_exec_exit = [
            h.on_repl_exec_exit for h in hooks
            if type(h).on_repl_exec_exit is not Hook.on_repl_exec_exit
        ]

    def dispatch_invoke_enter(self, event: InvokeEnter) -> List[HookAction]:
        methods = self._invoke_enter
        if not methods:
            return []
        actions: List[HookAction] = []
        for method in methods:
            action = method(event)
            if isinstance(action, Abort):
                raise action
            if action is not None:
                actions.append(action)
        return actions

    def dispatch_invoke_send(self, event: InvokeSend) -> List[HookAction]:
        methods = self._invoke_send
        if not methods:
            return []
        actions: List[HookAction] = []
        for method in methods:
            action = method(event)
            if isinstance(action, Abort):
                raise action
            if action is not None:
                actions.append(action)
        return actions

    def dispatch_invoke_exit(self, event: InvokeExit) -> None:
        for method in self._invoke_exit:
            try:
                method(event)
            except Exception:
                pass

    def dispatch_llm_query_enter(self, event: LLMQueryEnter) -> List[HookAction]:
        methods = self._llm_query_enter
        if not methods:
            return []
        actions: List[HookAction] = []
        for method in methods:
            action = method(event)
            if isinstance(action, Abort):
                raise action
            if action is not None:
                actions.append(action)
        return actions

    def dispatch_llm_stream_chunk(self, invoke_id: str, chunk: str, total_chunks: int) -> None:
        for method in self._llm_stream_chunk:
            try:
                method(invoke_id, chunk, total_chunks)
            except Exception:
                pass

    def dispatch_llm_query_exit(self, event: LLMQueryExit) -> List[HookAction]:
        methods = self._llm_query_exit
        if not methods:
            return []
        actions: List[HookAction] = []
        for method in methods:
            action = method(event)
            if isinstance(action, Abort):
                raise action
            if action is not None:
                actions.append(action)
        return actions

    def dispatch_repl_exec_enter(self, event: REPLExecEnter) -> List[HookAction]:
        methods = self._repl_exec_enter
        if not methods:
            return []
        actions: List[HookAction] = []
        for method in methods:
            action = method(event)
            if isinstance(action, Abort):
                raise action
            if action is not None:
                actions.append(action)
        return actions

    def dispatch_repl_exec_exit(self, event: REPLExecExit) -> List[HookAction]:
        methods = self._repl_exec_exit
        if not methods:
            return []
        actions: List[HookAction] = []
        for method in methods:
            action = method(event)
            if isinstance(action, Abort):
                raise action
            if action is not None:
                actions.append(action)
        return actions
