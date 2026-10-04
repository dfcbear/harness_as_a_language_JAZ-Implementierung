"""Dynamic scoping for JAZ.

Variables, hooks, and configurations defined within a `with scope(...)` or
hook context manager are dynamically inherited by all subsequent invoke calls
and their recursive sub-invokes.
"""

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Dict, List, Optional, Tuple

# Context variables storing dynamic scope
_scoped_vars: ContextVar[Dict[str, Any]] = ContextVar("jaz_scoped_vars", default={})
_scoped_hooks: ContextVar[List[Any]] = ContextVar("jaz_scoped_hooks", default=[])
_scoped_config: ContextVar[Optional[Any]] = ContextVar("jaz_scoped_config", default=None)
_call_depth: ContextVar[int] = ContextVar("jaz_call_depth", default=0)
_parent_invoke_id: ContextVar[Optional[str]] = ContextVar("jaz_parent_invoke_id", default=None)


class Scope:
    """Context manager for dynamically scoping variables to invoke() calls."""

    def __init__(self, **variables: Any):
        self.variables = variables
        self._tokens: List[Tuple[ContextVar, Token]] = []

    def __enter__(self):
        current = _scoped_vars.get()
        merged = {**current, **self.variables}
        token = _scoped_vars.set(merged)
        self._tokens.append((_scoped_vars, token))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for var, token in reversed(self._tokens):
            var.reset(token)
        self._tokens.clear()
        return False


def scope(**variables: Any) -> Scope:
    """Create a dynamic scope for variables accessible to invoke and sub-invokes.
    
    Example:
        with scope(web_search=web_search, db=database):
            invoke(task="Find relevant papers")
    """
    return Scope(**variables)


def get_scoped_variables() -> Dict[str, Any]:
    """Retrieve currently active dynamically scoped variables."""
    return dict(_scoped_vars.get())


def get_scoped_hooks() -> List[Any]:
    """Retrieve currently active dynamically scoped hooks."""
    return list(_scoped_hooks.get())


def get_scoped_config() -> Optional[Any]:
    """Retrieve currently active dynamically scoped ConfigOverride."""
    return _scoped_config.get()


def get_current_depth() -> int:
    """Get current invoke recursion depth (0 for user code, 1 for top-level invoke)."""
    return _call_depth.get()


def get_parent_invoke_id() -> Optional[str]:
    """Get the invoke_id of the parent invoke, if currently within an invoke."""
    return _parent_invoke_id.get()


def _enter_scoped_hook(hook: Any) -> Token:
    """Internal helper to register a hook activated via context manager."""
    current = list(_scoped_hooks.get())
    current.append(hook)
    return _scoped_hooks.set(current)


def _exit_scoped_hook(hook: Any, token: Token) -> None:
    """Internal helper to deregister a scoped hook."""
    _scoped_hooks.reset(token)


def _enter_scoped_config(config_override: Any) -> Token:
    """Internal helper to activate a scoped ConfigOverride."""
    return _scoped_config.set(config_override)


def _exit_scoped_config(token: Token) -> None:
    """Internal helper to deactivate a scoped ConfigOverride."""
    _scoped_config.reset(token)
