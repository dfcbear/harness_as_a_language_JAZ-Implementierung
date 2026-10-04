"""Base REPL interface for JAZ."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from .types import REPLResult


class BaseREPL(ABC):
    """Abstract base class for Python REPL environments."""

    @abstractmethod
    def execute(self, code: str) -> REPLResult:
        """Execute a code snippet in the REPL environment."""
        pass

    @abstractmethod
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the REPL namespace."""
        pass

    @abstractmethod
    def get_variable(self, name: str) -> Any:
        """Get a variable from the REPL namespace."""
        pass

    @abstractmethod
    def has_variable(self, name: str) -> bool:
        """Check if a variable exists in the REPL namespace."""
        pass

    @abstractmethod
    def delete_variable(self, name: str) -> None:
        """Remove a variable from the REPL namespace."""
        pass

    @abstractmethod
    def get_namespace(self) -> Dict[str, Any]:
        """Return the current namespace dictionary."""
        pass

    def close(self) -> None:
        """Clean up REPL resources."""
        pass
