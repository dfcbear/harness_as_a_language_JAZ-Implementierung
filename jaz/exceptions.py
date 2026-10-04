"""Exceptions for the JAZ framework."""

from typing import Any, Optional


class JazError(Exception):
    """Base exception for all JAZ runtime errors."""
    pass


class FatalError(JazError):
    """Unrecoverable error that aborts the agent loop immediately."""
    pass


class AbortError(JazError):
    """Raised when an execution or hook explicitly aborts the loop."""
    def __init__(self, reason: str = "Execution aborted"):
        super().__init__(reason)
        self.reason = reason


class BudgetExhaustedError(FatalError):
    """Raised when an allocated budget has been exceeded."""
    pass


class BudgetPoolExhaustedError(BudgetExhaustedError):
    """Raised when a shared BudgetPool has been exhausted."""
    pass


class IterationLimitExceededError(FatalError):
    """Raised when the maximum number of REPL iterations is exceeded."""
    pass


class RecursionLimitExceededError(FatalError):
    """Raised when the maximum sub-agent recursion depth is exceeded."""
    pass


class ContextWindowExceededError(FatalError):
    """Raised when the model's context window length is exceeded."""
    pass


class ValidationError(JazError):
    """Raised when return value or code validation fails."""
    def __init__(self, message: str, invalid_value: Any = None):
        super().__init__(message)
        self.invalid_value = invalid_value


class ReturnRejectedError(ValidationError):
    """Raised when a return validator rejects a returned value."""
    pass
