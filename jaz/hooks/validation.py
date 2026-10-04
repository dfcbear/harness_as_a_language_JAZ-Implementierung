"""Validation hooks for return values and REPL code."""

from typing import Any, Callable, Tuple, Type, Union

from ..exceptions import ReturnRejectedError, ValidationError
from .actions import AddMessages
from .dispatcher import Hook
from .events import REPLExecEnter, REPLExecExit


class ReturnType(Hook):
    """Enforces that the return value of invoke matches expected type(s)."""

    def __init__(self, expected_type: Union[Type, Tuple[Type, ...]]):
        super().__init__()
        self.expected_type = expected_type

    def on_repl_exec_exit(self, event: REPLExecExit):
        if not event.has_returned:
            return None

        val = event.return_value
        if not isinstance(val, self.expected_type):
            event.has_returned = False
            msg = (
                f"Validation Error: Return value was rejected because its type "
                f"`{type(val).__name__}` does not match required type `{self.expected_type}`. "
                f"Received value: {repr(val)[:200]}. Please correct the return value."
            )
            return AddMessages([{"role": "user", "content": msg}])
        return None


class ValidateReturn(Hook):
    """Validates returned value using a custom validation function."""

    def __init__(self, validator_fn: Callable[[Any], Union[bool, Tuple[bool, str]]]):
        super().__init__()
        self.validator_fn = validator_fn

    def on_repl_exec_exit(self, event: REPLExecExit):
        if not event.has_returned:
            return None

        try:
            res = self.validator_fn(event.return_value)
            if isinstance(res, tuple):
                is_valid, msg = res
            else:
                is_valid, msg = bool(res), "Validation check failed."

            if not is_valid:
                event.has_returned = False
                return AddMessages([
                    {"role": "user", "content": f"Validation Error: {msg} Keep working."}
                ])
        except Exception as e:
            event.has_returned = False
            return AddMessages([
                {"role": "user", "content": f"Validation Error during return check: {e}. Keep working."}
            ])
        return None


class ValidateREPLCode(Hook):
    """Validates code string before execution in the REPL."""

    def __init__(self, validator_fn: Callable[[str], Union[bool, Tuple[bool, str]]]):
        super().__init__()
        self.validator_fn = validator_fn

    def on_repl_exec_enter(self, event: REPLExecEnter):
        res = self.validator_fn(event.code)
        if isinstance(res, tuple):
            is_valid, msg = res
        else:
            is_valid, msg = bool(res), "Code failed validation check."

        if not is_valid:
            raise ValidationError(msg)
        return None
