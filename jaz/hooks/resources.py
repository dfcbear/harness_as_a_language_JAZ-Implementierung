"""Resource control hooks: BudgetPool, IterationLimit, RecursionLimit, ContextWindowWarning."""

import threading
from typing import Optional

from ..exceptions import (
    BudgetPoolExhaustedError,
    IterationLimitExceededError,
    RecursionLimitExceededError,
)
from .actions import AddMessages
from .dispatcher import Hook
from .events import InvokeEnter, LLMQueryExit, REPLExecEnter


class BudgetPool(Hook):
    """Shared budget pool enforcing cost or token limits across an invoke tree."""

    def __init__(self, max_cost: Optional[float] = None, max_tokens: Optional[int] = None):
        super().__init__()
        self.max_cost = max_cost
        self.max_tokens = max_tokens
        self.current_cost = 0.0
        self.current_tokens = 0
        self._lock = threading.Lock()

    def record_usage(self, cost: float, tokens: int) -> None:
        with self._lock:
            self.current_cost += cost
            self.current_tokens += tokens
            if self.max_cost is not None and self.current_cost > self.max_cost:
                raise BudgetPoolExhaustedError(
                    f"BudgetPool exceeded max cost: ${self.current_cost:.4f} > ${self.max_cost:.4f}"
                )
            if self.max_tokens is not None and self.current_tokens > self.max_tokens:
                raise BudgetPoolExhaustedError(
                    f"BudgetPool exceeded max tokens: {self.current_tokens} > {self.max_tokens}"
                )

    def on_llm_query_exit(self, event: LLMQueryExit):
        tokens = event.input_tokens + event.output_tokens
        self.record_usage(event.cost, tokens)
        return None

    def remaining_cost(self) -> Optional[float]:
        if self.max_cost is None:
            return None
        return max(0.0, self.max_cost - self.current_cost)

    def __repr__(self) -> str:
        return (
            f"BudgetPool(cost=${self.current_cost:.4f}/{self.max_cost}, "
            f"tokens={self.current_tokens}/{self.max_tokens})"
        )


class IterationLimit(Hook):
    """Limits the number of execution turns per invoke call."""

    def __init__(self, max_iterations: int = 30):
        super().__init__()
        self.max_iterations = max_iterations

    def on_repl_exec_enter(self, event: REPLExecEnter):
        if event.turn > self.max_iterations:
            raise IterationLimitExceededError(
                f"Iteration limit reached: turn {event.turn} > {self.max_iterations}"
            )
        return None


class RecursionLimit(Hook):
    """Limits the maximum nesting depth of recursive sub-invokes."""

    def __init__(self, max_depth: int = 5):
        super().__init__()
        self.max_depth = max_depth

    def on_invoke_enter(self, event: InvokeEnter):
        if event.depth > self.max_depth:
            raise RecursionLimitExceededError(
                f"Recursion limit reached: depth {event.depth} > {self.max_depth}"
            )
        return None


class ContextWindowWarning(Hook):
    """Warns the agent when context window reaches a threshold, prompting delegation.
    
    This is the core mechanism used in the JAZ paper for tail-recursive delegation.
    Tracks warned state per invoke_id to allow sub-invokes to independently trigger.
    """

    def __init__(
        self,
        threshold_tokens: Optional[int] = None,
        threshold_ratio: float = 0.70,
        critical_ratio: float = 0.88,
        warning_message: Optional[str] = None,
    ):
        super().__init__()
        self.threshold_tokens = threshold_tokens
        self.threshold_ratio = threshold_ratio
        self.critical_ratio = critical_ratio
        self.warning_message = warning_message or (
            "Your context window is close to full. You must finish your REPL session now by delegating all remaining work to a subagent.\n\n"
            "IMPORTANT NOTES:\n"
            "- You must give the subagent both the previous agent's REPL history (`prev_history`) if available, as well as your own REPL history (`__history__`), so that existing work is not lost.\n\n"
            "Follow this template:\n"
            "```python\n"
            "return invoke(\n"
            "    task = remaining_task,\n"
            "    prev_history = globals().get('prev_history', []) + __history__,\n"
            "    state = ...,\n"
            "    prev_progress_summary = \"So far, ...\",\n"
            "    next_steps = \"Your next step is to ...\",\n"
            ")\n"
            "```"
        )
        self._warned_invokes: set[str] = set()
        self._critical_warned_invokes: set[str] = set()
        self._lock = threading.Lock()

    def on_llm_query_exit(self, event: LLMQueryExit):
        # Resolve limit
        limit = self.threshold_tokens
        if limit is None:
            try:
                from ..config import get_default_config
                cfg = get_default_config()
                if cfg and cfg.llm and cfg.llm.context_window:
                    limit = cfg.llm.context_window
            except Exception:
                limit = None

        if limit is None:
            return None

        total = event.input_tokens + event.output_tokens
        ratio = total / limit

        with self._lock:
            # Check critical threshold (e.g. 88%)
            if ratio >= self.critical_ratio and event.invoke_id not in self._critical_warned_invokes:
                self._critical_warned_invokes.add(event.invoke_id)
                self._warned_invokes.add(event.invoke_id)
                crit_msg = (
                    f"CRITICAL WARNING: Your context window is {ratio:.0%} full ({total:,} / {limit:,} tokens)!\n"
                    "You MUST execute `return invoke(..., prev_history=globals().get('prev_history', []) + __history__)` "
                    "in THIS REPL turn to prevent a hard context window overflow crash!"
                )
                return AddMessages([{"role": "user", "content": crit_msg}])

            # Check standard threshold (70%)
            if ratio >= self.threshold_ratio and event.invoke_id not in self._warned_invokes:
                self._warned_invokes.add(event.invoke_id)
                return AddMessages([{"role": "user", "content": self.warning_message}])

        return None



class BudgetForcing(Hook):
    """Injects a reminder when budget is running low."""

    def __init__(self, threshold_cost: float, message: Optional[str] = None):
        super().__init__()
        self.threshold_cost = threshold_cost
        self.message = message or (
            "REMINDER: Your budget is running low. Please wrap up and return your final answer."
        )
        self._triggered = False

    def on_llm_query_exit(self, event: LLMQueryExit):
        if self._triggered:
            return None
        # Evaluated when combined with BudgetPool
        return None
