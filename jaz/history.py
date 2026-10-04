"""Interaction history representations for JAZ.

Each step in an agent REPL session is captured in a Step object.
The list of steps forms the __history__ variable made available in the REPL.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Iterator, List, Optional


@dataclass
class Step:
    """Represents a single execution step in the REPL.
    
    Verbatim fields from JAZ paper (Appendix E.3):
    - llm_response: Full model response for that iteration containing code.
    - repl_output: Printed output from that iteration, including any error traceback.
    - repl_exception: The exception object raised if that iteration hit an error, else None.
    """
    turn: int
    code: str
    repl_output: str
    llm_response: str = ""
    repl_exception: Optional[Any] = None
    error: Optional[str] = None
    return_value: Any = None
    has_returned: bool = False
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.llm_response and self.code:
            self.llm_response = self.code
        if not self.code and self.llm_response:
            self.code = self.llm_response

    def __repr__(self) -> str:
        code_preview = self.code.strip().replace("\n", "; ")
        if len(code_preview) > 50:
            code_preview = code_preview[:47] + "..."
        out_preview = self.repl_output.strip().replace("\n", "; ")
        if len(out_preview) > 50:
            out_preview = out_preview[:47] + "..."
        return f"Step(turn={self.turn}, code={code_preview!r}, repl_output={out_preview!r})"

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "code": self.code,
            "llm_response": self.llm_response,
            "repl_output": self.repl_output,
            "repl_exception": str(self.repl_exception) if self.repl_exception is not None else None,
            "error": self.error,
            "has_returned": self.has_returned,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
        }


class History(list):
    """A list of Step objects representing the interaction history.
    
    Provides convenient search and inspection methods for agents and harnesses.
    """

    def __init__(self, steps: Optional[List[Step]] = None):
        super().__init__(steps or [])

    def append_step(
        self,
        turn: int,
        code: str,
        repl_output: str,
        error: Optional[str] = None,
        return_value: Any = None,
        has_returned: bool = False,
        execution_time: float = 0.0,
        llm_response: str = "",
        repl_exception: Optional[Any] = None,
    ) -> Step:
        step = Step(
            turn=turn,
            code=code,
            repl_output=repl_output,
            llm_response=llm_response or code,
            repl_exception=repl_exception,
            error=error,
            return_value=return_value,
            has_returned=has_returned,
            execution_time=execution_time,
        )
        self.append(step)
        return step

    def search(self, query: str, case_sensitive: bool = False) -> List[tuple[int, Step, int]]:
        """Search across repl_output of steps.
        
        Returns a list of tuples: (step_index, step, match_position).
        """
        results = []
        target = query if case_sensitive else query.lower()
        for idx, step in enumerate(self):
            haystack = step.repl_output if case_sensitive else step.repl_output.lower()
            pos = haystack.find(target)
            if pos >= 0:
                results.append((idx, step, pos))
        return results

    def text_summary(self) -> str:
        lines = []
        for step in self:
            lines.append(f"[Step {step.turn}]")
            lines.append(f">>> {step.code}")
            if step.repl_output:
                lines.append(step.repl_output)
            if step.error:
                lines.append(f"Error: {step.error}")
        return "\n".join(lines)

    def to_json(self) -> list[dict]:
        return [step.to_dict() for step in self]
