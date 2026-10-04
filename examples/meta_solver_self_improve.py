"""Example: Continual Self-Improvement (Meta-Agent / Solver-Agent).

Matches the AppWorld continual self-improvement pattern from Paper Section 4.2:
The top-level invoke acts as a meta-agent that launches sub-invokes (solver agents),
evaluates their trajectory from __history__, and improves prompt and tools iteratively.
"""

from jaz import invoke, scope
from jaz.hooks import IterationLimit, RecursionLimit


class MockEnvironment:
    """Simulated environment exposing tasks and automated evaluation."""

    def __init__(self):
        self.tasks = [
            {"id": "t1", "input": "apple", "expected": "APPLE_PROCESSED"},
            {"id": "t2", "input": "banana", "expected": "BANANA_PROCESSED"},
            {"id": "t3", "input": "cherry", "expected": "CHERRY_PROCESSED"},
        ]
        self.cursor = 0

    def get_next_task(self):
        if self.cursor < len(self.tasks):
            t = self.tasks[self.cursor]
            self.cursor += 1
            return t
        return None

    def evaluate(self, task_id, output):
        for t in self.tasks:
            if t["id"] == task_id:
                passed = (output == t["expected"])
                return {"passed": passed, "feedback": "Match" if passed else "Did not add _PROCESSED suffix"}
        return {"passed": False, "feedback": "Unknown task"}


def main():
    env = MockEnvironment()

    meta_prompt = """
    You are a meta-agent tasked with continual self-improvement.
    Run each task by launching a subagent with `invoke(task=task, instructions=instructions)`.
    Instruct the subagent to return `(answer, __history__)`.
    Evaluate the result with `env.evaluate(task['id'], answer)`.
    If the subagent failed, inspect its trajectory and update the instructions or helper tools.
    When all tasks are completed, return a summary report.
    """

    print("Running Meta/Solver Continual Self-Improvement Agent...")
    result = invoke(
        IterationLimit(25),
        RecursionLimit(3),
        task=meta_prompt,
        env=env,
    )

    print(f"Meta-Agent Report: {result}")


if __name__ == "__main__":
    main()
