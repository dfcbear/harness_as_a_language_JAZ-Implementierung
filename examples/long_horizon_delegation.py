"""Example: Long-horizon execution via tail-recursive delegation.

Matches the StuLife pattern from Paper Section 4.1:
When an agent's context fills up, it delegates remaining work to a subagent
passing the accumulated history losslessly by reference.
"""

from jaz import invoke, scope
from jaz.hooks import ContextWindowWarning, IterationLimit


def main():
    tasks = [
        {"id": 1, "note": "Meeting room is Beta-402"},
        {"id": 2, "action": "Working on research summary"},
        {"id": 3, "action": "Preparing slides"},
        {"id": 4, "query": "Which room was scheduled in task 1?"},
    ]

    print("Running Long-Horizon Agent with tail-recursive delegation...")
    # ContextWindowWarning set to prompt delegation when context grows
    warning_hook = ContextWindowWarning(
        threshold_tokens=2000,
        warning_message=(
            "Context window limit approaching! Please delegate remaining tasks to a subagent: "
            "`return invoke(task=task, task_list=remaining, prev_history=globals().get('prev_history', []) + __history__)`"
        ),
    )

    result = invoke(
        warning_hook,
        IterationLimit(20),
        task="Process each task in task_list in order. Answer the final query.",
        task_list=tasks,
    )

    print(f"Final Answer: {result}")


if __name__ == "__main__":
    main()
