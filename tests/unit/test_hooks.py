"""Unit tests for JAZ hooks: BudgetPool, IterationLimit, ReturnType, etc."""

import pytest
from jaz import ConfigOverride, invoke
from jaz.exceptions import BudgetPoolExhaustedError, IterationLimitExceededError, RecursionLimitExceededError
from jaz.hooks import BudgetPool, IterationLimit, RecursionLimit, ReturnType, ValidateReturn
from jaz.llm import ScriptedLLM


def test_iteration_limit():
    llm = ScriptedLLM(responses=[
        "print('loop 1')",
        "print('loop 2')",
        "print('loop 3')",
    ])
    with ConfigOverride(llm=llm):
        with pytest.raises(IterationLimitExceededError):
            invoke(IterationLimit(max_iterations=2), task="infinite")


def test_recursion_limit():
    llm = ScriptedLLM(responses=[
        "return invoke(task='sub 1')",
        "return invoke(task='sub 2')",
        "return invoke(task='sub 3')",
    ])
    with ConfigOverride(llm=llm):
        with RecursionLimit(max_depth=2):
            with pytest.raises(RecursionLimitExceededError):
                invoke(task="root")


def test_return_type_rejection_and_retry():
    llm = ScriptedLLM(responses=[
        # Turn 1: returns string instead of float (should be rejected by ReturnType)
        "return 'not a float'",
        # Turn 2: returns proper float
        "return 3.1415",
    ])
    with ConfigOverride(llm=llm):
        result = invoke(ReturnType(float), task="get pi")
        assert result == pytest.approx(3.1415)


def test_budget_pool_shared_across_recursion():
    pool = BudgetPool(max_tokens=2000)
    # LLM queries generate tokens
    llm = ScriptedLLM(responses=[
        "return invoke(task='sub')",
        "return 42",
    ])
    with pool:
        with ConfigOverride(llm=llm):
            res = invoke(task="parent")
            assert res == 42
            assert pool.current_tokens > 0


def test_subagent_hook_inheritance_and_ui():
    from jaz.ui import TerminalLiveUI
    ui = TerminalLiveUI()

    llm = ScriptedLLM(responses=[
        # Root agent invokes subagent
        "sub_res = invoke(task='compute subtask')\nreturn f'Root got: {sub_res}'",
        # Subagent computes and returns
        "return 'subtask result'",
    ])

    with ConfigOverride(llm=llm):
        result = invoke(ui, task="main task")
        assert result == "Root got: subtask result"

    # UI must have recorded both root agent and subagent!
    assert len(ui.invokes) == 2
    root_id = [k for k, v in ui.invokes.items() if v["depth"] == 1][0]
    sub_id = [k for k, v in ui.invokes.items() if v["depth"] == 2][0]

    assert ui.invokes[root_id]["status"] == "completed"
    assert ui.invokes[sub_id]["status"] == "completed"
    assert ui.invokes[sub_id]["parent_id"] == root_id
    assert "compute subtask" in ui.invokes[sub_id]["task_preview"]

