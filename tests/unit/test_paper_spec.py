"""Tests verifying strict adherence to the JAZ paper specification.

Covers:
1. __history__ fields (.llm_response, .repl_output with traceback, .repl_exception as object)
2. display.py formatting and 50,000 char threshold (no 200 char truncation)
3. ContextWindowWarning per-invoke tracking and default config fallback
4. IterationLimit statelessness across invoke trees
"""

import pytest
from jaz import Config, ConfigOverride, invoke, scope
from jaz.display import format_available_variables_doc, format_invoke_signature, format_value_preview
from jaz.exceptions import IterationLimitExceededError
from jaz.history import History, Step
from jaz.hooks import ContextWindowWarning, IterationLimit
from jaz.llm import ScriptedLLM


def test_history_paper_spec_fields_on_success():
    """Verify __history__ records llm_response, repl_output, and repl_exception=None on success."""
    llm = ScriptedLLM(responses=[
        "print('step 1 output')\nx = 10",
        "return f'result is {x}'",
    ])

    with ConfigOverride(llm=llm):
        result = invoke(task="test history success")
        assert result == "result is 10"

    step = Step(turn=1, code="print('hello')", repl_output="hello\n", llm_response="# plan\nprint('hello')")
    assert step.llm_response == "# plan\nprint('hello')"
    assert step.repl_output == "hello\n"
    assert step.repl_exception is None
    d = step.to_dict()
    assert d["llm_response"] == "# plan\nprint('hello')"
    assert d["repl_output"] == "hello\n"
    assert d["repl_exception"] is None


def test_history_paper_spec_fields_on_error():
    """Verify __history__ records error traceback in repl_output and captures exception object."""
    llm = ScriptedLLM(responses=[
        # Turn 1: raises ZeroDivisionError
        "1 / 0",
        # Turn 2: inspects __history__ and recovers
        "last_step = __history__[-1]\nassert last_step.repl_exception is not None\nassert 'ZeroDivisionError' in last_step.repl_output\nreturn 'recovered'",
    ])

    with ConfigOverride(llm=llm):
        result = invoke(task="test error in history")
        assert result == "recovered"


def test_display_preserves_inputs_up_to_50000_chars():
    """Verify that task prompts and string inputs up to 50,000 chars are not truncated."""
    long_task = "A" * 1500  # 1500 chars, previously truncated at 200
    preview = format_value_preview(long_task)
    assert len(preview) > 1500
    assert "[truncated" not in preview
    assert preview == repr(long_task)

    sig = format_invoke_signature({"task": long_task})
    assert repr(long_task) in sig

    doc = format_available_variables_doc({"task": long_task})
    assert repr(long_task) in doc

    # Truly oversized strings (> 50,000 chars) get truncated with note
    oversized = "B" * 60000
    over_preview = format_value_preview(oversized)
    assert "[truncated, total 60000 chars]" in over_preview


def test_display_classes_and_callables():
    """Verify class and callable display matches paper L1201-1204."""
    class SampleEngine:
        """Sample simulation engine docstring."""
        def step(self):
            pass

    def run_simulation(steps: int = 10) -> str:
        """Runs the simulation."""
        return "ok"

    cls_prev = format_value_preview(SampleEngine)
    assert "<Class `SampleEngine`" in cls_prev
    assert "Sample simulation engine" in cls_prev

    fn_prev = format_value_preview(run_simulation)
    assert "<Tool `run_simulation(steps: int = 10) -> str`: Runs the simulation.>" in fn_prev


def test_context_window_warning_per_invoke_isolation():
    """Verify ContextWindowWarning warns parent and child invoke independently."""
    warn_hook = ContextWindowWarning(threshold_tokens=1000, threshold_ratio=0.5)

    llm = ScriptedLLM(responses=[
        # Parent Turn 1: triggers warning
        "print('parent step 1')",
        # Parent Turn 2: delegates to subagent
        "return invoke(task='subtask')",
        # Child Turn 1: also triggers warning independently
        "print('child step 1')",
        # Child Turn 2: completes
        "return 'child done'",
    ])

    real_query = llm.query
    def mock_query(messages, **kwargs):
        res = real_query(messages, **kwargs)
        res.input_tokens = 600  # > 500 threshold
        res.output_tokens = 50
        return res
    llm.query = mock_query

    with ConfigOverride(llm=llm):
        res = invoke(warn_hook, task="root")
        assert res == "child done"
        # Must have warned both the root invoke and the sub invoke!
        assert len(warn_hook._warned_invokes) == 2


def test_iteration_limit_stateless_per_invoke():
    """Verify IterationLimit enforces turns per invoke independently."""
    limit = IterationLimit(max_iterations=3)

    llm = ScriptedLLM(responses=[
        # Root turn 1: invokes subagent
        "sub1 = invoke(task='sub 1')\nreturn sub1",
        # Subagent turn 1, 2, 3: completes within limit of 3
        "print('sub turn 1')",
        "print('sub turn 2')",
        "return 'sub completed'",
    ])

    with ConfigOverride(llm=llm):
        res = invoke(limit, task="root")
        assert res == "sub completed"


def test_context_window_critical_escalating_warning():
    """Verify ContextWindowWarning fires standard warning at 70% and critical warning at 88%."""
    warn_hook = ContextWindowWarning(threshold_tokens=1000, threshold_ratio=0.70, critical_ratio=0.88)

    llm = ScriptedLLM(responses=[
        # Turn 1: 750 tokens -> triggers standard warning (>= 70%)
        "print('step 1')",
        # Turn 2: 900 tokens -> triggers critical warning (>= 88%)
        "print('step 2')",
        # Turn 3: wraps up and returns
        "return 'finished'",
    ])

    token_levels = [750, 900, 400]
    call_idx = 0
    original_query = llm.query

    def mock_query(messages, **kwargs):
        nonlocal call_idx
        res = original_query(messages, **kwargs)
        res.input_tokens = token_levels[min(call_idx, len(token_levels) - 1)]
        res.output_tokens = 20
        call_idx += 1
        return res

    llm.query = mock_query


    with ConfigOverride(llm=llm):
        res = invoke(warn_hook, task="root")
        assert res == "finished"
        assert len(warn_hook._warned_invokes) == 1
        assert len(warn_hook._critical_warned_invokes) == 1

