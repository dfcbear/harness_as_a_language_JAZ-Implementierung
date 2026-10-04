"""Unit tests for the core invoke loop using ScriptedLLM."""

import pytest
from jaz import ConfigOverride, invoke, scope
from jaz.llm import ScriptedLLM


def test_invoke_variables_and_return():
    llm = ScriptedLLM(responses=[
        "res = task.upper()\nreturn res"
    ])
    with ConfigOverride(llm=llm):
        result = invoke(task="hello world")
        assert result == "HELLO WORLD"


def test_invoke_history_is_variable():
    llm = ScriptedLLM(responses=[
        # Turn 1: compute step
        "step1 = num * 2\nprint(f'step1={step1}')",
        # Turn 2: inspect __history__
        "prev_code = __history__[0].code\nreturn (len(__history__), 'step1' in prev_code)",
    ])
    with ConfigOverride(llm=llm):
        result = invoke(num=21)
        assert result == (1, True)


def test_recursive_sub_invoke():
    parent_llm = ScriptedLLM(responses=[
        # Parent calls sub-invoke
        "sub_res = invoke(sub_task='inner job')\nreturn f'parent got: {sub_res}'",
        # Sub-invoke runs
        "return f'sub_{sub_task}'",
    ])
    with ConfigOverride(llm=parent_llm):
        result = invoke(task="outer job")
        assert result == "parent got: sub_inner job"


def test_scoped_tool_and_variables_inherited():
    def double(n: int) -> int:
        return n * 2

    llm = ScriptedLLM(responses=[
        "return double(val)",
    ])
    with scope(double=double, val=50):
        with ConfigOverride(llm=llm):
            result = invoke()
            assert result == 100


def test_parallel_sub_invokes_inside_repl():
    # Parent launches 2 subagents concurrently using ThreadPoolExecutor in REPL
    def sub_responder(messages):
        user_msg = messages[-1]["content"]
        if "task='track'" in user_msg:
            return "return 'TrackDone'"
        return "return 'CarDone'"

    llm = ScriptedLLM(responses=[
        # Turn 1: Parent runs ThreadPoolExecutor with invoke
        (
            "from concurrent.futures import ThreadPoolExecutor\n"
            "with ThreadPoolExecutor(max_workers=2) as ex:\n"
            "    f1 = ex.submit(invoke, task='track')\n"
            "    f2 = ex.submit(invoke, task='car')\n"
            "    r1 = f1.result()\n"
            "    r2 = f2.result()\n"
            "return {'track': r1, 'car': r2}"
        ),
        sub_responder,
        sub_responder,
    ])
    with ConfigOverride(llm=llm):
        result = invoke(task="build game")
        assert result == {"track": "TrackDone", "car": "CarDone"}


def test_load_config_env_expansion(monkeypatch):
    import os
    from jaz.config import _expand_env_value, load_config_from_env_or_file
    monkeypatch.setenv("TEST_JAZ_VAR", "my_secret_token_123")
    assert _expand_env_value("${TEST_JAZ_VAR}") == "my_secret_token_123"
    assert _expand_env_value("${NONEXISTENT:-fallback}") == "fallback"

