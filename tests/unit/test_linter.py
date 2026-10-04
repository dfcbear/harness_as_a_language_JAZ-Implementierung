"""Unit tests for PythonLinterHook (Auto-Repair and Pre-Execution Diagnostics)."""

import pytest
from jaz import ConfigOverride, invoke, scope
from jaz.hooks.linter import PythonLinterHook
from jaz.llm import ScriptedLLM


def test_linter_auto_repair_conversational_text():
    noisy_response = """Here is the code to solve your task:
result = 100 + 42
return result
Hope this helps!
"""
    llm = ScriptedLLM(responses=[noisy_response])
    linter = PythonLinterHook(auto_fix=True)

    with ConfigOverride(llm=llm):
        res = invoke(linter, task="compute answer")
        assert res == 142


def test_linter_detects_undefined_tool_and_suggests_fix():
    def get_user_profile(user_id: int):
        return {"id": user_id, "name": "Alice"}

    # Agent hallucinates name 'get_user_info' instead of 'get_user_profile'
    llm = ScriptedLLM(responses=[
        # Turn 1: hallucinated tool name -> intercepted by linter
        "profile = get_user_info(1)\nreturn profile",
        # Turn 2: corrected after seeing linter suggestion
        "profile = get_user_profile(1)\nreturn profile['name']",
    ])

    linter = PythonLinterHook(auto_fix=True, check_undefined_tools=True, suggest_similar_names=True)

    with scope(get_user_profile=get_user_profile):
        with ConfigOverride(llm=llm):
            res = invoke(linter, task="Get Alice")
            assert res == "Alice"


def test_linter_blocks_disallowed_modules():
    llm = ScriptedLLM(responses=[
        # Turn 1: tries to import disallowed module
        "import subprocess\nreturn 'done'",
        # Turn 2: uses allowed math instead
        "import math\nreturn math.sqrt(16)",
    ])

    linter = PythonLinterHook(disallowed_modules=["subprocess", "shutil"])

    with ConfigOverride(llm=llm):
        res = invoke(linter, task="test security")
        assert res == 4.0


def test_linter_allows_imported_calls_and_thread_pool():
    code_with_imports = """from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as pool:
    f = pool.submit(int, "42")
    return f.result()
"""
    llm = ScriptedLLM(responses=[code_with_imports])
    linter = PythonLinterHook(auto_fix=True, check_undefined_tools=True)

    with ConfigOverride(llm=llm):
        res = invoke(linter, task="run thread pool")
        assert res == 42

