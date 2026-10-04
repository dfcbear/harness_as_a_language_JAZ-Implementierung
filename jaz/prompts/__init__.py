"""Prompt templates for JAZ."""

from pathlib import Path


def get_default_system_prompt() -> str:
    prompt_file = Path(__file__).parent / "system.md"
    if prompt_file.is_file():
        return prompt_file.read_text(encoding="utf-8")
    return "You are an agent executing inside a Python REPL. Return results using `return <val>`."
