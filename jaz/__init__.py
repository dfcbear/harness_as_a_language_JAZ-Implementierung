"""JAZ: A Minimalist Agent Framework With Maximal Expressivity.

Based on "Harness as a Language" (arXiv:2609.26891).
"""

from .config import (
    Config,
    ConfigOverride,
    get_default_config,
    load_config_from_env_or_file,
    set_default_config,
)
from .core import invoke
from .history import History, Step
from .llm import BaseLLM, OpenAICompatibleLLM, ScriptedLLM
from .protocol import BaseProtocol, CodeOnlyProtocol, RawCodeProtocol
from .scope import get_current_depth, get_scoped_variables, scope

__all__ = [
    "invoke",
    "scope",
    "Config",
    "ConfigOverride",
    "get_default_config",
    "set_default_config",
    "load_config_from_env_or_file",
    "History",
    "Step",
    "BaseLLM",
    "OpenAICompatibleLLM",
    "ScriptedLLM",
    "BaseProtocol",
    "RawCodeProtocol",
    "CodeOnlyProtocol",
    "get_current_depth",
    "get_scoped_variables",
]

__version__ = "0.1.0"
