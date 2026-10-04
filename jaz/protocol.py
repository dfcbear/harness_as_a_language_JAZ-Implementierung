"""Protocols defining how the LLM and the REPL communicate.

Handles parsing code from raw LLM responses and formatting REPL outputs/errors
back to the model.
"""

from abc import ABC, abstractmethod
import re
from typing import Optional


class BaseProtocol(ABC):
    """Abstract base class for JAZ communication protocols."""

    def __init__(self, max_output_chars: int = 12000):
        self.max_output_chars = max_output_chars

    @abstractmethod
    def extract_code(self, raw_response: str) -> str:
        """Extract executable Python code from the model's response."""
        pass

    def format_repl_output(self, output: str, error: Optional[str] = None) -> str:
        """Format REPL output and error for the model."""
        res = []
        if output:
            res.append(output)
        if error:
            res.append(f"Error:\n{error}")
        text = "\n".join(res) if res else "(No output)"
        return self.truncate_output(text)

    def truncate_output(self, text: str) -> str:
        """Truncate overly long output to protect context window."""
        if len(text) <= self.max_output_chars:
            return text
        head = self.max_output_chars // 2
        tail = self.max_output_chars // 2
        omitted = len(text) - (head + tail)
        return f"{text[:head]}\n\n... [truncated {omitted} characters] ...\n\n{text[-tail:]}"


def _clean_and_extract_code(raw_response: str, require_fence: bool = False) -> str:
    """Clean model output, stripping thinking tags, tool XML wrappers, and extracting code blocks."""
    text = raw_response.strip()

    # 1. Discard thinking tokens if present
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Extract markdown code fences if present
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        text = "\n\n".join(b.strip() for b in blocks)
    elif require_fence:
        # Check for unclosed code fence
        unclosed = re.search(r"^```(?:python)?\s*\n(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
        if unclosed:
            text = unclosed.group(1).strip()
    else:
        # Check for unclosed code fence
        unclosed = re.search(r"^```(?:python)?\s*\n(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
        if unclosed:
            text = unclosed.group(1).strip()

    # 3. Strip tool call and function XML wrappers (e.g. <tool_call>, <function=...>, </function>, etc.)
    text = re.sub(r"</?(?:tool_call|function|parameter|action)[^>]*>", "", text).strip()

    # 4. Strip stray tool call assignment lines like `function=print`, `function=code`, `tool_call=...`
    filtered_lines = []
    for line in text.splitlines():
        if re.match(r"^\s*(?:function|tool_call|call)\s*=\s*\w+\s*$", line):
            continue
        filtered_lines.append(line)
    text = "\n".join(filtered_lines).strip()
    return text


class RawCodeProtocol(BaseProtocol):
    """JAZ's default protocol.
    
    Treats the model's response directly as executable Python code.
    Gracefully strips markdown fences, thinking tags, and tool call XML wrappers.
    """

    def extract_code(self, raw_response: str) -> str:
        return _clean_and_extract_code(raw_response, require_fence=False)


class CodeOnlyProtocol(BaseProtocol):
    """Protocol that extracts Python markdown code fences or cleans tool calls.
    
    Useful for chat models that include explanations around code blocks.
    """

    def extract_code(self, raw_response: str) -> str:
        return _clean_and_extract_code(raw_response, require_fence=True)
