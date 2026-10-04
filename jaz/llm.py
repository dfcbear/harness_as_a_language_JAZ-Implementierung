"""LLM client abstractions for JAZ.

Provides a fully configurable OpenAI-compatible client for self-hosted or cloud
endpoints, as well as a ScriptedLLM for deterministic unit testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import time
from typing import Any, Callable, Dict, List, Optional, Union
import httpx

from .exceptions import ContextWindowExceededError, FatalError


@dataclass
class LLMResponse:
    """Standardized response from an LLM query."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    duration: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Abstract base class for language model backends."""

    def __init__(self, model: str, context_window: int = 32768):
        self.model = model
        self.context_window = context_window

    @abstractmethod
    def query(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Send chat messages to the model and return LLMResponse."""
        pass


class OpenAICompatibleLLM(BaseLLM):
    """Client for any OpenAI-compatible API endpoint (self-hosted vLLM, Ollama, LM Studio, OpenAI)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        api_key: str = "EMPTY",
        context_window: int = 32768,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        timeout: float = 300.0,
        stream: bool = True,
        max_retries: int = 3,
        price_per_1k_input: float = 0.0,
        price_per_1k_output: float = 0.0,
    ):
        super().__init__(model=model, context_window=context_window)
        # Normalize base_url to ensure it doesn't end with a trailing slash
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.stream = stream
        self.max_retries = max_retries
        self.price_per_1k_input = price_per_1k_input
        self.price_per_1k_output = price_per_1k_output

    def query(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        on_chunk: Optional[Callable[[str, int], None]] = kwargs.get("on_chunk")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        target_max_tokens = kwargs.get("max_tokens", self.max_tokens)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if target_max_tokens is not None:
            payload["max_tokens"] = target_max_tokens

        start_time = time.time()
        last_error = None

        stream_timeout = httpx.Timeout(connect=60.0, read=300.0, write=60.0, pool=60.0)
        fallback_timeout = httpx.Timeout(connect=60.0, read=max(600.0, self.timeout * 3), write=60.0, pool=60.0)

        for attempt in range(self.max_retries):
            # Attempt 1: Streaming SSE (resets read timeout on every received chunk)
            if self.stream:
                try:
                    stream_payload = dict(payload)
                    stream_payload["stream"] = True
                    stream_payload["stream_options"] = {"include_usage": True}

                    content_chunks: List[str] = []
                    reasoning_chunks: List[str] = []
                    input_toks = 0
                    output_toks = 0

                    with httpx.Client(timeout=stream_timeout) as client:
                        with client.stream("POST", url, json=stream_payload, headers=headers) as resp:
                            if resp.status_code == 200:
                                for raw_line in resp.iter_lines():
                                    line = raw_line.strip()
                                    if not line or not line.startswith("data:"):
                                        continue
                                    line_data = line[5:].strip()
                                    if line_data == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(line_data)
                                    except Exception:
                                        continue

                                    # Check usage in chunk
                                    usage = chunk.get("usage")
                                    if usage:
                                        input_toks = usage.get("prompt_tokens", input_toks)
                                        output_toks = usage.get("completion_tokens", output_toks)

                                    choices = chunk.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        c = delta.get("content")
                                        if c:
                                            content_chunks.append(c)
                                            if on_chunk:
                                                try:
                                                    on_chunk(c, len(content_chunks))
                                                except Exception:
                                                    pass
                                        r = delta.get("reasoning_content")
                                        if r:
                                            reasoning_chunks.append(r)
                                            if on_chunk and not c:
                                                try:
                                                    on_chunk(r, len(reasoning_chunks))
                                                except Exception:
                                                    pass

                                    # Safety break: stop if output exceeds max_tokens
                                    if target_max_tokens and output_toks >= target_max_tokens:
                                        break
                                    if target_max_tokens and len(content_chunks) > target_max_tokens * 2:
                                        break

                                    # Degenerate repetition detection (catch infinite loops early)
                                    if len(content_chunks) >= 30 and len(content_chunks) % 10 == 0:
                                        tail = "".join(content_chunks[-15:])
                                        prev_tail = "".join(content_chunks[-30:-15])
                                        if len(tail) > 30 and tail == prev_tail:
                                            # Repeating phrase loop detected! Break to prevent token burning
                                            break

                                full_content = "".join(content_chunks)
                                if not full_content and reasoning_chunks:
                                    full_content = "".join(reasoning_chunks)

                                if input_toks == 0:
                                    total_prompt_chars = sum(len(m.get("content", "")) for m in messages)
                                    input_toks = max(1, total_prompt_chars // 4)
                                if output_toks == 0:
                                    output_toks = max(1, len(full_content) // 4)

                                cost = (
                                    (input_toks / 1000.0) * self.price_per_1k_input
                                    + (output_toks / 1000.0) * self.price_per_1k_output
                                )

                                return LLMResponse(
                                    content=full_content,
                                    input_tokens=input_toks,
                                    output_tokens=output_toks,
                                    cost=cost,
                                    duration=time.time() - start_time,
                                    raw={"usage": {"prompt_tokens": input_toks, "completion_tokens": output_toks}},
                                )
                            else:
                                if resp.status_code == 400:
                                    err_text = resp.text
                                    if any(w in err_text for w in ("ContextWindowExceededError", "context length", "maximum context", "input_tokens")):
                                        raise ContextWindowExceededError(
                                            f"LLM context window limit exceeded ({self.context_window} tokens): {err_text}"
                                        )
                                    raise FatalError(f"LLM API returned client error 400 (non-retryable): {err_text}")
                                raise RuntimeError(f"LLM streaming API returned status {resp.status_code}: {resp.text}")
                except FatalError:
                    raise
                except Exception as e:
                    last_error = e

            # Fallback: Non-streaming POST
            try:
                with httpx.Client(timeout=fallback_timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    if resp.status_code != 200:
                        if resp.status_code == 400:
                            err_text = resp.text
                            if any(w in err_text for w in ("ContextWindowExceededError", "context length", "maximum context", "input_tokens")):
                                raise ContextWindowExceededError(
                                    f"LLM context window limit exceeded ({self.context_window} tokens): {err_text}"
                                )
                            raise FatalError(f"LLM API returned client error 400 (non-retryable): {err_text}")
                        raise RuntimeError(
                            f"LLM API returned status {resp.status_code}: {resp.text}"
                        )
                    data = resp.json()
                    choice = data["choices"][0]["message"]["content"] or ""

                    usage = data.get("usage", {})
                    input_toks = usage.get("prompt_tokens", 0)
                    output_toks = usage.get("completion_tokens", 0)

                    if input_toks == 0:
                        total_prompt_chars = sum(len(m.get("content", "")) for m in messages)
                        input_toks = max(1, total_prompt_chars // 4)
                    if output_toks == 0:
                        output_toks = max(1, len(choice) // 4)

                    cost = (
                        (input_toks / 1000.0) * self.price_per_1k_input
                        + (output_toks / 1000.0) * self.price_per_1k_output
                    )

                    return LLMResponse(
                        content=choice,
                        input_tokens=input_toks,
                        output_tokens=output_toks,
                        cost=cost,
                        duration=time.time() - start_time,
                        raw=data,
                    )
            except FatalError:
                raise
            except Exception as e:
                last_error = e
                time.sleep(1.0 * (attempt + 1))

        raise FatalError(f"Failed to query LLM after {self.max_retries} attempts: {last_error}")



class ScriptedLLM(BaseLLM):
    """Scripted or mock LLM for testing without real network or API keys."""

    def __init__(
        self,
        responses: Optional[List[Union[str, Callable[[List[Dict[str, str]]], str]]]] = None,
        model: str = "scripted-test",
        context_window: int = 32768,
    ):
        super().__init__(model=model, context_window=context_window)
        import threading
        self.responses = list(responses or [])
        self.queries_received: List[List[Dict[str, str]]] = []
        self.cursor = 0
        self._lock = threading.Lock()

    def query(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        with self._lock:
            self.queries_received.append(messages)
            if self.cursor < len(self.responses):
                item = self.responses[self.cursor]
                self.cursor += 1
            else:
                item = "return 'SCRIPTED_DEFAULT_DONE'"

        if callable(item):
            content = item(messages)
        else:
            content = str(item)

        in_toks = sum(len(m.get("content", "")) for m in messages) // 4
        out_toks = len(content) // 4
        return LLMResponse(
            content=content,
            input_tokens=max(1, in_toks),
            output_tokens=max(1, out_toks),
            cost=0.0,
            duration=0.01,
        )
