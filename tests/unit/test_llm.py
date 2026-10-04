"""Unit tests for OpenAICompatibleLLM streaming, callbacks, and error handling."""

import pytest
import httpx
from jaz.llm import OpenAICompatibleLLM
from jaz.exceptions import ContextWindowExceededError, FatalError


def test_openai_compatible_llm_streaming(monkeypatch):
    sse_lines = [
        'data: {"choices": [{"delta": {"content": "print("}}]}',
        'data: {"choices": [{"delta": {"content": "\'streaming works!\')"}}]}',
        'data: {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}',
        'data: [DONE]',
    ]

    class FakeStreamResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_lines(self):
            for line in sse_lines:
                yield line

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self, method, url, json=None, headers=None):
            return FakeStreamResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)

    chunks_received = []

    def on_chunk(c, total):
        chunks_received.append((c, total))

    llm = OpenAICompatibleLLM(base_url="http://fake:8000/v1", stream=True)
    resp = llm.query([{"role": "user", "content": "hi"}], on_chunk=on_chunk)

    assert resp.content == "print('streaming works!')"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5
    assert len(chunks_received) == 2
    assert chunks_received[0] == ("print(", 1)
    assert chunks_received[1] == ("'streaming works!')", 2)


def test_openai_compatible_llm_400_context_exceeded(monkeypatch):
    class FakeStreamResponse400:
        status_code = 400
        text = "This model maximum context length is 131072 tokens, but request has 135000 tokens"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self, method, url, json=None, headers=None):
            return FakeStreamResponse400()

    monkeypatch.setattr(httpx, "Client", FakeClient)

    llm = OpenAICompatibleLLM(base_url="http://fake:8000/v1", stream=True, max_retries=1)
    with pytest.raises(ContextWindowExceededError):
        llm.query([{"role": "user", "content": "hi"}])


def test_openai_compatible_llm_400_fatal_error(monkeypatch):
    class FakeStreamResponse400:
        status_code = 400
        text = "Invalid JSON schema in parameter"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self, method, url, json=None, headers=None):
            return FakeStreamResponse400()

    monkeypatch.setattr(httpx, "Client", FakeClient)

    llm = OpenAICompatibleLLM(base_url="http://fake:8000/v1", stream=True, max_retries=1)
    with pytest.raises(FatalError):
        llm.query([{"role": "user", "content": "hi"}])
