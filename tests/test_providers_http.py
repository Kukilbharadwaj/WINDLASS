"""Tests for the HTTP-backed providers, against a mock transport.

The Ollama adapter needs no optional dependency — it speaks HTTP through the
core ``httpx`` client — so there is no excuse for it to be untested. These tests
drive it through ``httpx.MockTransport``, exercising the real request building,
response parsing, streaming and error translation without a daemon.

The same approach works for any provider whose SDK accepts an httpx transport.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

from windlass.core.exceptions import ProviderError, ProviderTimeoutError
from windlass.core.types import Message, ToolCall, ToolResult
from windlass.providers.llm.ollama import OllamaLLM


def _ollama(handler) -> OllamaLLM:
    """Build an Ollama client whose transport is ``handler``."""
    llm = OllamaLLM(model="llama3.2")
    llm._client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    return llm


class TestOllamaCompletion:
    async def test_request_shape_and_response_parsing(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "Paris."},
                    "done": True,
                    "prompt_eval_count": 11,
                    "eval_count": 3,
                },
            )

        completion = await _ollama(handler).acomplete("Capital of France?")

        assert captured["url"].endswith("/api/chat")
        assert captured["body"]["model"] == "llama3.2"
        assert captured["body"]["stream"] is False
        assert captured["body"]["messages"] == [{"role": "user", "content": "Capital of France?"}]
        assert completion.content == "Paris."
        assert completion.usage.prompt_tokens == 11
        assert completion.usage.completion_tokens == 3
        assert completion.usage.total_tokens == 14

    async def test_generation_options_are_mapped_to_ollama_names(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

        llm = _ollama(handler)
        llm.temperature = 0.1
        llm.max_tokens = 64
        await llm.acomplete("hi")

        assert captured["options"]["temperature"] == 0.1
        assert captured["options"]["num_predict"] == 64  # not "max_tokens"

    async def test_tool_calls_are_parsed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                        ],
                    },
                    "done": True,
                },
            )

        completion = await _ollama(handler).acomplete("weather?")
        assert completion.tool_calls[0].name == "get_weather"
        assert completion.tool_calls[0].arguments == {"city": "Paris"}
        assert completion.finish_reason.value == "tool_calls"

    async def test_string_encoded_tool_arguments_are_decoded(self):
        """Ollama sometimes serialises arguments as a JSON string."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}],
                    },
                    "done": True,
                },
            )

        completion = await _ollama(handler).acomplete("go")
        assert completion.tool_calls[0].arguments == {"a": 1}

    async def test_tool_results_round_trip_into_the_request(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "done"}, "done": True})

        await _ollama(handler).acomplete(
            [
                Message.user("weather?"),
                Message.assistant("", tool_calls=[ToolCall(id="c1", name="f", arguments={"x": 1})]),
                Message.tool(ToolResult(call_id="c1", name="f", content="sunny")),
            ]
        )

        roles = [m["role"] for m in captured["messages"]]
        assert roles == ["user", "assistant", "tool"]
        assert captured["messages"][1]["tool_calls"][0]["function"]["name"] == "f"


class TestOllamaStreaming:
    async def test_ndjson_stream_is_reassembled(self):
        """Ollama streams newline-delimited JSON, not SSE."""

        def handler(request: httpx.Request) -> httpx.Response:
            chunks = [
                {"message": {"content": "Par"}, "done": False},
                {"message": {"content": "is."}, "done": False},
                {"message": {"content": ""}, "done": True, "prompt_eval_count": 5, "eval_count": 2},
            ]
            body = "\n".join(json.dumps(c) for c in chunks)
            return httpx.Response(200, content=body.encode())

        events = [e async for e in _ollama(handler).astream("Capital?")]

        assert "".join(e.delta for e in events if e.type == "text") == "Paris."
        assert events[-1].type == "done"
        assert events[-1].usage.total_tokens == 7

    async def test_malformed_lines_are_skipped_not_fatal(self):
        def handler(request: httpx.Request) -> httpx.Response:
            lines = [
                '{"message": {"content": "a"}}',
                "not json",
                '{"message": {"content": "b"}, "done": true}',
            ]
            return httpx.Response(200, content="\n".join(lines).encode())

        events = [e async for e in _ollama(handler).astream("go")]
        assert "".join(e.delta for e in events if e.type == "text") == "ab"


class TestOllamaErrors:
    async def test_connection_refused_explains_how_to_start_the_daemon(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(ProviderError, match="ollama serve"):
            await _ollama(handler).acomplete("hi", retry=False)

    async def test_unknown_model_suggests_pulling_it(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model not found"})

        with pytest.raises(ProviderError, match=re.escape("ollama pull llama3.2")):
            await _ollama(handler).acomplete("hi", retry=False)

    async def test_timeout_is_translated(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        with pytest.raises(ProviderTimeoutError, match="did not respond"):
            await _ollama(handler).acomplete("hi", retry=False)

    async def test_server_error_surfaces_the_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="internal boom")

        with pytest.raises(ProviderError, match="internal boom"):
            await _ollama(handler).acomplete("hi", retry=False)


class TestOllamaMisc:
    async def test_list_models(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/tags"
            return httpx.Response(
                200, json={"models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:7b"}]}
            )

        assert await _ollama(handler).alist_models() == ["llama3.2:latest", "qwen2.5:7b"]

    def test_defaults(self):
        assert OllamaLLM.default_model() == "llama3.2"
        assert OllamaLLM(model="x").base_url.startswith("http")

    async def test_keep_alive_is_sent_when_configured(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

        llm = OllamaLLM(model="llama3.2", keep_alive="10m")
        llm._client = httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        )
        await llm.acomplete("hi")
        assert captured["keep_alive"] == "10m"

    async def test_the_native_client_is_exposed(self):
        llm = _ollama(lambda r: httpx.Response(200, json={"message": {"content": ""}}))
        assert isinstance(llm.native(), httpx.AsyncClient)
        await llm.aclose()
