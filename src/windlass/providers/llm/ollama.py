"""Ollama adapter for locally hosted models.

Ollama needs no extra install: it speaks HTTP and Windlass already depends on
``httpx``. Start the daemon, pull a model, and point Windlass at it::

    ollama pull llama3.2
    python -c "from windlass import Windlass; print(Windlass.llm('ollama').complete('hi'))"

That makes Ollama the zero-cost, zero-key way to run the whole framework against
a real model — useful for local development and for CI that wants more than the
``fake`` provider.

Example:
    >>> from windlass import Windlass                                     # doctest: +SKIP
    >>> Windlass.llm("ollama", model="llama3.2").complete("hi").content  # doctest: +SKIP
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from windlass.core.config import settings
from windlass.core.exceptions import ProviderError, ProviderTimeoutError, ResponseError
from windlass.core.registry import register
from windlass.core.types import Completion, FinishReason, Message, StreamEvent, ToolCall, Usage
from windlass.interfaces.llm import LLM

__all__ = ["OllamaLLM"]


@register.llm(
    "ollama",
    aliases=("local",),
    description="Locally hosted models via the Ollama daemon (no extra dependency).",
)
class OllamaLLM(LLM):
    """Chat completions against a local Ollama daemon.

    Args:
        model: Model tag, e.g. ``llama3.2``, ``qwen2.5``, ``mistral``.
        base_url: Daemon address. Defaults to ``OLLAMA_BASE_URL`` or
            ``http://localhost:11434``.
        keep_alive: How long the daemon keeps the model resident, e.g. ``"5m"``.
        options: Raw Ollama options (``num_ctx``, ``top_k``, ``repeat_penalty``,
            ...), merged into the request.
        **config: Forwarded to :class:`~windlass.interfaces.llm.LLM`.

    Raises:
        ProviderError: When the daemon is unreachable. The message explains how
            to start it.

    Performance:
        The first call after a cold start pays model-load time — often several
        seconds. Set ``keep_alive`` to keep the model warm between requests.
    """

    provider_name = "ollama"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        model: str = "",
        *,
        base_url: str | None = None,
        keep_alive: str | None = None,
        options: dict[str, Any] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(model=model, **config)
        self.base_url = (base_url or settings().ollama_base_url).rstrip("/")
        self.keep_alive = keep_alive
        self.options = dict(options or {})
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    @classmethod
    def default_model(cls) -> str:
        """Return ``"llama3.2"``."""
        return "llama3.2"

    def native(self) -> Any:
        """Return the underlying ``httpx.AsyncClient``."""
        return self._client

    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Request one chat completion.

        Args:
            messages: The conversation.
            tools: OpenAI-format tool definitions. Support depends on the model.
            **kwargs: Request overrides.

        Returns:
            The completion.

        Raises:
            ProviderError: When the daemon is unreachable or returns an error.
            ProviderTimeoutError: When the request exceeds the timeout.
        """
        payload = self._payload(messages, tools, kwargs, stream=False)
        data = await self._post("/api/chat", payload)

        message = data.get("message") or {}
        calls = [
            ToolCall(
                name=(call.get("function") or {}).get("name", ""),
                arguments=_as_dict((call.get("function") or {}).get("arguments")),
            )
            for call in message.get("tool_calls") or []
        ]
        return Completion(
            content=message.get("content", ""),
            tool_calls=calls,
            finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
            model=data.get("model", self.model),
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                completion_tokens=data.get("eval_count", 0) or 0,
            ),
            raw=data,
        )

    async def astream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion.

        Ollama streams newline-delimited JSON objects rather than SSE.

        Args:
            messages: The conversation.
            tools: OpenAI-format tool definitions.
            **kwargs: Request overrides.

        Yields:
            Text deltas, then tool calls, then ``done``.
        """
        payload = self._payload(messages, tools, kwargs, stream=True)
        calls: list[ToolCall] = []
        usage = Usage()

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = chunk.get("message") or {}
                    if message.get("content"):
                        yield StreamEvent(type="text", delta=message["content"], raw=chunk)
                    for call in message.get("tool_calls") or []:
                        fn = call.get("function") or {}
                        calls.append(
                            ToolCall(
                                name=fn.get("name", ""), arguments=_as_dict(fn.get("arguments"))
                            )
                        )
                    if chunk.get("done"):
                        usage = Usage(
                            prompt_tokens=chunk.get("prompt_eval_count", 0) or 0,
                            completion_tokens=chunk.get("eval_count", 0) or 0,
                        )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

        for call in calls:
            yield StreamEvent(type="tool_call", tool_call=call)
        yield StreamEvent(
            type="done",
            finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
            usage=usage,
        )

    async def alist_models(self) -> list[str]:
        """Return the model tags the daemon has pulled.

        Returns:
            Model tags, e.g. ``["llama3.2:latest", "nomic-embed-text:latest"]``.

        Raises:
            ProviderError: When the daemon is unreachable.
        """
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        return [m["name"] for m in response.json().get("models", [])]

    # -- helpers ----------------------------------------------------------
    def _payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        overrides: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the Ollama request body."""
        merged = self._merged(**overrides)
        options = {**self.options}
        if "temperature" in merged:
            options["temperature"] = merged["temperature"]
        if merged.get("max_tokens"):
            options["num_predict"] = merged["max_tokens"]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_ollama_messages(messages),
            "stream": stream,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        return payload

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON and return the decoded body."""
        try:
            response = await self._client.post(path, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        try:
            return response.json()
        except ValueError as exc:
            raise ResponseError("Ollama returned a non-JSON response.", provider="ollama") from exc

    def _transport_error(self, exc: httpx.HTTPError) -> ProviderError:
        """Turn an httpx failure into an actionable Windlass error."""
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(
                f"Ollama did not respond within {self.timeout}s.",
                provider="ollama",
                hint="Large models are slow to load; raise the timeout or set keep_alive.",
                original=exc,
            )
        if isinstance(exc, httpx.ConnectError):
            return ProviderError(
                f"Cannot reach the Ollama daemon at {self.base_url}.",
                provider="ollama",
                hint="Start it with `ollama serve`, then `ollama pull " f"{self.model}`.",
                original=exc,
            )
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = ""
        response = getattr(exc, "response", None)
        if response is not None:
            detail = response.text[:300]
        if status == 404:
            return ProviderError(
                f"Ollama has no model named {self.model!r}.",
                provider="ollama",
                hint=f"Run: ollama pull {self.model}",
                original=exc,
            )
        return ProviderError(
            f"Ollama request failed: {exc}{' — ' + detail if detail else ''}",
            provider="ollama",
            original=exc,
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


def _to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate Windlass messages into Ollama's chat format."""
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.role.value
        entry: dict[str, Any] = {"role": role, "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls
            ]
        out.append(entry)
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce tool-call arguments, which Ollama may send as a dict or a string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}
