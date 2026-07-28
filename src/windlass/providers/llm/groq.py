"""Groq adapter.

Groq serves open-weight models (Llama, Mixtral, Gemma) on custom silicon at
very high token rates. Its API is OpenAI-compatible, so this adapter reuses the
message translation from :mod:`windlass.providers.llm.openai` and only differs in
client construction and error mapping.

Install with::

    pip install "windlass[groq]"

Example:
    >>> from windlass import Windlass                                    # doctest: +SKIP
    >>> Windlass.llm("groq", model="llama-3.3-70b-versatile")           # doctest: +SKIP
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from windlass.core.config import settings
from windlass.core.exceptions import (
    AuthenticationError,
    MalformedToolCallError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    ResponseError,
)
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.core.types import Completion, Message, StreamEvent, ToolCall, Usage
from windlass.interfaces.llm import LLM
from windlass.providers.llm.openai import to_openai_messages

__all__ = ["GroqLLM"]


@register.llm("groq", description="Groq-hosted open models at very high throughput.")
class GroqLLM(LLM):
    """Chat completions via the official ``groq`` SDK.

    Args:
        model: Model id, e.g. ``llama-3.3-70b-versatile``.
        api_key: Credential. Falls back to ``GROQ_API_KEY``.
        base_url: Endpoint override.
        max_retries: SDK-level retries; 0 because Windlass retries itself.
        **config: Forwarded to :class:`~windlass.interfaces.llm.LLM`.

    Raises:
        MissingDependencyError: When ``groq`` is not installed.
        AuthenticationError: When no API key can be found.
    """

    provider_name = "groq"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        model: str = "",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 0,
        **config: Any,
    ) -> None:
        super().__init__(model=model, **config)
        self._sdk = require("groq", extra="groq", feature="The Groq provider")
        key = api_key or settings().require_secret(
            "groq_api_key", provider="groq", env_var="GROQ_API_KEY"
        )
        kwargs: dict[str, Any] = {
            "api_key": key,
            "timeout": self.timeout,
            "max_retries": max_retries,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = self._sdk.AsyncGroq(**kwargs)

    @classmethod
    def default_model(cls) -> str:
        """Return ``"llama-3.3-70b-versatile"``."""
        return "llama-3.3-70b-versatile"

    def native(self) -> Any:
        """Return the underlying ``groq.AsyncGroq`` client."""
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
            tools: OpenAI-format tool definitions.
            **kwargs: Request overrides.

        Returns:
            The completion.

        Raises:
            ProviderError: For any API failure.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            **self._merged(**kwargs),
        }
        if tools:
            payload["tools"] = tools
        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise self._translate(exc) from exc

        try:
            choice = response.choices[0]
        except (AttributeError, IndexError) as exc:
            raise ResponseError("Groq returned no choices.", provider="groq") from exc

        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_loads(tc.function.arguments),
                raw_arguments=tc.function.arguments,
            )
            for tc in (choice.message.tool_calls or [])
        ]
        usage = Usage()
        if getattr(response, "usage", None):
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )
        return Completion(
            content=choice.message.content or "",
            tool_calls=calls,
            finish_reason=self._finish_reason(choice.finish_reason),
            model=getattr(response, "model", self.model),
            usage=usage,
            raw=response,
        )

    async def astream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion.

        Args:
            messages: The conversation.
            tools: OpenAI-format tool definitions.
            **kwargs: Request overrides.

        Yields:
            Text deltas, then completed tool calls, then ``done``.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "stream": True,
            **self._merged(**kwargs),
        }
        if tools:
            payload["tools"] = tools

        partials: dict[int, dict[str, Any]] = {}
        finish: str | None = None
        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if getattr(choice.delta, "content", None):
                    yield StreamEvent(type="text", delta=choice.delta.content, raw=chunk)
                for fragment in getattr(choice.delta, "tool_calls", None) or []:
                    slot = partials.setdefault(
                        fragment.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if fragment.id:
                        slot["id"] = fragment.id
                    if fragment.function and fragment.function.name:
                        slot["name"] = fragment.function.name
                    if fragment.function and fragment.function.arguments:
                        slot["arguments"] += fragment.function.arguments
                if choice.finish_reason:
                    finish = choice.finish_reason
        except Exception as exc:
            raise self._translate(exc) from exc

        for slot in partials.values():
            yield StreamEvent(type="tool_call", tool_call=_assemble(slot))
        yield StreamEvent(type="done", finish_reason=self._finish_reason(finish))

    def _translate(self, exc: BaseException) -> ProviderError:
        """Map a ``groq`` SDK exception onto the Windlass hierarchy."""
        name = type(exc).__name__
        message = str(exc)
        if name in {"AuthenticationError", "PermissionDeniedError"}:
            return AuthenticationError(
                f"Groq rejected the credentials: {message}",
                provider="groq",
                hint="Check GROQ_API_KEY.",
                original=exc,
            )
        if name == "RateLimitError":
            # Groq advertises how long to wait in a Retry-After header. Passing
            # it through lets the backoff honour the server's own guidance
            # instead of guessing — the difference between clearing a 290ms
            # per-minute limit on the next attempt and burning all three.
            return RateLimitError(
                f"Groq rate limit: {message}",
                provider="groq",
                retry_after=_retry_after(exc),
                status_code=429,
                original=exc,
            )
        if "Timeout" in name:
            return ProviderTimeoutError(
                f"Groq request timed out: {message}", provider="groq", original=exc
            )
        malformed = _malformed_tool_call(exc)
        if malformed is not None:
            return MalformedToolCallError(
                "Groq rejected the model's tool call as unparseable.",
                provider="groq",
                raw=malformed,
                original=exc,
            )
        return ProviderError(f"Groq call failed ({name}): {message}", provider="groq", original=exc)

    async def aclose(self) -> None:
        """Close the SDK's HTTP connection pool."""
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()


def _retry_after(exc: BaseException) -> float | None:
    """Return the Retry-After the provider advertised, in seconds.

    Args:
        exc: The vendor exception.

    Returns:
        The advertised wait, or ``None`` when the provider did not send one.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _malformed_tool_call(exc: BaseException) -> str | None:
    """Return the rejected generation when ``exc`` is a tool-call parse failure.

    Groq answers a tool call it cannot parse with HTTP 400 and
    ``code: "tool_use_failed"``, putting the model's raw text in
    ``failed_generation``. That is a model mistake rather than a provider
    failure, so it is worth distinguishing: the agent can show the model what it
    got wrong and let it retry, instead of the whole run dying.

    Matched on the error body rather than an exception class, because the
    ``groq`` SDK reports it as a plain ``BadRequestError`` shared with a dozen
    unrelated causes.

    Args:
        exc: The exception raised by the Groq SDK.

    Returns:
        The rejected generation, ``""`` when the code matched but no generation
        was reported, or ``None`` when this is a different error.

    Example:
        >>> class Boom(Exception):
        ...     body = {"error": {"code": "tool_use_failed",
        ...                       "failed_generation": "<function=a>{"}}
        >>> _malformed_tool_call(Boom())
        '<function=a>{'
        >>> _malformed_tool_call(ValueError("something else")) is None
        True
    """
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("code") == "tool_use_failed":
        return str(error.get("failed_generation") or "")

    # Some SDK versions only stringify the payload; fall back to the message.
    message = str(exc)
    if "tool_use_failed" in message:
        return _extract_failed_generation(message)
    return None


def _extract_failed_generation(message: str) -> str:
    """Pull ``failed_generation`` out of a stringified error payload.

    Args:
        message: The exception's string form.

    Returns:
        The generation, or ``""`` when it cannot be located.

    Example:
        >>> _extract_failed_generation("400 {'failed_generation': '<function=x>{}'}")
        '<function=x>{}'
    """
    marker = "'failed_generation':"
    start = message.find(marker)
    if start < 0:
        marker = '"failed_generation":'
        start = message.find(marker)
    if start < 0:
        return ""
    remainder = message[start + len(marker) :].lstrip()
    if not remainder or remainder[0] not in "'\"":
        return ""
    quote = remainder[0]
    end = remainder.find(quote, 1)
    return remainder[1:end] if end > 0 else remainder[1:]


def _loads(raw: str | None) -> dict[str, Any]:
    """Parse a tool-call argument string, tolerating malformed JSON."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _assemble(slot: dict[str, Any]) -> ToolCall:
    """Build a :class:`ToolCall` from accumulated stream fragments."""
    fields: dict[str, Any] = {
        "name": slot["name"],
        "arguments": _loads(slot["arguments"]),
        "raw_arguments": slot["arguments"],
    }
    if slot["id"]:
        fields["id"] = slot["id"]
    return ToolCall(**fields)
