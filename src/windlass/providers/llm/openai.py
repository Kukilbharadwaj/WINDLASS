"""OpenAI chat-completions adapter.

Covers OpenAI proper and every OpenAI-compatible endpoint — Azure OpenAI,
vLLM, LM Studio, OpenRouter, Together — by pointing ``base_url`` at them.

Install with::

    pip install "windlass[openai]"

Example:
    >>> from windlass import Windlass                      # doctest: +SKIP
    >>> llm = Windlass.llm("openai", model="gpt-4o-mini")  # doctest: +SKIP
    >>> llm.complete("Say hi").content                    # doctest: +SKIP
    'Hi!'
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from windlass.core.config import settings
from windlass.core.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    ResponseError,
)
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.core.types import Completion, Message, Role, StreamEvent, ToolCall, Usage
from windlass.interfaces.llm import LLM

__all__ = ["OpenAILLM", "to_openai_messages", "translate_openai_error"]


@register.llm(
    "openai",
    aliases=("gpt", "azure-openai"),
    description="OpenAI and OpenAI-compatible chat models.",
)
class OpenAILLM(LLM):
    """Chat completions via the official ``openai`` SDK.

    Args:
        model: Model id, e.g. ``gpt-4o``, ``gpt-4o-mini``, ``o3-mini``.
        api_key: Credential. Falls back to ``OPENAI_API_KEY``.
        base_url: Endpoint override for compatible gateways.
        organization: OpenAI organization id.
        max_retries: SDK-level retries. Windlass applies its own policy on top,
            so this defaults to 0 to avoid multiplying the two.
        default_headers: Extra headers sent with every request.
        **config: Forwarded to :class:`~windlass.interfaces.llm.LLM` and, for
            unknown keys, to the SDK request.

    Raises:
        MissingDependencyError: When ``openai`` is not installed.
        AuthenticationError: When no API key can be found.

    Performance:
        One shared ``AsyncOpenAI`` client per instance keeps the HTTP connection
        pool warm; construct the LLM once and reuse it.
    """

    provider_name = "openai"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        model: str = "",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        max_retries: int = 0,
        default_headers: dict[str, str] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(model=model, **config)
        self._sdk = require("openai", extra="openai", feature="The OpenAI provider")
        cfg = settings()
        self._api_key = api_key or cfg.secret("openai_api_key")
        self._base_url = base_url or cfg.openai_base_url
        if not self._api_key and not self._base_url:
            raise AuthenticationError(
                "No API key configured for the OpenAI provider.",
                provider="openai",
                hint=(
                    "Set OPENAI_API_KEY in your environment or a .env file, or pass\n"
                    "    Windlass.llm('openai', api_key='sk-...')\n"
                    "Local gateways only need base_url."
                ),
            )
        self._client = self._sdk.AsyncOpenAI(
            api_key=self._api_key or "not-needed",
            base_url=self._base_url,
            organization=organization,
            timeout=self.timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )

    @classmethod
    def default_model(cls) -> str:
        """Return ``"gpt-4o-mini"`` — capable and inexpensive."""
        return "gpt-4o-mini"

    def native(self) -> Any:
        """Return the underlying ``openai.AsyncOpenAI`` client (Level 3 access).

        Example:
            >>> client = llm.native()                       # doctest: +SKIP
            >>> await client.images.generate(prompt="a cat")  # doctest: +SKIP
        """
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
            **kwargs: Request overrides (``temperature``, ``response_format``,
                ``seed``, ``tool_choice``, ...).

        Returns:
            The completion, with the SDK response on ``raw``.

        Raises:
            AuthenticationError: Invalid credentials.
            RateLimitError: Quota or rate limit hit.
            ProviderTimeoutError: The request timed out.
            ProviderError: Any other API failure.
        """
        payload = self._payload(messages, tools, kwargs)
        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise translate_openai_error(exc, self._sdk) from exc
        return self._to_completion(response)

    async def astream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion.

        Tool calls arrive fragmented across chunks; they are reassembled here and
        emitted as complete :class:`~windlass.core.types.ToolCall` objects.

        Args:
            messages: The conversation.
            tools: OpenAI-format tool definitions.
            **kwargs: Request overrides.

        Yields:
            Text deltas, then completed tool calls, then ``done``.
        """
        payload = self._payload(messages, tools, kwargs)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        partials: dict[int, dict[str, Any]] = {}
        usage: Usage | None = None
        finish: str | None = None

        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = Usage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        completion_tokens=chunk.usage.completion_tokens or 0,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if getattr(delta, "content", None):
                    yield StreamEvent(type="text", delta=delta.content, raw=chunk)
                for fragment in getattr(delta, "tool_calls", None) or []:
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
            raise translate_openai_error(exc, self._sdk) from exc

        for slot in partials.values():
            yield StreamEvent(type="tool_call", tool_call=_build_tool_call(slot))
        yield StreamEvent(type="done", finish_reason=self._finish_reason(finish), usage=usage)

    # -- translation ------------------------------------------------------
    def _payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the SDK request body."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            **self._merged(**overrides),
        }
        # Reasoning models reject temperature and use a different token cap.
        if self.model.startswith(("o1", "o3", "o4")):
            payload.pop("temperature", None)
            if "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
        if tools:
            payload["tools"] = tools
        extra = {
            k: v
            for k, v in self.config.items()
            if k
            not in {
                "model",
                "temperature",
                "max_tokens",
                "timeout",
                "system_prompt",
                "api_key",
                "base_url",
            }
            and v is not None
        }
        payload.update(extra)
        return payload

    def _to_completion(self, response: Any) -> Completion:
        """Translate an SDK response into a :class:`Completion`."""
        try:
            choice = response.choices[0]
        except (AttributeError, IndexError) as exc:
            raise ResponseError(
                "OpenAI returned no choices.",
                provider="openai",
                context={"raw": str(response)[:500]},
            ) from exc

        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_parse_arguments(tc.function.arguments),
                raw_arguments=tc.function.arguments,
            )
            for tc in (choice.message.tool_calls or [])
        ]
        usage = Usage()
        if getattr(response, "usage", None):
            details = getattr(response.usage, "prompt_tokens_details", None)
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                cached_tokens=getattr(details, "cached_tokens", 0) or 0,
            )
        return Completion(
            content=choice.message.content or "",
            tool_calls=calls,
            finish_reason=self._finish_reason(choice.finish_reason),
            model=response.model or self.model,
            usage=usage,
            raw=response,
        )

    async def aclose(self) -> None:
        """Close the SDK's HTTP connection pool."""
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate Windlass messages into the OpenAI wire format.

    Shared by every OpenAI-compatible adapter (Groq, Ollama, local gateways), so
    the mapping lives in exactly one place.

    Args:
        messages: The conversation.

    Returns:
        Message dicts in OpenAI's schema.

    Example:
        >>> from windlass.core.types import Message
        >>> to_openai_messages([Message.user("hi")])
        [{'role': 'user', 'content': 'hi'}]
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.role.value if isinstance(message.role, Role) else str(message.role)
        if role == "tool":
            out.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id or "",
                }
            )
            continue

        entry: dict[str, Any] = {"role": role, "content": message.content}
        if message.name and role != "tool":
            entry["name"] = message.name
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments
                        or json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
            # OpenAI rejects an empty string alongside tool_calls.
            if not message.content:
                entry["content"] = None
        out.append(entry)
    return out


def translate_openai_error(exc: BaseException, sdk: Any) -> ProviderError:
    """Map an ``openai`` SDK exception onto the Windlass hierarchy.

    Args:
        exc: The raised exception.
        sdk: The imported ``openai`` module, used for ``isinstance`` checks.

    Returns:
        The matching :class:`~windlass.core.exceptions.ProviderError` subclass.
    """
    name = type(exc).__name__
    message = str(exc)

    if isinstance(exc, getattr(sdk, "AuthenticationError", ())) or name == "AuthenticationError":
        return AuthenticationError(
            f"OpenAI rejected the credentials: {message}",
            provider="openai",
            hint="Check OPENAI_API_KEY, and that the key has access to this model.",
            original=exc,
        )
    if isinstance(exc, getattr(sdk, "RateLimitError", ())) or name == "RateLimitError":
        retry_after: float | None = None
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers:
            advertised = headers.get("retry-after")
            try:
                retry_after = float(advertised) if advertised else None
            except (TypeError, ValueError):
                retry_after = None
        return RateLimitError(
            f"OpenAI rate limit: {message}",
            provider="openai",
            retry_after=retry_after,
            original=exc,
        )
    if isinstance(exc, getattr(sdk, "APITimeoutError", ())) or "Timeout" in name:
        return ProviderTimeoutError(
            f"OpenAI request timed out: {message}",
            provider="openai",
            hint="Raise the timeout with Windlass.llm('openai', timeout=120).",
            original=exc,
        )
    if isinstance(exc, getattr(sdk, "BadRequestError", ())) or name == "BadRequestError":
        return ProviderError(
            f"OpenAI rejected the request: {message}",
            provider="openai",
            hint="Usually a bad model name, an oversized prompt, or an invalid tool schema.",
            original=exc,
        )
    return ProviderError(f"OpenAI call failed ({name}): {message}", provider="openai", original=exc)


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Parse a tool-call argument string, tolerating malformed JSON."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _build_tool_call(slot: dict[str, Any]) -> ToolCall:
    """Assemble a :class:`ToolCall` from accumulated stream fragments.

    A streamed call arrives in pieces: the id and name in the first fragment,
    the arguments JSON spread across many. An id is generated when the provider
    omitted one, so the tool result can still be correlated.
    """
    fields: dict[str, Any] = {
        "name": slot["name"],
        "arguments": _parse_arguments(slot["arguments"]),
        "raw_arguments": slot["arguments"],
    }
    if slot["id"]:
        fields["id"] = slot["id"]
    return ToolCall(**fields)
