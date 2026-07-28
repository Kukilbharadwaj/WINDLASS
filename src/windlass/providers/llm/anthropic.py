"""Anthropic Claude adapter.

Claude's Messages API differs from OpenAI's in three ways that matter, and this
adapter absorbs all three so callers never have to think about them:

* the system prompt is a top-level parameter, not a message;
* content is a list of typed blocks (``text``, ``tool_use``, ``tool_result``)
  rather than a string;
* consecutive same-role messages must be merged.

Install with::

    pip install "windlass[anthropic]"

Example:
    >>> from windlass import Windlass                              # doctest: +SKIP
    >>> Windlass.llm("anthropic").complete("Say hi").content       # doctest: +SKIP
    'Hi!'
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from windlass.core.config import settings
from windlass.core.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.core.types import Completion, Message, StreamEvent, ToolCall, Usage
from windlass.interfaces.llm import LLM

__all__ = ["AnthropicLLM", "to_anthropic_messages"]

#: Claude requires an explicit output ceiling on every request.
DEFAULT_MAX_TOKENS = 4096


@register.llm(
    "anthropic",
    aliases=("claude",),
    description="Anthropic Claude models via the Messages API.",
)
class AnthropicLLM(LLM):
    """Chat completions via the official ``anthropic`` SDK.

    Args:
        model: Model id, e.g. ``claude-sonnet-4-5``.
        api_key: Credential. Falls back to ``ANTHROPIC_API_KEY``.
        base_url: Endpoint override.
        max_retries: SDK-level retries; 0 because Windlass retries itself.
        thinking_budget: Enables extended thinking with this token budget.
            ``None`` leaves it off.
        **config: Forwarded to :class:`~windlass.interfaces.llm.LLM`.

    Raises:
        MissingDependencyError: When ``anthropic`` is not installed.
        AuthenticationError: When no API key can be found.
    """

    provider_name = "anthropic"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        model: str = "",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 0,
        thinking_budget: int | None = None,
        **config: Any,
    ) -> None:
        super().__init__(model=model, **config)
        self._sdk = require("anthropic", extra="anthropic", feature="The Anthropic provider")
        key = api_key or settings().require_secret(
            "anthropic_api_key", provider="anthropic", env_var="ANTHROPIC_API_KEY"
        )
        self.thinking_budget = thinking_budget
        self._client = self._sdk.AsyncAnthropic(
            api_key=key,
            base_url=base_url,
            timeout=self.timeout,
            max_retries=max_retries,
        )

    @classmethod
    def default_model(cls) -> str:
        """Return ``"claude-sonnet-4-5"``."""
        return "claude-sonnet-4-5"

    def native(self) -> Any:
        """Return the underlying ``anthropic.AsyncAnthropic`` client."""
        return self._client

    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Request one message completion.

        Args:
            messages: The conversation. A leading system message is lifted into
                the top-level ``system`` parameter automatically.
            tools: Anthropic-format tool definitions (``input_schema``).
            **kwargs: Request overrides.

        Returns:
            The completion, with the SDK response on ``raw``.

        Raises:
            AuthenticationError: Invalid credentials.
            RateLimitError: Quota or rate limit hit.
            ProviderError: Any other API failure.
        """
        payload = self._payload(messages, tools, kwargs)
        try:
            response = await self._client.messages.create(**payload)
        except Exception as exc:
            raise self._translate(exc) from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input or {}),
                    )
                )

        return Completion(
            content="".join(text_parts),
            tool_calls=calls,
            finish_reason=self._finish_reason(response.stop_reason),
            model=response.model or self.model,
            usage=Usage(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
                cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            ),
            raw=response,
        )

    async def astream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a message completion.

        Args:
            messages: The conversation.
            tools: Anthropic-format tool definitions.
            **kwargs: Request overrides.

        Yields:
            Text deltas, then completed tool calls, then ``done``.
        """
        import json as _json

        payload = self._payload(messages, tools, kwargs)
        partials: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason: str | None = None

        try:
            async with self._client.messages.stream(**payload) as stream:
                async for event in stream:
                    kind = getattr(event, "type", "")
                    if kind == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            partials[event.index] = {
                                "id": block.id,
                                "name": block.name,
                                "arguments": "",
                            }
                    elif kind == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield StreamEvent(type="text", delta=delta.text, raw=event)
                        elif delta.type == "input_json_delta" and event.index in partials:
                            partials[event.index]["arguments"] += delta.partial_json
                    elif kind == "message_delta":
                        stop_reason = getattr(event.delta, "stop_reason", None) or stop_reason
                        if getattr(event, "usage", None):
                            usage = Usage(
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=event.usage.output_tokens or 0,
                            )
                    elif kind == "message_start":
                        usage = Usage(
                            prompt_tokens=event.message.usage.input_tokens or 0,
                            completion_tokens=0,
                        )
        except Exception as exc:
            raise self._translate(exc) from exc

        for slot in partials.values():
            try:
                arguments = _json.loads(slot["arguments"]) if slot["arguments"] else {}
            except ValueError:
                arguments = {}
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(
                    id=slot["id"],
                    name=slot["name"],
                    arguments=arguments,
                    raw_arguments=slot["arguments"],
                ),
            )
        yield StreamEvent(type="done", finish_reason=self._finish_reason(stop_reason), usage=usage)

    # -- translation ------------------------------------------------------
    def _payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the SDK request body."""
        system, converted = to_anthropic_messages(messages)
        merged = self._merged(**overrides)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "max_tokens": merged.pop("max_tokens", None) or DEFAULT_MAX_TOKENS,
        }
        if "temperature" in merged:
            payload["temperature"] = merged.pop("temperature")
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
        if self.thinking_budget:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            payload.pop("temperature", None)  # extended thinking requires the default
        payload.update(merged)
        return payload

    def _translate(self, exc: BaseException) -> ProviderError:
        """Map an ``anthropic`` SDK exception onto the Windlass hierarchy."""
        name = type(exc).__name__
        message = str(exc)
        if name in {"AuthenticationError", "PermissionDeniedError"}:
            return AuthenticationError(
                f"Anthropic rejected the credentials: {message}",
                provider="anthropic",
                hint="Check ANTHROPIC_API_KEY.",
                original=exc,
            )
        if name == "RateLimitError":
            return RateLimitError(
                f"Anthropic rate limit: {message}", provider="anthropic", original=exc
            )
        if "Timeout" in name:
            return ProviderTimeoutError(
                f"Anthropic request timed out: {message}", provider="anthropic", original=exc
            )
        return ProviderError(
            f"Anthropic call failed ({name}): {message}", provider="anthropic", original=exc
        )

    async def aclose(self) -> None:
        """Close the SDK's HTTP connection pool."""
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()


def to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Translate Windlass messages into Anthropic's format.

    Performs the three transformations Claude requires: system extraction,
    typed content blocks, and merging of consecutive same-role turns (the API
    rejects two ``user`` messages in a row, which happens naturally when several
    tools return at once).

    Args:
        messages: The conversation.

    Returns:
        A ``(system_prompt, messages)`` pair.

    Example:
        >>> from windlass.core.types import Message
        >>> system, msgs = to_anthropic_messages(
        ...     [Message.system("be nice"), Message.user("hi")]
        ... )
        >>> system, msgs[0]["role"]
        ('be nice', 'user')
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message.role.value
        if role == "system":
            system_parts.append(message.content)
            continue

        if role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content,
            }
            if message.metadata.get("is_error"):
                block["is_error"] = True
            _append(converted, "user", block)
            continue

        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        for call in message.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        if not blocks:
            continue
        for block in blocks:
            _append(converted, role, block)

    return "\n\n".join(p for p in system_parts if p), converted


def _append(messages: list[dict[str, Any]], role: str, block: dict[str, Any]) -> None:
    """Append a content block, merging into the previous same-role message."""
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].append(block)
    else:
        messages.append({"role": role, "content": [block]})
