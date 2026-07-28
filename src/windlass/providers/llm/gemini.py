"""Google Gemini adapter.

Gemini's API names things differently from everyone else: turns are ``contents``
with ``parts``, the assistant role is ``model``, tools are ``function_declarations``
grouped under a ``Tool``, and generation options live in a nested
``generation_config``. This adapter absorbs all of that.

Install with::

    pip install "windlass[gemini]"

Example:
    >>> from windlass import Windlass                                # doctest: +SKIP
    >>> Windlass.llm("gemini", model="gemini-2.0-flash")            # doctest: +SKIP
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

__all__ = ["GeminiLLM", "to_gemini_contents"]


@register.llm(
    "gemini",
    aliases=("google", "google-genai"),
    description="Google Gemini models via the google-genai SDK.",
)
class GeminiLLM(LLM):
    """Chat completions via the ``google-genai`` SDK.

    Args:
        model: Model id, e.g. ``gemini-2.0-flash``.
        api_key: Credential. Falls back to ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``.
        vertexai: Route through Vertex AI instead of the Gemini Developer API.
        project: GCP project id (Vertex only).
        location: GCP region (Vertex only).
        **config: Forwarded to :class:`~windlass.interfaces.llm.LLM`.

    Raises:
        MissingDependencyError: When ``google-genai`` is not installed.
        AuthenticationError: When no API key can be found and Vertex is off.
    """

    provider_name = "gemini"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        model: str = "",
        *,
        api_key: str | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        **config: Any,
    ) -> None:
        super().__init__(model=model, **config)
        genai = require("google.genai", extra="gemini", feature="The Gemini provider")
        self._genai = genai
        self._types = require("google.genai.types", extra="gemini", feature="The Gemini provider")
        if vertexai:
            self._client = genai.Client(vertexai=True, project=project, location=location)
        else:
            key = api_key or settings().require_secret(
                "google_api_key", provider="gemini", env_var="GOOGLE_API_KEY"
            )
            self._client = genai.Client(api_key=key)

    @classmethod
    def default_model(cls) -> str:
        """Return ``"gemini-2.0-flash"``."""
        return "gemini-2.0-flash"

    def native(self) -> Any:
        """Return the underlying ``google.genai.Client``."""
        return self._client

    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Request one generation.

        Args:
            messages: The conversation.
            tools: Gemini-format function declarations.
            **kwargs: Request overrides.

        Returns:
            The completion.

        Raises:
            ProviderError: For any API failure.
        """
        system, contents = to_gemini_contents(messages)
        config = self._config(system, tools, kwargs)
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:
            raise self._translate(exc) from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for candidate in getattr(response, "candidates", None) or []:
            for part in getattr(candidate.content, "parts", None) or []:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fn = getattr(part, "function_call", None)
                if fn is not None:
                    calls.append(ToolCall(name=fn.name, arguments=dict(fn.args or {})))

        finish = None
        if getattr(response, "candidates", None):
            finish = getattr(response.candidates[0], "finish_reason", None)

        return Completion(
            content="".join(text_parts),
            tool_calls=calls,
            finish_reason=self._finish_reason(str(finish) if finish else None),
            model=self.model,
            usage=self._usage_from(response),
            raw=response,
        )

    async def astream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a generation.

        Args:
            messages: The conversation.
            tools: Gemini-format function declarations.
            **kwargs: Request overrides.

        Yields:
            Text deltas, then tool calls, then ``done``.
        """
        system, contents = to_gemini_contents(messages)
        config = self._config(system, tools, kwargs)
        calls: list[ToolCall] = []
        usage = Usage()

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
            async for chunk in stream:
                for candidate in getattr(chunk, "candidates", None) or []:
                    for part in getattr(candidate.content, "parts", None) or []:
                        if getattr(part, "text", None):
                            yield StreamEvent(type="text", delta=part.text, raw=chunk)
                        fn = getattr(part, "function_call", None)
                        if fn is not None:
                            calls.append(ToolCall(name=fn.name, arguments=dict(fn.args or {})))
                if getattr(chunk, "usage_metadata", None):
                    usage = self._usage_from(chunk)
        except Exception as exc:
            raise self._translate(exc) from exc

        for call in calls:
            yield StreamEvent(type="tool_call", tool_call=call)
        yield StreamEvent(type="done", usage=usage)

    # -- translation ------------------------------------------------------
    def _config(
        self,
        system: str,
        tools: list[dict[str, Any]] | None,
        overrides: dict[str, Any],
    ) -> Any:
        """Build a ``GenerateContentConfig`` from Windlass settings."""
        merged = self._merged(**overrides)
        fields: dict[str, Any] = {}
        if "temperature" in merged:
            fields["temperature"] = merged["temperature"]
        if merged.get("max_tokens"):
            fields["max_output_tokens"] = merged["max_tokens"]
        if system:
            fields["system_instruction"] = system
        if tools:
            fields["tools"] = [self._types.Tool(function_declarations=tools)]
        return self._types.GenerateContentConfig(**fields)

    @staticmethod
    def _usage_from(response: Any) -> Usage:
        """Extract token accounting from a Gemini response."""
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return Usage()
        return Usage(
            prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
        )

    def _translate(self, exc: BaseException) -> ProviderError:
        """Map a Gemini SDK exception onto the Windlass hierarchy."""
        name = type(exc).__name__
        message = str(exc)
        lowered = message.lower()
        if "api key" in lowered or "permission" in lowered or "unauthenticated" in lowered:
            return AuthenticationError(
                f"Gemini rejected the credentials: {message}",
                provider="gemini",
                hint="Check GOOGLE_API_KEY.",
                original=exc,
            )
        if "quota" in lowered or "rate" in lowered or "429" in message:
            return RateLimitError(f"Gemini rate limit: {message}", provider="gemini", original=exc)
        if "timeout" in lowered or "deadline" in lowered:
            return ProviderTimeoutError(
                f"Gemini request timed out: {message}", provider="gemini", original=exc
            )
        return ProviderError(
            f"Gemini call failed ({name}): {message}", provider="gemini", original=exc
        )


def to_gemini_contents(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Translate Windlass messages into Gemini's ``contents`` format.

    Args:
        messages: The conversation.

    Returns:
        A ``(system_instruction, contents)`` pair. The assistant role is
        renamed to ``model`` and tool results become ``function_response`` parts.

    Example:
        >>> from windlass.core.types import Message
        >>> system, contents = to_gemini_contents([Message.user("hi")])
        >>> contents[0]["role"]
        'user'
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for message in messages:
        role = message.role.value
        if role == "system":
            system_parts.append(message.content)
            continue
        if role == "tool":
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": message.name or "tool",
                                "response": {"result": message.content},
                            }
                        }
                    ],
                }
            )
            continue

        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"text": message.content})
        for call in message.tool_calls:
            parts.append({"function_call": {"name": call.name, "args": call.arguments}})
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})

    return "\n\n".join(p for p in system_parts if p), contents
