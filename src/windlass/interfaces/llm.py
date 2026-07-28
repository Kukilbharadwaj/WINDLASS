"""The LLM interface.

Every model provider — OpenAI, Anthropic, Gemini, Groq, Ollama, or your own —
implements this one class. Agents, RAG pipelines, evaluators and guardrails all
talk to :class:`LLM`, never to a vendor SDK, which is what makes swapping
providers a one-word change.

Implementers override exactly two coroutines:

* :meth:`LLM.agenerate` — a single completion.
* :meth:`LLM.astream_generate` — incremental events (optional; the default
  emits the finished completion as one chunk).

Everything else — the blocking API, prompt coercion, retries, tool-schema
plumbing — is provided here.

Example:
    >>> from windlass.providers.llm.fake import FakeLLM
    >>> llm = FakeLLM(responses=["Paris"])
    >>> llm.complete("Capital of France?").content
    'Paris'
"""

from __future__ import annotations

import abc
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from windlass.core.concurrency import iter_sync, run_sync
from windlass.core.config import settings
from windlass.core.retry import retry_async
from windlass.core.types import (
    Completion,
    FinishReason,
    Message,
    StreamEvent,
    Usage,
    normalize_messages,
)
from windlass.interfaces.base import Component

__all__ = ["LLM", "PromptLike"]

#: Anything an LLM entry point accepts as a prompt.
PromptLike = str | Message | Sequence[Message] | Sequence[dict[str, Any]]


class LLM(Component):
    """Abstract chat-completion model.

    Args:
        model: Model identifier passed to the provider.
        temperature: Sampling temperature. Falls back to the global setting.
        max_tokens: Completion ceiling. ``None`` defers to the provider.
        timeout: Per-request timeout in seconds.
        system_prompt: Prepended to every call that does not already start with
            a system message.
        name: Component name for traces. Defaults to the provider name.
        **config: Provider-specific options forwarded verbatim to the SDK.

    Attributes:
        model: The configured model identifier.
        temperature: The configured sampling temperature.
        max_tokens: The configured completion ceiling.
        supports_tools: Whether this provider can do native tool calling.
        supports_streaming: Whether this provider can stream.

    Example:
        Implementing a provider takes one method::

            class MyLLM(LLM):
                provider_name = "mine"

                async def agenerate(self, messages, *, tools=None, **kw):
                    text = await my_sdk.chat(messages[-1].content)
                    return Completion(content=text, model=self.model)
    """

    kind = "llm"

    #: Name this provider is registered under; also the default component name.
    provider_name: str = "llm"

    #: Declare capability so agents can fail fast instead of mid-run.
    supports_tools: bool = True
    supports_streaming: bool = True

    def __init__(
        self,
        model: str = "",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        system_prompt: str | None = None,
        name: str | None = None,
        **config: Any,
    ) -> None:
        cfg = settings()
        super().__init__(
            name=name or self.provider_name,
            model=model,
            temperature=cfg.temperature if temperature is None else temperature,
            max_tokens=cfg.max_tokens if max_tokens is None else max_tokens,
            timeout=cfg.request_timeout if timeout is None else timeout,
            system_prompt=system_prompt,
            **config,
        )
        self.model = model or self.default_model()
        self.temperature: float = self.config["temperature"]
        self.max_tokens: int | None = self.config["max_tokens"]
        self.timeout: float = self.config["timeout"]
        self.system_prompt: str | None = system_prompt

    # -- provider hooks ---------------------------------------------------
    @classmethod
    def default_model(cls) -> str:
        """Return the model used when the caller does not name one.

        Returns:
            A provider-appropriate default model identifier.
        """
        return ""

    @abc.abstractmethod
    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Produce one completion.

        This is the only method a provider must implement.

        Args:
            messages: The conversation, already normalised and with the system
                prompt applied.
            tools: JSON-schema tool definitions in OpenAI function-calling
                format. Adapters translate these into their own shape.
            **kwargs: Per-call overrides (``temperature``, ``max_tokens``,
                ``response_format``, ...).

        Returns:
            The completion, with ``raw`` set to the provider's response.

        Raises:
            ProviderError: For any provider-side failure.
        """

    async def astream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Produce incremental completion events.

        The default implementation calls :meth:`agenerate` and emits the result
        as a single ``text`` event followed by ``done`` — correct, but not
        actually incremental. Providers that support streaming should override
        this.

        Args:
            messages: The normalised conversation.
            tools: Tool definitions.
            **kwargs: Per-call overrides.

        Yields:
            :class:`~windlass.core.types.StreamEvent` values.
        """
        completion = await self.agenerate(messages, tools=tools, **kwargs)
        if completion.content:
            yield StreamEvent(type="text", delta=completion.content)
        for call in completion.tool_calls:
            yield StreamEvent(type="tool_call", tool_call=call)
        yield StreamEvent(
            type="done", finish_reason=completion.finish_reason, usage=completion.usage
        )

    # -- public API -------------------------------------------------------
    async def acomplete(
        self,
        prompt: PromptLike,
        *,
        tools: list[dict[str, Any]] | None = None,
        retry: bool = True,
        **kwargs: Any,
    ) -> Completion:
        """Generate a completion, with retries and timing.

        Args:
            prompt: A string, a message, or a full transcript.
            tools: Tool definitions the model may call.
            retry: Apply the configured retry policy to transient failures.
            **kwargs: Per-call overrides forwarded to the provider.

        Returns:
            The completion. ``latency_ms`` is always populated.

        Raises:
            ProviderError: When the provider fails and retries are exhausted.
            AuthenticationError: When credentials are missing or rejected.

        Performance:
            One network round trip. Set ``retry=False`` on latency-critical
            paths where you would rather fail fast than wait out a backoff.

        Example:
            >>> import asyncio
            >>> from windlass.providers.llm.fake import FakeLLM
            >>> asyncio.run(FakeLLM(responses=["hi"]).acomplete("hello")).content
            'hi'
        """
        messages = self._prepare(prompt)
        started = time.perf_counter()

        async def _call() -> Completion:
            return await self.agenerate(messages, tools=tools, **kwargs)

        completion = (
            await retry_async(_call, description=f"{self.name}.generate")
            if retry
            else await _call()
        )
        if not completion.latency_ms:
            completion.latency_ms = (time.perf_counter() - started) * 1000
        if not completion.model:
            completion.model = self.model
        return completion

    def complete(
        self,
        prompt: PromptLike,
        *,
        tools: list[dict[str, Any]] | None = None,
        retry: bool = True,
        **kwargs: Any,
    ) -> Completion:
        """Blocking :meth:`acomplete`.

        Safe to call from a notebook or from inside a running event loop; the
        call is dispatched to a background loop when necessary.

        Args:
            prompt: A string, a message, or a full transcript.
            tools: Tool definitions the model may call.
            retry: Apply the configured retry policy.
            **kwargs: Per-call overrides.

        Returns:
            The completion.
        """
        return run_sync(self.acomplete(prompt, tools=tools, retry=retry, **kwargs))

    async def astream(
        self,
        prompt: PromptLike,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion as it is generated.

        Args:
            prompt: A string, a message, or a full transcript.
            tools: Tool definitions.
            **kwargs: Per-call overrides.

        Yields:
            Stream events. Text deltas arrive as they are produced; the final
            event is always ``type="done"``.

        Example:
            >>> import asyncio
            >>> from windlass.providers.llm.fake import FakeLLM
            >>> async def main():
            ...     llm = FakeLLM(responses=["hey"])
            ...     return "".join([e.delta async for e in llm.astream("hi")])
            >>> asyncio.run(main())
            'hey'
        """
        messages = self._prepare(prompt)
        async for event in self.astream_generate(messages, tools=tools, **kwargs):
            yield event

    def stream(
        self,
        prompt: PromptLike,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Blocking :meth:`astream`.

        Args:
            prompt: A string, a message, or a full transcript.
            tools: Tool definitions.
            **kwargs: Per-call overrides.

        Yields:
            Stream events, pulled one at a time so nothing is buffered ahead of
            the consumer.
        """
        return iter_sync(self.astream(prompt, tools=tools, **kwargs))

    async def astream_text(self, prompt: PromptLike, **kwargs: Any) -> AsyncIterator[str]:
        """Stream only the text deltas — the common case for chat UIs.

        Args:
            prompt: A string, a message, or a full transcript.
            **kwargs: Per-call overrides.

        Yields:
            Non-empty text fragments.
        """
        async for event in self.astream(prompt, **kwargs):
            if event.type == "text" and event.delta:
                yield event.delta

    async def abatch(
        self,
        prompts: Sequence[PromptLike],
        *,
        concurrency: int | None = None,
        **kwargs: Any,
    ) -> list[Completion]:
        """Complete many prompts with bounded concurrency.

        Args:
            prompts: The prompts to run.
            concurrency: Maximum simultaneous requests. Defaults to the global
                ``max_concurrency`` setting.
            **kwargs: Per-call overrides applied to every prompt.

        Returns:
            Completions in the same order as ``prompts``.

        Performance:
            Bounded on purpose — an unbounded fan-out over a few hundred prompts
            is the fastest way to get rate limited.

        Example:
            >>> import asyncio
            >>> from windlass.providers.llm.fake import FakeLLM
            >>> llm = FakeLLM(responses=["a", "b"])
            >>> [c.content for c in asyncio.run(llm.abatch(["1", "2"]))]
            ['a', 'b']
        """
        from windlass.core.concurrency import gather_bounded

        limit = concurrency or settings().max_concurrency
        return await gather_bounded([self.acomplete(p, **kwargs) for p in prompts], limit=limit)

    def batch(
        self, prompts: Sequence[PromptLike], *, concurrency: int | None = None, **kwargs: Any
    ) -> list[Completion]:
        """Blocking :meth:`abatch`."""
        return run_sync(self.abatch(prompts, concurrency=concurrency, **kwargs))

    # -- helpers ----------------------------------------------------------
    def _prepare(self, prompt: PromptLike) -> list[Message]:
        """Normalise a prompt and apply the configured system prompt."""
        messages = normalize_messages(prompt)
        if self.system_prompt and not (messages and messages[0].role.value == "system"):
            messages = [Message.system(self.system_prompt), *messages]
        return messages

    def _merged(self, **kwargs: Any) -> dict[str, Any]:
        """Merge per-call overrides over the component defaults.

        Drops ``None`` values so a provider never receives an explicit null for
        an option it would rather default itself.
        """
        merged: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        merged.update(kwargs)
        return {k: v for k, v in merged.items() if v is not None}

    @staticmethod
    def _usage(prompt_tokens: int = 0, completion_tokens: int = 0) -> Usage:
        """Build a :class:`Usage` record."""
        return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    @staticmethod
    def _finish_reason(value: str | None) -> FinishReason:
        """Map a provider's stop reason onto :class:`FinishReason`."""
        mapping = {
            "stop": FinishReason.STOP,
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "max_tokens": FinishReason.LENGTH,
            "MAX_TOKENS": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "tool_use": FinishReason.TOOL_CALLS,
            "function_call": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.CONTENT_FILTER,
            "SAFETY": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(value or "stop", FinishReason.STOP)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, temperature={self.temperature})"
