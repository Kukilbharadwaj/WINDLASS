"""A deterministic, dependency-free LLM for tests, demos and CI.

``FakeLLM`` is not a toy. It is the reason the entire Windlass test suite runs
offline in a few seconds with no API keys, and the reason ``pip install
windlass`` alone gives you a working end-to-end pipeline to learn from.

It can:

* replay a scripted list of responses;
* answer via a callable, so a test can assert on the prompt it received;
* emit tool calls on cue, exercising the whole agent loop;
* record every call it received for assertions.

Example:
    >>> llm = FakeLLM(responses=["first", "second"])
    >>> llm.complete("a").content, llm.complete("b").content
    ('first', 'second')
    >>> llm.calls[0][-1].content
    'a'
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from windlass.core.registry import register
from windlass.core.types import Completion, FinishReason, Message, StreamEvent, ToolCall, Usage
from windlass.interfaces.llm import LLM

__all__ = ["EchoLLM", "FakeLLM"]


@register.llm(
    "fake",
    aliases=("mock", "test"),
    description="Deterministic in-process model for tests and demos (no dependencies).",
)
class FakeLLM(LLM):
    """A scripted language model.

    Args:
        responses: Texts to return, one per call. Exhausting the list cycles
            back to the start when ``cycle`` is True, otherwise the last entry
            repeats.
        handler: Callable receiving ``(messages, tools)`` and returning either a
            string or a full :class:`~windlass.core.types.Completion`. Takes
            precedence over ``responses``.
        tool_calls: Tool calls to emit, one list per call. Use this to drive an
            agent through a specific tool-use path in a test.
        cycle: Whether ``responses`` wraps around.
        latency: Artificial delay per call in seconds, for testing timeouts and
            progress indicators.
        model: Reported model name.
        **config: Forwarded to :class:`~windlass.interfaces.llm.LLM`.

    Attributes:
        calls: Every message list this model was asked to complete.
        tool_schemas: The tool definitions passed on each call.

    Example:
        Driving an agent through one tool call, then a final answer::

            llm = FakeLLM(
                responses=["", "The weather is sunny."],
                tool_calls=[[ToolCall(name="weather", arguments={"city": "Paris"})], []],
            )
    """

    provider_name = "fake"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        responses: Sequence[str] | str | None = None,
        *,
        handler: Callable[[list[Message], list[dict[str, Any]] | None], Any] | None = None,
        tool_calls: Sequence[Sequence[ToolCall]] | None = None,
        cycle: bool = False,
        latency: float = 0.0,
        model: str = "fake-1",
        **config: Any,
    ) -> None:
        super().__init__(model=model, **config)
        if isinstance(responses, str):
            responses = [responses]
        self.responses: list[str] = list(responses or ["This is a fake response."])
        self.handler = handler
        self.tool_calls: list[list[ToolCall]] = [list(t) for t in (tool_calls or [])]
        self.cycle = cycle
        self.latency = latency
        self.calls: list[list[Message]] = []
        self.tool_schemas: list[list[dict[str, Any]] | None] = []
        self._counter = itertools.count()

    @classmethod
    def default_model(cls) -> str:
        """Return ``"fake-1"``."""
        return "fake-1"

    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Return the next scripted response.

        Args:
            messages: The conversation, recorded on :attr:`calls`.
            tools: Tool definitions, recorded on :attr:`tool_schemas`.
            **kwargs: Ignored.

        Returns:
            The scripted completion.
        """
        self.calls.append(list(messages))
        self.tool_schemas.append(tools)
        if self.latency:
            await asyncio.sleep(self.latency)

        index = next(self._counter)

        if self.handler is not None:
            produced = self.handler(messages, tools)
            if isinstance(produced, Completion):
                return produced
            return Completion(
                content=str(produced), model=self.model, usage=self._fake_usage(messages)
            )

        calls = self.tool_calls[index] if index < len(self.tool_calls) else []
        text = self._next_text(index)
        return Completion(
            content=text,
            tool_calls=list(calls),
            finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
            model=self.model,
            usage=self._fake_usage(messages, text),
            raw={"fake": True, "index": index},
        )

    async def astream_generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the scripted response one word at a time.

        Args:
            messages: The conversation.
            tools: Tool definitions.
            **kwargs: Ignored.

        Yields:
            Word-level text events, then any tool calls, then ``done``.
        """
        completion = await self.agenerate(messages, tools=tools, **kwargs)
        words = completion.content.split(" ")
        for position, word in enumerate(words):
            if not word and len(words) == 1:
                break
            delta = word if position == len(words) - 1 else f"{word} "
            if self.latency:
                await asyncio.sleep(self.latency / max(1, len(words)))
            yield StreamEvent(type="text", delta=delta)
        for call in completion.tool_calls:
            yield StreamEvent(type="tool_call", tool_call=call)
        yield StreamEvent(
            type="done", finish_reason=completion.finish_reason, usage=completion.usage
        )

    # -- test helpers -----------------------------------------------------
    def last_prompt(self) -> str:
        """Return the text of the most recent user message.

        Returns:
            The last user message's content, or ``""`` when there was none.

        Raises:
            IndexError: If the model has not been called yet.
        """
        messages = self.calls[-1]
        for message in reversed(messages):
            if message.role.value == "user":
                return message.content
        return ""

    def reset(self) -> None:
        """Clear recorded calls and rewind the response script."""
        self.calls.clear()
        self.tool_schemas.clear()
        self._counter = itertools.count()

    def _next_text(self, index: int) -> str:
        """Pick the response for call number ``index``."""
        if self.cycle:
            return self.responses[index % len(self.responses)]
        return self.responses[min(index, len(self.responses) - 1)]

    @staticmethod
    def _fake_usage(messages: list[Message], completion: str = "") -> Usage:
        """Produce plausible token counts so usage assertions have something to bite on."""
        prompt_chars = sum(len(m.content) for m in messages)
        return Usage(
            prompt_tokens=max(1, prompt_chars // 4),
            completion_tokens=max(1, len(completion) // 4),
        )


@register.llm(
    "echo",
    description="Returns the last user message verbatim (no dependencies).",
)
class EchoLLM(LLM):
    """A model that echoes its input.

    Useful for wiring tests where you need to prove that a prompt reached the
    model intact — the answer *is* the prompt.

    Args:
        prefix: Prepended to the echoed text.
        **config: Forwarded to :class:`~windlass.interfaces.llm.LLM`.

    Example:
        >>> EchoLLM().complete("hello there").content
        'hello there'
        >>> EchoLLM(prefix="You said: ").complete("hi").content
        'You said: hi'
    """

    provider_name = "echo"
    supports_tools = False

    def __init__(self, prefix: str = "", **config: Any) -> None:
        super().__init__(model="echo-1", **config)
        self.prefix = prefix

    @classmethod
    def default_model(cls) -> str:
        """Return ``"echo-1"``."""
        return "echo-1"

    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Echo the last user message.

        Args:
            messages: The conversation.
            tools: Ignored — this model does not call tools.
            **kwargs: Ignored.

        Returns:
            A completion containing the echoed text.
        """
        text = next(
            (m.content for m in reversed(messages) if m.role.value == "user"),
            messages[-1].content if messages else "",
        )
        return Completion(content=f"{self.prefix}{text}", model=self.model)
