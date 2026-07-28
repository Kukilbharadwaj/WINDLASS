"""Conversation memory: buffer, sliding window and summarising strategies.

All three are recency-based — they replay recent turns into the prompt. The
difference is what they do when the conversation outgrows the context window:

* :class:`BufferMemory` keeps everything and eventually overflows. Correct for
  short sessions, wrong for long ones.
* :class:`WindowMemory` keeps the last ``n`` turns. Cheap, predictable, forgets
  the beginning.
* :class:`SummaryMemory` summarises what falls out of the window, so the model
  keeps a compressed account of the whole conversation.

All are keyed by ``thread_id`` and safe to share across concurrent sessions.

Example:
    >>> from windlass.core.types import Message
    >>> m = WindowMemory(window=2)
    >>> m.add([Message.user("one"), Message.user("two"), Message.user("three")])
    >>> [msg.content for msg in m.get()]
    ['two', 'three']
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from windlass.core.exceptions import ConfigurationError
from windlass.core.registry import register
from windlass.core.types import Message, Role
from windlass.interfaces.llm import LLM
from windlass.interfaces.memory import DEFAULT_THREAD, Memory

__all__ = ["SUMMARY_PROMPT", "BufferMemory", "SummaryMemory", "WindowMemory"]

SUMMARY_PROMPT = """\
Update the running summary of a conversation.

Existing summary:
{summary}

New exchanges to fold in:
{messages}

Write the updated summary. Preserve names, decisions, numbers, preferences and
open questions — those are what the assistant will need later. Drop pleasantries
and anything already resolved. Answer with the summary text only.
"""


@register.memory(
    "buffer",
    aliases=("conversation", "simple"),
    description="Keeps the full transcript per thread (no dependencies).",
)
class BufferMemory(Memory):
    """Stores the complete transcript for each thread.

    Args:
        max_messages: Ceiling on how many messages :meth:`aget` returns.
            ``None`` returns everything.
        max_stored: Hard ceiling on messages retained per thread, to bound
            memory in a long-running process. Oldest messages are dropped.
        **config: Forwarded to :class:`~windlass.interfaces.memory.Memory`.

    Example:
        >>> from windlass.core.types import Message
        >>> m = BufferMemory()
        >>> m.add(Message.user("hi"), thread_id="a")
        >>> m.get(thread_id="a")[0].content
        'hi'
        >>> m.get(thread_id="b")
        []
    """

    provider_name = "buffer"

    def __init__(
        self, *, max_messages: int | None = None, max_stored: int = 1000, **config: Any
    ) -> None:
        super().__init__(max_messages=max_messages, **config)
        self.max_stored = max_stored
        self._threads: dict[str, list[Message]] = {}
        self._lock = threading.RLock()

    async def aadd(
        self, messages: Message | Sequence[Message], *, thread_id: str = DEFAULT_THREAD
    ) -> None:
        """Append messages to a thread."""
        items = self._as_list(messages)
        if not items:
            return
        with self._lock:
            bucket = self._threads.setdefault(thread_id, [])
            bucket.extend(items)
            if len(bucket) > self.max_stored:
                del bucket[: len(bucket) - self.max_stored]

    async def aget(
        self,
        *,
        thread_id: str = DEFAULT_THREAD,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Return the thread's messages, oldest first."""
        with self._lock:
            messages = list(self._threads.get(thread_id, []))
        return self._filter(messages, limit)

    async def aclear(self, *, thread_id: str | None = None) -> None:
        """Forget one thread, or all of them."""
        with self._lock:
            if thread_id is None:
                self._threads.clear()
            else:
                self._threads.pop(thread_id, None)

    async def athreads(self) -> list[str]:
        """Return the known thread ids."""
        with self._lock:
            return sorted(self._threads)


@register.memory(
    "window",
    aliases=("sliding", "recent"),
    description="Keeps only the most recent turns (no dependencies).",
)
class WindowMemory(BufferMemory):
    """Keeps a sliding window of recent messages.

    Args:
        window: How many messages to retain and return. Counted in messages, not
            turns, so a user/assistant exchange is two.
        **config: Forwarded to :class:`BufferMemory`.

    Raises:
        ValueError: If ``window`` is not positive.

    Note:
        A window that cuts between an assistant's tool call and the matching
        tool result produces a transcript some providers reject. :meth:`aget`
        detects that and widens the window to keep the pair together.

    Example:
        >>> from windlass.core.types import Message
        >>> m = WindowMemory(window=1)
        >>> m.add([Message.user("a"), Message.user("b")])
        >>> [x.content for x in m.get()]
        ['b']
    """

    provider_name = "window"

    def __init__(self, *, window: int = 20, **config: Any) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        config.pop("max_messages", None)
        super().__init__(max_messages=window, **config)
        self.window = window

    async def aget(
        self,
        *,
        thread_id: str = DEFAULT_THREAD,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Return the most recent messages, keeping tool pairs intact."""
        with self._lock:
            messages = list(self._threads.get(thread_id, []))
        size = limit if limit is not None else self.window
        if not self.return_system:
            messages = [m for m in messages if m.role != Role.SYSTEM]
        if size <= 0 or len(messages) <= size:
            return messages
        start = len(messages) - size
        # Never start on an orphaned tool result.
        while start > 0 and messages[start].role == Role.TOOL:
            start -= 1
        return messages[start:]


@register.memory(
    "summary",
    aliases=("summarizing", "compressed"),
    description="Summarises older turns with an LLM to keep long chats in budget.",
)
class SummaryMemory(BufferMemory):
    """Keeps recent turns verbatim and a running summary of everything older.

    When the transcript exceeds ``window``, the overflow is folded into a
    summary that is replayed as a system message. The conversation stays inside
    the context window without simply forgetting its own beginning.

    Args:
        llm: Model used to write summaries. Use a small, cheap one.
        window: How many recent messages to keep verbatim.
        summarize_every: Summarise once this many messages have overflowed, so
            the model is not called on every single turn.
        **config: Forwarded to :class:`BufferMemory`.

    Raises:
        ConfigurationError: When no model is supplied.

    Performance:
        One model call per ``summarize_every`` overflowed messages, made inline
        during :meth:`aget`. Summarisation failures are logged and skipped —
        memory degrades to a plain window rather than breaking the conversation.

    Example:
        >>> from windlass.core.types import Message
        >>> from windlass.providers.llm.fake import FakeLLM
        >>> m = SummaryMemory(
        ...     llm=FakeLLM(responses=["User greeted twice."]),
        ...     window=2,
        ...     summarize_every=2,
        ... )
        >>> m.add([Message.user(f"msg {i}") for i in range(6)])
        >>> recalled = m.get()
        >>> recalled[0].role.value
        'system'
    """

    provider_name = "summary"

    def __init__(
        self,
        *,
        llm: LLM | None = None,
        window: int = 10,
        summarize_every: int = 6,
        **config: Any,
    ) -> None:
        if llm is None:
            raise ConfigurationError(
                "Summary memory needs a language model.",
                hint="Pass llm=Windlass.llm('openai', model='gpt-4o-mini').",
            )
        config.pop("max_messages", None)
        super().__init__(max_messages=None, **config)
        self.llm = llm
        self.window = window
        self.summarize_every = max(1, summarize_every)
        self._summaries: dict[str, str] = {}
        self._summarized: dict[str, int] = {}

    async def aget(
        self,
        *,
        thread_id: str = DEFAULT_THREAD,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Return a summary message followed by the recent window.

        Args:
            thread_id: Conversation to recall.
            query: Ignored — recall is recency-based.
            limit: Override for the window size.

        Returns:
            The recalled messages, with any summary prepended as a system
            message.
        """
        with self._lock:
            messages = list(self._threads.get(thread_id, []))
            already = self._summarized.get(thread_id, 0)

        if not self.return_system:
            messages = [m for m in messages if m.role != Role.SYSTEM]
        size = limit if limit is not None else self.window

        overflow = messages[already : max(already, len(messages) - size)]
        if len(overflow) >= self.summarize_every:
            await self._fold(thread_id, overflow)
            already += len(overflow)

        recent = messages[-size:] if size > 0 else []
        while len(recent) > 1 and recent[0].role == Role.TOOL:
            recent = recent[1:]

        summary = self._summaries.get(thread_id, "")
        if summary:
            return [Message.system(f"Summary of the earlier conversation:\n{summary}"), *recent]
        return recent

    async def _fold(self, thread_id: str, overflow: list[Message]) -> None:
        """Fold overflowed messages into the running summary."""
        transcript = "\n".join(str(m) for m in overflow)
        existing = self._summaries.get(thread_id, "(none yet)")
        try:
            completion = await self.llm.acomplete(
                SUMMARY_PROMPT.format(summary=existing, messages=transcript),
                max_tokens=400,
            )
        except Exception as exc:
            self._log.warning("Conversation summarisation failed: %s", exc)
            return
        text = completion.content.strip()
        if not text:
            return
        with self._lock:
            self._summaries[thread_id] = text
            self._summarized[thread_id] = self._summarized.get(thread_id, 0) + len(overflow)

    def summary(self, thread_id: str = DEFAULT_THREAD) -> str:
        """Return the current summary for a thread, or ``""``."""
        return self._summaries.get(thread_id, "")

    async def aclear(self, *, thread_id: str | None = None) -> None:
        """Forget messages and summaries."""
        await super().aclear(thread_id=thread_id)
        with self._lock:
            if thread_id is None:
                self._summaries.clear()
                self._summarized.clear()
            else:
                self._summaries.pop(thread_id, None)
                self._summarized.pop(thread_id, None)
