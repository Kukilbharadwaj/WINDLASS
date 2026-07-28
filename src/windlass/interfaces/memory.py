"""The memory interface.

Memory is what turns a stateless model call into a conversation. Windlass
separates two concerns that are often conflated:

* **Conversation memory** — the recent transcript, replayed into the prompt.
  Buffer, sliding-window and summarising strategies live here.
* **Long-term memory** — durable facts recalled by *relevance* rather than
  recency, backed by a vector store.

Both implement this interface, so an agent can hold one, the other, or a
composite of both without knowing the difference.

Implementers override :meth:`Memory.aadd` and :meth:`Memory.aget`.

Example:
    >>> from windlass.providers.memory.conversation import BufferMemory
    >>> from windlass.core.types import Message
    >>> m = BufferMemory()
    >>> m.add(Message.user("hi"))
    >>> len(m.get())
    1
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from windlass.core.concurrency import run_sync
from windlass.core.types import Message
from windlass.interfaces.base import Component

__all__ = ["Memory"]

#: Thread id used when the caller does not specify one.
DEFAULT_THREAD = "default"


class Memory(Component):
    """Abstract conversation / long-term memory.

    Memory is keyed by ``thread_id`` so one instance can serve many concurrent
    users — an agent handling a hundred chat sessions needs one memory object,
    not a hundred.

    Args:
        max_messages: Ceiling on how many messages :meth:`aget` returns.
            ``None`` means unlimited.
        return_system: Whether system messages are included in recall. Usually
            False: the system prompt is supplied by the agent, not the history.
        name: Component name for traces.
        **config: Strategy-specific options.

    Example:
        Implementing a memory takes two methods::

            class NullMemory(Memory):
                provider_name = "null"

                async def aadd(self, messages, *, thread_id="default"): ...
                async def aget(self, *, thread_id="default", query=None):
                    return []
    """

    kind = "memory"
    provider_name: str = "memory"

    #: True when recall is relevance-based and benefits from a ``query``.
    semantic: bool = False

    def __init__(
        self,
        *,
        max_messages: int | None = None,
        return_system: bool = False,
        name: str | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            name=name or self.provider_name,
            max_messages=max_messages,
            return_system=return_system,
            **config,
        )
        self.max_messages = max_messages
        self.return_system = return_system

    # -- provider hooks ---------------------------------------------------
    @abc.abstractmethod
    async def aadd(
        self, messages: Message | Sequence[Message], *, thread_id: str = DEFAULT_THREAD
    ) -> None:
        """Record one or more messages.

        Args:
            messages: A message or a sequence of them.
            thread_id: Conversation this belongs to.

        Raises:
            WindlassMemoryError: When the backend cannot persist the messages.
        """

    @abc.abstractmethod
    async def aget(
        self,
        *,
        thread_id: str = DEFAULT_THREAD,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Recall messages for a thread.

        Args:
            thread_id: Conversation to recall.
            query: Current user input. Semantic memories use it to rank recall;
                recency-based memories ignore it.
            limit: Override for :attr:`max_messages`.

        Returns:
            Messages in chronological order, oldest first.
        """

    async def aclear(self, *, thread_id: str | None = None) -> None:
        """Forget a thread, or everything when ``thread_id`` is ``None``.

        Args:
            thread_id: Thread to clear. ``None`` clears every thread.
        """

    async def athreads(self) -> list[str]:
        """Return the thread ids this memory knows about."""
        return []

    # -- sync API ---------------------------------------------------------
    def add(
        self, messages: Message | Sequence[Message], *, thread_id: str = DEFAULT_THREAD
    ) -> None:
        """Blocking :meth:`aadd`."""
        run_sync(self.aadd(messages, thread_id=thread_id))

    def get(
        self,
        *,
        thread_id: str = DEFAULT_THREAD,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Blocking :meth:`aget`."""
        return run_sync(self.aget(thread_id=thread_id, query=query, limit=limit))

    def clear(self, *, thread_id: str | None = None) -> None:
        """Blocking :meth:`aclear`."""
        run_sync(self.aclear(thread_id=thread_id))

    def threads(self) -> list[str]:
        """Blocking :meth:`athreads`."""
        return run_sync(self.athreads())

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _as_list(messages: Message | Sequence[Message]) -> list[Message]:
        """Coerce the ``messages`` argument into a list."""
        return [messages] if isinstance(messages, Message) else list(messages)

    def _filter(self, messages: list[Message], limit: int | None) -> list[Message]:
        """Apply the system-message and length policies to a recall result."""
        out = messages
        if not self.return_system:
            out = [m for m in out if m.role.value != "system"]
        effective = limit if limit is not None else self.max_messages
        if effective is not None and effective >= 0:
            out = out[-effective:] if effective else []
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_messages={self.max_messages})"
