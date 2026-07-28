"""Long-term memory: durable facts recalled by relevance.

Conversation memory answers "what did we just say?". Long-term memory answers
"what do I know about this user?" — and the right recall mechanism for that is
*similarity*, not recency. A preference stated three months ago should surface
when it becomes relevant again, not scroll out of a window.

:class:`VectorMemory` embeds each remembered item and retrieves the closest ones
to the current input. It works with any Windlass vector store, so the same code
runs against an in-memory index in tests and Pinecone in production.

Example:
    >>> from windlass.providers.embeddings.hash import HashEmbedder
    >>> from windlass.core.types import Message
    >>> memory = VectorMemory(
    ...     embedder=HashEmbedder(dimensions=128), top_k=1, score_threshold=None
    ... )
    >>> memory.add(Message.user("I am allergic to peanuts"))
    >>> memory.add(Message.user("My favourite colour is blue"))
    >>> "peanuts" in memory.get(query="what foods should I avoid")[0].content
    True
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from windlass.core.exceptions import ConfigurationError
from windlass.core.registry import register
from windlass.core.types import Chunk, Message, Role
from windlass.interfaces.embedding import Embedder
from windlass.interfaces.memory import DEFAULT_THREAD, Memory
from windlass.interfaces.vectordb import VectorStore

__all__ = ["CompositeMemory", "VectorMemory"]


@register.memory(
    "vector",
    aliases=("long-term", "longterm", "semantic"),
    description="Durable facts recalled by embedding similarity.",
)
class VectorMemory(Memory):
    """Long-term memory backed by a vector store.

    Args:
        embedder: Model used to embed remembered items and the recall query.
            Required.
        vectorstore: Where memories live. Defaults to a fresh in-memory store,
            which is fine for a single process but does not survive a restart —
            pass a persistent store for real deployments.
        top_k: How many memories to recall per query.
        score_threshold: Drop recalled memories scoring below this. Prevents
            irrelevant memories being injected into every prompt just because
            they were the closest of a bad set.
        store_roles: Which message roles are worth remembering. Assistant
            chatter usually is not.
        collection: Collection name used when creating the default store.
        **config: Forwarded to :class:`~windlass.interfaces.memory.Memory`.

    Raises:
        ConfigurationError: When no embedder is supplied.

    Performance:
        One embedding call per :meth:`aadd` batch and one per :meth:`aget`.
        Recall cost is whatever the underlying store charges.
    """

    provider_name = "vector"
    semantic = True

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        vectorstore: VectorStore | None = None,
        top_k: int = 5,
        score_threshold: float | None = 0.2,
        store_roles: tuple[str, ...] = ("user", "assistant"),
        collection: str = "windlass-memory",
        **config: Any,
    ) -> None:
        if embedder is None:
            raise ConfigurationError(
                "Long-term memory needs an embedding model.",
                hint="Pass embedder=Windlass.embedding('huggingface'), or use "
                ".memory('vector') on a RAG-aware builder, which injects one.",
            )
        super().__init__(**config)
        self.embedder = embedder
        if vectorstore is None:
            from windlass.providers.vectordb.memory import InMemoryVectorStore

            vectorstore = InMemoryVectorStore(
                collection=collection, dimensions=embedder.dimension()
            )
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.store_roles = tuple(store_roles)

    def native(self) -> Any:
        """Return the underlying vector store's native handle."""
        return self.vectorstore.native()

    async def aadd(
        self, messages: Message | Sequence[Message], *, thread_id: str = DEFAULT_THREAD
    ) -> None:
        """Embed and store messages worth remembering.

        Args:
            messages: A message or a sequence of them.
            thread_id: Conversation this belongs to; recorded in metadata so
                recall can be scoped per user.
        """
        items = [
            m
            for m in self._as_list(messages)
            if m.content.strip() and m.role.value in self.store_roles
        ]
        if not items:
            return

        vectors = await self.embedder.aembed([m.content for m in items])
        chunks = [
            Chunk(
                content=message.content,
                embedding=vector,
                metadata={
                    "thread_id": thread_id,
                    "role": message.role.value,
                    "stored_at": time.time(),
                    **message.metadata,
                },
            )
            for message, vector in zip(items, vectors, strict=True)
        ]
        await self.vectorstore.aadd(chunks)

    async def aget(
        self,
        *,
        thread_id: str = DEFAULT_THREAD,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Recall the memories most relevant to ``query``.

        Args:
            thread_id: Scope recall to one conversation.
            query: The current user input. Without it, nothing is recalled —
                relevance-based memory has no notion of "the recent ones".
            limit: Override for :attr:`top_k`.

        Returns:
            Recalled messages, oldest first so they read as history.
        """
        if not query:
            return []

        vector = await self.embedder.aembed_query(query)
        hits = await self.vectorstore.asearch(
            vector, limit or self.top_k, filters={"thread_id": thread_id}
        )
        if self.score_threshold is not None:
            hits = [h for h in hits if h.score >= self.score_threshold]

        recalled = [
            Message(
                role=Role(h.chunk.metadata.get("role", "user")),
                content=h.chunk.content,
                metadata={"recalled": True, "score": round(h.score, 4)},
            )
            for h in hits
        ]
        recalled.sort(key=lambda m: m.metadata.get("score", 0.0))
        return recalled

    async def aclear(self, *, thread_id: str | None = None) -> None:
        """Forget one thread's memories, or all of them."""
        if thread_id is None:
            await self.vectorstore.aclear()
        else:
            await self.vectorstore.adelete(filters={"thread_id": thread_id})

    async def aremember(
        self, fact: str, *, thread_id: str = DEFAULT_THREAD, **metadata: Any
    ) -> None:
        """Store a bare fact, without wrapping it in a conversation turn.

        Args:
            fact: The text to remember.
            thread_id: Scope for the memory.
            **metadata: Extra fields stored alongside it.

        Example:
            >>> import asyncio
            >>> from windlass.providers.embeddings.hash import HashEmbedder
            >>> m = VectorMemory(embedder=HashEmbedder(dimensions=64))
            >>> asyncio.run(m.aremember("prefers dark mode", source="settings"))
        """
        message = Message(role=Role.USER, content=fact, metadata=metadata)
        await self.aadd([message], thread_id=thread_id)

    def remember(self, fact: str, *, thread_id: str = DEFAULT_THREAD, **metadata: Any) -> None:
        """Blocking :meth:`aremember`."""
        from windlass.core.concurrency import run_sync

        run_sync(self.aremember(fact, thread_id=thread_id, **metadata))


@register.memory(
    "composite",
    aliases=("hybrid-memory", "both"),
    description="Combines conversation memory with long-term recall.",
)
class CompositeMemory(Memory):
    """Runs several memories together.

    The usual configuration is a window for the recent transcript plus vector
    memory for durable facts, which is how a production assistant both follows
    the current thread and remembers what you told it last month.

    Args:
        memories: The memories to combine, in the order their recall should be
            concatenated. Long-term memories should come first so the recent
            transcript stays closest to the model's most recent attention.
        **config: Forwarded to :class:`~windlass.interfaces.memory.Memory`.

    Raises:
        ConfigurationError: When fewer than one memory is supplied.

    Example:
        >>> from windlass.providers.memory.conversation import WindowMemory
        >>> from windlass.core.types import Message
        >>> combined = CompositeMemory(memories=[WindowMemory(window=2)])
        >>> combined.add(Message.user("hello"))
        >>> len(combined.get())
        1
    """

    provider_name = "composite"
    semantic = True

    def __init__(self, *, memories: Sequence[Memory] | None = None, **config: Any) -> None:
        items = list(memories or [])
        if not items:
            raise ConfigurationError(
                "Composite memory needs at least one memory to combine.",
                hint="CompositeMemory(memories=[WindowMemory(), VectorMemory(...)])",
            )
        super().__init__(**config)
        self.memories = items

    async def aadd(
        self, messages: Message | Sequence[Message], *, thread_id: str = DEFAULT_THREAD
    ) -> None:
        """Write to every constituent memory."""
        from windlass.core.concurrency import gather_bounded

        await gather_bounded(
            [m.aadd(messages, thread_id=thread_id) for m in self.memories],
            limit=len(self.memories),
        )

    async def aget(
        self,
        *,
        thread_id: str = DEFAULT_THREAD,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Concatenate recall from every constituent memory, de-duplicated."""
        from windlass.core.concurrency import gather_bounded

        results = await gather_bounded(
            [m.aget(thread_id=thread_id, query=query, limit=limit) for m in self.memories],
            limit=len(self.memories),
            return_exceptions=True,
        )
        seen: set[str] = set()
        combined: list[Message] = []
        for result in results:
            if isinstance(result, BaseException):
                self._log.warning("A memory failed during recall: %s", result)
                continue
            for message in result:
                key = f"{message.role.value}:{message.content}"
                if key in seen:
                    continue
                seen.add(key)
                combined.append(message)
        return combined

    async def aclear(self, *, thread_id: str | None = None) -> None:
        """Clear every constituent memory."""
        for memory in self.memories:
            await memory.aclear(thread_id=thread_id)

    def describe(self) -> dict[str, Any]:
        """Return a summary including each constituent memory."""
        return {**super().describe(), "memories": [m.describe() for m in self.memories]}
