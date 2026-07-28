"""The vector-store interface.

A vector store persists chunks together with their embeddings and answers
nearest-neighbour queries. FAISS, Chroma, Pinecone and the built-in in-memory
store all sit behind this one class, so moving from a laptop prototype to a
managed index is a one-word change in the builder.

Implementers override four coroutines: :meth:`VectorStore.aadd`,
:meth:`VectorStore.asearch`, :meth:`VectorStore.adelete` and
:meth:`VectorStore.acount`.

Example:
    >>> from windlass.providers.vectordb.memory import InMemoryVectorStore
    >>> from windlass.core.types import Chunk
    >>> store = InMemoryVectorStore(dimensions=3)
    >>> store.add([Chunk(content="hi", embedding=[1.0, 0.0, 0.0])])
    1
    >>> store.search([1.0, 0.0, 0.0], k=1)[0].chunk.content
    'hi'
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from windlass.core.concurrency import run_sync
from windlass.core.exceptions import ConfigurationError
from windlass.core.types import Chunk, ScoredChunk
from windlass.interfaces.base import Component

__all__ = ["MetadataFilter", "VectorStore"]

#: A metadata filter. Values may be scalars (equality) or operator dicts such as
#: ``{"$in": [...]}``, ``{"$gte": 5}``, ``{"$ne": "draft"}``.
MetadataFilter = dict[str, Any]


class VectorStore(Component):
    """Abstract vector database.

    Args:
        collection: Name of the collection / index / namespace to use.
        dimensions: Vector dimensionality. Required by stores that must create
            an index up front; inferred on first write by those that can.
        metric: Similarity metric — ``cosine``, ``dot`` or ``euclidean``.
        persist_path: Where an on-disk store should keep its data.
        name: Component name for traces.
        **config: Store-specific options.

    Attributes:
        collection: The active collection name.
        metric: The configured similarity metric.
        supports_filters: Whether metadata filtering is pushed down to the store
            rather than applied client-side.
        supports_hybrid: Whether the store has native sparse+dense search.

    Example:
        Implementing a store means four methods::

            class MyStore(VectorStore):
                provider_name = "mine"

                async def aadd(self, chunks): ...
                async def asearch(self, vector, k=5, filters=None): ...
                async def adelete(self, ids=None, filters=None): ...
                async def acount(self): ...
    """

    kind = "vectordb"
    provider_name: str = "vectordb"

    supports_filters: bool = True
    supports_hybrid: bool = False

    def __init__(
        self,
        *,
        collection: str = "windlass",
        dimensions: int | None = None,
        metric: str = "cosine",
        persist_path: str | None = None,
        name: str | None = None,
        **config: Any,
    ) -> None:
        if metric not in {"cosine", "dot", "euclidean", "l2"}:
            raise ConfigurationError(
                f"Unsupported metric {metric!r}.",
                hint="Use 'cosine', 'dot' or 'euclidean'.",
            )
        super().__init__(
            name=name or self.provider_name,
            collection=collection,
            dimensions=dimensions,
            metric=metric,
            persist_path=persist_path,
            **config,
        )
        self.collection = collection
        self.dimensions = dimensions
        self.metric = metric
        self.persist_path = persist_path

    # -- provider hooks ---------------------------------------------------
    @abc.abstractmethod
    async def aadd(self, chunks: Sequence[Chunk]) -> int:
        """Insert or update chunks.

        Chunk ids are deterministic, so re-adding the same content must upsert
        rather than duplicate.

        Args:
            chunks: Chunks with :attr:`~windlass.core.types.Chunk.embedding` set.

        Returns:
            How many chunks were written.

        Raises:
            ConfigurationError: If any chunk is missing its embedding.
            ProviderError: For store-side failures.
        """

    @abc.abstractmethod
    async def asearch(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Find the ``k`` nearest chunks to ``vector``.

        Args:
            vector: The query embedding.
            k: How many results to return.
            filters: Metadata constraints. Stores without native filtering
                should call :meth:`match_filters` client-side.
            **kwargs: Store-specific search options.

        Returns:
            Hits sorted by descending score, each with ``rank`` set.
        """

    @abc.abstractmethod
    async def adelete(
        self, ids: Sequence[str] | None = None, *, filters: MetadataFilter | None = None
    ) -> int:
        """Delete chunks by id or by metadata filter.

        Args:
            ids: Chunk ids to remove.
            filters: Metadata constraints selecting what to remove.

        Returns:
            How many chunks were deleted.

        Raises:
            ValueError: If neither ``ids`` nor ``filters`` is given — deleting
                the entire collection must go through :meth:`aclear`.
        """

    @abc.abstractmethod
    async def acount(self) -> int:
        """Return how many chunks the collection holds."""

    async def aget(self, ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id.

        The default returns ``[]``; stores that can look up by id should
        override it. Parent-child retrieval relies on this to expand a matched
        child into its parent.

        Args:
            ids: Chunk ids to fetch.

        Returns:
            The chunks that exist, in whatever order the store returns them.
        """
        return []

    async def aclear(self) -> None:
        """Remove every chunk from the collection.

        The default deletes by listing ids, which is correct but slow; stores
        with a native truncate should override it.
        """
        await self.adelete(filters={})

    async def apersist(self) -> None:
        """Flush pending writes to durable storage.

        A no-op for stores that write through.
        """

    # -- sync API ---------------------------------------------------------
    def add(self, chunks: Sequence[Chunk]) -> int:
        """Blocking :meth:`aadd`."""
        return run_sync(self.aadd(chunks))

    def search(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Blocking :meth:`asearch`."""
        return run_sync(self.asearch(vector, k, filters=filters, **kwargs))

    def delete(
        self, ids: Sequence[str] | None = None, *, filters: MetadataFilter | None = None
    ) -> int:
        """Blocking :meth:`adelete`."""
        return run_sync(self.adelete(ids, filters=filters))

    def count(self) -> int:
        """Blocking :meth:`acount`."""
        return run_sync(self.acount())

    def get(self, ids: Sequence[str]) -> list[Chunk]:
        """Blocking :meth:`aget`."""
        return run_sync(self.aget(ids))

    def clear(self) -> None:
        """Blocking :meth:`aclear`."""
        run_sync(self.aclear())

    def persist(self) -> None:
        """Blocking :meth:`apersist`."""
        run_sync(self.apersist())

    def __len__(self) -> int:
        return self.count()

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def validate_embeddings(chunks: Sequence[Chunk]) -> None:
        """Assert that every chunk carries an embedding.

        Args:
            chunks: Chunks about to be written.

        Raises:
            ConfigurationError: Naming the first chunk that is missing one.
        """
        for chunk in chunks:
            if not chunk.embedding:
                raise ConfigurationError(
                    f"Chunk {chunk.id!r} has no embedding.",
                    hint="Embed chunks before adding them, or use Pipeline.ingest() "
                    "which does it for you.",
                    context={"chunk_id": chunk.id},
                )

    @staticmethod
    def match_filters(metadata: dict[str, Any], filters: MetadataFilter | None) -> bool:
        """Evaluate a metadata filter client-side.

        Supports plain equality plus the Mongo-style operators ``$eq``, ``$ne``,
        ``$gt``, ``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin``, ``$contains``
        and ``$exists``. Stores without native filtering use this so that filter
        semantics are identical across every backend.

        Args:
            metadata: The chunk's metadata.
            filters: The constraints. ``None`` or ``{}`` matches everything.

        Returns:
            True when the metadata satisfies every constraint.

        Example:
            >>> VectorStore.match_filters({"year": 2024}, {"year": {"$gte": 2020}})
            True
            >>> VectorStore.match_filters({"tag": "a"}, {"tag": {"$in": ["b"]}})
            False
        """
        if not filters:
            return True
        for key, condition in filters.items():
            value = metadata.get(key)
            if not isinstance(condition, dict):
                if value != condition:
                    return False
                continue
            for op, operand in condition.items():
                if not _apply_operator(value, op, operand):
                    return False
        return True

    @staticmethod
    def rank(hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """Sort hits by descending score and assign 1-based ranks.

        Args:
            hits: Unordered hits.

        Returns:
            The same objects, sorted and with ``rank`` populated.
        """
        hits.sort(key=lambda h: h.score, reverse=True)
        for position, hit in enumerate(hits, start=1):
            hit.rank = position
        return hits

    def __repr__(self) -> str:
        return f"{type(self).__name__}(collection={self.collection!r}, metric={self.metric!r})"


def _apply_operator(value: Any, op: str, operand: Any) -> bool:
    """Evaluate one filter operator, returning False on type mismatches."""
    try:
        match op:
            case "$eq":
                return bool(value == operand)
            case "$ne":
                return bool(value != operand)
            case "$gt":
                return value is not None and value > operand
            case "$gte":
                return value is not None and value >= operand
            case "$lt":
                return value is not None and value < operand
            case "$lte":
                return value is not None and value <= operand
            case "$in":
                return value in operand
            case "$nin":
                return value not in operand
            case "$contains":
                return operand in (value or "")
            case "$exists":
                return (value is not None) == bool(operand)
            case _:
                raise ValueError(f"Unsupported filter operator {op!r}")
    except TypeError:
        # Comparing incomparable types means "does not match", not "crash".
        return False
