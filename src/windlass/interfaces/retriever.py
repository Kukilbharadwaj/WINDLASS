"""The retriever interface.

A retriever answers "which chunks are relevant to this query?". That is a
different job from the vector store, which only answers "which vectors are
closest to this one" — the distinction is what lets BM25, hybrid fusion,
contextual retrieval and reranking all be retrievers while only some of them
touch a vector database.

Implementers override one coroutine, :meth:`Retriever.aretrieve_chunks`.

Example:
    >>> from windlass.providers.retrievers.bm25 import BM25Retriever
    >>> from windlass.core.types import Chunk
    >>> r = BM25Retriever()
    >>> r.index([Chunk(content="the cat sat"), Chunk(content="dogs bark")])
    2
    >>> r.retrieve("cat").hits[0].chunk.content
    'the cat sat'
"""

from __future__ import annotations

import abc
import time
from collections.abc import Sequence
from typing import Any

from windlass.core.concurrency import gather_bounded, run_sync
from windlass.core.types import Chunk, ScoredChunk, SearchResult
from windlass.interfaces.base import Component
from windlass.interfaces.vectordb import MetadataFilter

__all__ = ["Retriever"]


class Retriever(Component):
    """Abstract retrieval strategy.

    Args:
        top_k: Default number of chunks to return.
        score_threshold: Drop hits scoring below this. ``None`` keeps everything.
            Useful for "answer only if we actually found something" flows.
        rerank: Optional reranker applied to the candidate set before
            truncation. Any object with an ``arerank(query, hits, k)`` coroutine.
        fetch_k: How many candidates to pull before reranking/filtering.
            Defaults to ``top_k`` when no reranker is configured, ``4 * top_k``
            when one is.
        name: Component name for traces.
        **config: Strategy-specific options.

    Attributes:
        top_k: The configured result count.
        requires_index: Whether :meth:`aindex` must be called before searching.

    Example:
        Implementing a retriever takes one method::

            class RandomRetriever(Retriever):
                provider_name = "random"

                async def aretrieve_chunks(self, query, k, *, filters=None, **kw):
                    picks = random.sample(self.corpus, k)
                    return [ScoredChunk(chunk=c, score=1.0) for c in picks]
    """

    kind = "retriever"
    provider_name: str = "retriever"

    #: True when the retriever keeps its own index that must be populated.
    requires_index: bool = False

    def __init__(
        self,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        rerank: Any = None,
        fetch_k: int | None = None,
        name: str | None = None,
        **config: Any,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        super().__init__(
            name=name or self.provider_name,
            top_k=top_k,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            **config,
        )
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.reranker = rerank
        self.fetch_k = fetch_k or (top_k * 4 if rerank is not None else top_k)

    # -- provider hooks ---------------------------------------------------
    @abc.abstractmethod
    async def aretrieve_chunks(
        self,
        query: str,
        k: int,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Return candidate chunks for ``query``.

        The only method a strategy must implement. Thresholding, reranking,
        truncation and timing are applied by :meth:`aretrieve`.

        Args:
            query: The search query.
            k: How many candidates to produce. This is ``fetch_k``, not
                ``top_k``, when a reranker is configured.
            filters: Metadata constraints.
            **kwargs: Strategy-specific options.

        Returns:
            Scored chunks, ideally already sorted by descending score.
        """

    async def aindex(self, chunks: Sequence[Chunk]) -> int:
        """Add chunks to whatever index this retriever maintains.

        The default is a no-op, which is right for retrievers that read from a
        shared vector store. Lexical retrievers override it.

        Args:
            chunks: Chunks to index.

        Returns:
            How many chunks were indexed.
        """
        return 0

    def index(self, chunks: Sequence[Chunk]) -> int:
        """Blocking :meth:`aindex`."""
        return run_sync(self.aindex(chunks))

    # -- public API -------------------------------------------------------
    async def aretrieve(
        self,
        query: str,
        k: int | None = None,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """Retrieve chunks for a query.

        Args:
            query: The search query.
            k: Override for :attr:`top_k`.
            filters: Metadata constraints.
            **kwargs: Strategy-specific options.

        Returns:
            A :class:`~windlass.core.types.SearchResult` with ranked hits,
            candidate count and latency.

        Raises:
            RetrievalError: When the underlying strategy fails.

        Performance:
            With a reranker configured, ``fetch_k`` candidates are retrieved and
            then narrowed to ``k``. Raising ``fetch_k`` improves recall at the
            cost of one larger rerank call.

        Example:
            >>> import asyncio
            >>> from windlass.providers.retrievers.bm25 import BM25Retriever
            >>> from windlass.core.types import Chunk
            >>> r = BM25Retriever()
            >>> _ = r.index([Chunk(content="vector search rocks")])
            >>> asyncio.run(r.aretrieve("vector")).hits[0].score > 0
            True
        """
        limit = k or self.top_k
        candidates = max(limit, self.fetch_k) if self.reranker is not None else limit
        started = time.perf_counter()

        hits = await self.aretrieve_chunks(query, candidates, filters=filters, **kwargs)

        if self.reranker is not None and hits:
            hits = await self.reranker.arerank(query, hits, limit)

        if self.score_threshold is not None:
            hits = [h for h in hits if h.score >= self.score_threshold]

        hits = sorted(hits, key=lambda h: h.score, reverse=True)[:limit]
        for position, hit in enumerate(hits, start=1):
            hit.rank = position
            if not hit.retriever:
                hit.retriever = self.name

        return SearchResult(
            query=query,
            hits=hits,
            total=len(hits),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """Blocking :meth:`aretrieve`."""
        return run_sync(self.aretrieve(query, k, filters=filters, **kwargs))

    async def abatch_retrieve(
        self,
        queries: Sequence[str],
        k: int | None = None,
        *,
        filters: MetadataFilter | None = None,
        concurrency: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve for many queries concurrently.

        Args:
            queries: The queries to run.
            k: Override for :attr:`top_k`.
            filters: Metadata constraints applied to every query.
            concurrency: Maximum simultaneous retrievals.

        Returns:
            One result per query, in input order.
        """
        from windlass.core.config import settings

        limit = concurrency or settings().max_concurrency
        return await gather_bounded(
            [self.aretrieve(q, k, filters=filters) for q in queries], limit=limit
        )

    def batch_retrieve(self, queries: Sequence[str], k: int | None = None) -> list[SearchResult]:
        """Blocking :meth:`abatch_retrieve`."""
        return run_sync(self.abatch_retrieve(queries, k))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(top_k={self.top_k})"
