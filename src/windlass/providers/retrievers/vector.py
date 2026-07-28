"""Dense vector retrieval.

The standard semantic search path: embed the query with the same model that
embedded the corpus, ask the vector store for nearest neighbours, return them.

Two refinements are built in:

* **MMR** (``diversity > 0``) trades a little relevance for coverage, so you
  stop getting five near-identical chunks from the same page.
* **Parent expansion** swaps matched child chunks for their larger parents when
  the pipeline uses parent-child chunking.

Example:
    >>> from windlass.providers.embeddings.hash import HashEmbedder
    >>> from windlass.providers.vectordb.memory import InMemoryVectorStore
    >>> from windlass.core.types import Chunk
    >>> emb = HashEmbedder(dimensions=128)
    >>> store = InMemoryVectorStore(dimensions=128)
    >>> chunks = [Chunk(content=t) for t in ["kittens purr", "engines roar"]]
    >>> for c, v in zip(chunks, emb.embed([c.content for c in chunks])):
    ...     c.embedding = v
    >>> _ = store.add(chunks)
    >>> r = VectorRetriever(embedder=emb, vectorstore=store)
    >>> r.retrieve("kittens").hits[0].chunk.content
    'kittens purr'
"""

from __future__ import annotations

from typing import Any

from windlass.core.exceptions import ConfigurationError, RetrievalError
from windlass.core.registry import register
from windlass.core.types import ScoredChunk
from windlass.core.vectors import mmr
from windlass.interfaces.embedding import Embedder
from windlass.interfaces.retriever import Retriever
from windlass.interfaces.vectordb import MetadataFilter, VectorStore

__all__ = ["VectorRetriever"]


@register.retriever(
    "vector",
    aliases=("dense", "semantic", "similarity"),
    description="Dense embedding search, optionally with MMR diversification.",
)
class VectorRetriever(Retriever):
    """Dense retrieval over a vector store.

    Args:
        embedder: Model used to embed the query. Must be the same model that
            embedded the corpus.
        vectorstore: Where the vectors live.
        top_k: Default number of results.
        diversity: MMR strength in ``[0, 1]``. ``0`` disables MMR (plain
            similarity); ``0.3`` is a good default when your corpus has
            near-duplicate passages.
        fetch_k: Candidates fetched before MMR narrows them. Defaults to
            ``4 * top_k`` when MMR is on.
        expand_parents: Swap matched child chunks for their parents. Set
            automatically by the RAG builder when parent-child chunking is used.
        parent_source: Object exposing ``expand(chunks)`` — normally the
            :class:`~windlass.providers.chunkers.hierarchical.ParentChildChunker`.
        **config: Forwarded to :class:`~windlass.interfaces.retriever.Retriever`.

    Raises:
        ConfigurationError: When the embedder or vector store is missing.

    Performance:
        One embedding call plus one store query per retrieval. Enabling MMR
        fetches ``fetch_k`` candidates and does ``O(fetch_k²)`` similarity work
        in-process — negligible for the usual ``fetch_k ≤ 100``.
    """

    provider_name = "vector"

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        vectorstore: VectorStore | None = None,
        top_k: int = 5,
        diversity: float = 0.0,
        fetch_k: int | None = None,
        expand_parents: bool = False,
        parent_source: Any = None,
        **config: Any,
    ) -> None:
        if embedder is None:
            raise ConfigurationError(
                "Vector retrieval needs an embedding model.",
                hint="Pass embedder=..., or build the retriever through "
                "Windlass.rag(), which wires it for you.",
            )
        if vectorstore is None:
            raise ConfigurationError(
                "Vector retrieval needs a vector store.",
                hint="Pass vectorstore=..., or build the retriever through Windlass.rag().",
            )
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("diversity must be between 0 and 1")

        super().__init__(
            top_k=top_k,
            fetch_k=fetch_k or (top_k * 4 if diversity else top_k),
            **config,
        )
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.diversity = diversity
        self.expand_parents = expand_parents
        self.parent_source = parent_source

    def native(self) -> Any:
        """Return the underlying vector store's native handle."""
        return self.vectorstore.native()

    async def aindex(self, chunks: Any) -> int:
        """Embed and write chunks to the store.

        Chunks that already carry an embedding are written as-is, so this is
        cheap to call on an already-embedded batch.

        Args:
            chunks: Chunks to index.

        Returns:
            How many chunks were written.
        """
        items = list(chunks)
        if not items:
            return 0
        missing = [c for c in items if not c.embedding]
        if missing:
            vectors = await self.embedder.aembed([c.content for c in missing])
            for chunk, vector in zip(missing, vectors, strict=True):
                chunk.embedding = vector
        return await self.vectorstore.aadd(items)

    async def aretrieve_chunks(
        self,
        query: str,
        k: int,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Embed the query and search the store.

        Args:
            query: The search query.
            k: How many results to return.
            filters: Metadata constraints, pushed down to the store.
            **kwargs: Forwarded to the store's search method.

        Returns:
            Ranked hits.

        Raises:
            RetrievalError: When embedding or the store query fails.
        """
        try:
            vector = await self.embedder.aembed_query(query)
        except Exception as exc:
            raise RetrievalError(
                f"Could not embed the query: {exc}",
                hint="Check that the embedding provider is reachable and configured.",
            ) from exc

        fetch = max(k, self.fetch_k) if self.diversity else k
        try:
            hits = await self.vectorstore.asearch(vector, fetch, filters=filters, **kwargs)
        except Exception as exc:
            raise RetrievalError(f"Vector store query failed: {exc}") from exc

        if self.diversity and len(hits) > k:
            hits = self._diversify(vector, hits, k)
        else:
            hits = hits[:k]

        if self.expand_parents:
            hits = self._expand(hits)
        return hits

    # -- helpers ----------------------------------------------------------
    def _diversify(
        self, query_vector: list[float], hits: list[ScoredChunk], k: int
    ) -> list[ScoredChunk]:
        """Re-select ``k`` hits with Maximal Marginal Relevance.

        Hits whose chunks lost their embeddings in transit (some stores do not
        return vectors) fall back to plain top-k, since MMR needs the vectors.
        """
        vectors = [h.chunk.embedding for h in hits]
        if any(v is None for v in vectors):
            return hits[:k]
        selected = mmr(query_vector, [v for v in vectors if v], k=k, diversity=self.diversity)
        chosen = [hits[i] for i in selected]
        for rank, hit in enumerate(chosen, start=1):
            hit.rank = rank
        return chosen

    def _expand(self, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """Replace child chunks with their parents, keeping the best score."""
        expander = getattr(self.parent_source, "expand", None)
        if callable(expander):
            best: dict[str, float] = {}
            for hit in hits:
                key = hit.chunk.parent_id or hit.chunk.id
                best[key] = max(best.get(key, float("-inf")), hit.score)
            parents = expander([h.chunk for h in hits])
            return [
                ScoredChunk(
                    chunk=parent,
                    score=best.get(parent.id, 0.0),
                    retriever=self.name,
                )
                for parent in parents
            ]

        # No chunker on hand: fall back to the parent text carried in metadata.
        seen: set[str] = set()
        expanded: list[ScoredChunk] = []
        for hit in hits:
            parent_text = hit.chunk.metadata.get("parent_content")
            key = hit.chunk.parent_id or hit.chunk.id
            if key in seen:
                continue
            seen.add(key)
            if parent_text:
                expanded.append(
                    ScoredChunk(
                        chunk=hit.chunk.model_copy(update={"content": str(parent_text)}),
                        score=hit.score,
                        retriever=self.name,
                    )
                )
            else:
                expanded.append(hit)
        return expanded

    def __repr__(self) -> str:
        return (
            f"VectorRetriever(top_k={self.top_k}, diversity={self.diversity}, "
            f"store={type(self.vectorstore).__name__})"
        )
