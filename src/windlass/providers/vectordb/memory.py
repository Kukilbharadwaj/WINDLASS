"""An in-process vector store with optional JSON persistence.

This is the default store, and it is deliberately not a toy: it does exact
nearest-neighbour search (no recall loss from approximation), supports the full
metadata filter language, upserts by chunk id, and can save to and load from
disk. For corpora up to roughly 100k chunks on a machine with NumPy installed it
is genuinely fast enough for production.

Beyond that, switch to FAISS (local, approximate, millions of vectors) or
Pinecone (managed, distributed). The interface does not change.

Example:
    >>> from windlass.core.types import Chunk
    >>> store = InMemoryVectorStore(dimensions=3)
    >>> store.add([
    ...     Chunk(content="cats", embedding=[1.0, 0.0, 0.0], metadata={"topic": "pets"}),
    ...     Chunk(content="ships", embedding=[0.0, 1.0, 0.0], metadata={"topic": "sea"}),
    ... ])
    2
    >>> store.search([1.0, 0.0, 0.0], k=1)[0].chunk.content
    'cats'
    >>> store.search([1.0, 0.0, 0.0], k=2, filters={"topic": "sea"})[0].chunk.content
    'ships'
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from windlass.core.exceptions import ProviderError
from windlass.core.registry import register
from windlass.core.types import Chunk, ScoredChunk
from windlass.core.vectors import HAS_NUMPY, cosine_similarity, dot, euclidean_distance, top_k
from windlass.interfaces.vectordb import MetadataFilter, VectorStore

__all__ = ["InMemoryVectorStore"]


@register.vectordb(
    "memory",
    aliases=("inmemory", "in-memory", "local"),
    description="Exact in-process vector search with optional JSON persistence (no dependencies).",
)
class InMemoryVectorStore(VectorStore):
    """Exact nearest-neighbour search over an in-memory list.

    Args:
        collection: Logical collection name, used in the persistence filename.
        dimensions: Expected vector length. Inferred from the first write when
            omitted, then enforced on every subsequent write.
        metric: ``cosine``, ``dot`` or ``euclidean``.
        persist_path: Directory (or file) to save to. When it already contains
            data for this collection it is loaded on construction.
        autosave: Persist after every write. Convenient for notebooks, slow for
            bulk ingestion — leave it off and call :meth:`persist` once.
        **config: Forwarded to :class:`~windlass.interfaces.vectordb.VectorStore`.

    Attributes:
        chunks: The stored chunks, keyed by id.

    Performance:
        Search is ``O(n)`` in the number of chunks. With NumPy installed the
        whole matrix is scored in one vectorised operation, which keeps 100k
        chunks in the low tens of milliseconds; without it, expect roughly two
        orders of magnitude slower.

    Thread safety:
        All mutating operations hold a re-entrant lock, so concurrent ingestion
        and search are safe.
    """

    provider_name = "memory"
    supports_filters = True

    def __init__(
        self,
        *,
        collection: str = "windlass",
        dimensions: int | None = None,
        metric: str = "cosine",
        persist_path: str | None = None,
        autosave: bool = False,
        **config: Any,
    ) -> None:
        super().__init__(
            collection=collection,
            dimensions=dimensions,
            metric=metric,
            persist_path=persist_path,
            **config,
        )
        self.autosave = autosave
        self.chunks: dict[str, Chunk] = {}
        self._lock = threading.RLock()
        if persist_path and self._file().exists():
            self.load(self._file())

    # -- writes -----------------------------------------------------------
    async def aadd(self, chunks: Sequence[Chunk]) -> int:
        """Insert or update chunks.

        Args:
            chunks: Chunks with embeddings set.

        Returns:
            How many chunks were written.

        Raises:
            ConfigurationError: If a chunk has no embedding.
            ProviderError: If a chunk's vector length disagrees with the
                collection's dimensionality — usually a sign that two different
                embedding models were used against one index.
        """
        if not chunks:
            return 0
        self.validate_embeddings(chunks)

        with self._lock:
            if self.dimensions is None:
                self.dimensions = len(chunks[0].embedding or [])
            for chunk in chunks:
                vector = chunk.embedding or []
                if len(vector) != self.dimensions:
                    raise ProviderError(
                        f"Chunk {chunk.id!r} has {len(vector)} dimensions but the "
                        f"{self.collection!r} collection expects {self.dimensions}.",
                        provider="memory",
                        hint="All chunks in one collection must come from the same "
                        "embedding model. Clear the store or use a new collection.",
                        context={"chunk_id": chunk.id},
                    )
                self.chunks[chunk.id] = chunk

        if self.autosave:
            await self.apersist()
        return len(chunks)

    async def adelete(
        self, ids: Sequence[str] | None = None, *, filters: MetadataFilter | None = None
    ) -> int:
        """Delete by id or metadata filter.

        Args:
            ids: Chunk ids to remove.
            filters: Metadata constraints. ``{}`` matches everything.

        Returns:
            How many chunks were removed.

        Raises:
            ValueError: When neither argument is supplied.
        """
        if ids is None and filters is None:
            raise ValueError("Pass ids or filters; use clear() to empty the collection.")

        with self._lock:
            targets: set[str] = set(ids or ())
            if filters is not None:
                targets |= {
                    cid
                    for cid, chunk in self.chunks.items()
                    if self.match_filters(chunk.metadata, filters)
                }
            removed = 0
            for cid in targets:
                if self.chunks.pop(cid, None) is not None:
                    removed += 1

        if removed and self.autosave:
            await self.apersist()
        return removed

    async def aclear(self) -> None:
        """Remove every chunk."""
        with self._lock:
            self.chunks.clear()
        if self.autosave:
            await self.apersist()

    # -- reads ------------------------------------------------------------
    async def asearch(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Return the ``k`` nearest chunks.

        Args:
            vector: The query embedding.
            k: How many results to return.
            filters: Metadata constraints applied *before* scoring, so a narrow
                filter makes search proportionally faster.
            **kwargs: Ignored.

        Returns:
            Ranked hits. Euclidean scores are negated distances, so higher is
            always better regardless of metric.
        """
        with self._lock:
            candidates = [
                chunk
                for chunk in self.chunks.values()
                if self.match_filters(chunk.metadata, filters)
            ]
        if not candidates or k <= 0:
            return []

        query = list(vector)
        if HAS_NUMPY and self.metric in {"cosine", "dot"}:
            matrix = [c.embedding or [] for c in candidates]
            ranked = top_k(query, matrix, k, metric=self.metric)
            hits = [
                ScoredChunk(chunk=candidates[i], score=score, retriever=self.name)
                for i, score in ranked
            ]
        else:
            score_fn = self._scorer()
            hits = [
                ScoredChunk(chunk=c, score=score_fn(query, c.embedding or []), retriever=self.name)
                for c in candidates
            ]
            hits = self.rank(hits)[:k]
        return self.rank(hits)

    async def aget(self, ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id, skipping ids that are not present."""
        with self._lock:
            return [self.chunks[cid] for cid in ids if cid in self.chunks]

    async def acount(self) -> int:
        """Return how many chunks are stored."""
        with self._lock:
            return len(self.chunks)

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk.

        Used by hybrid retrieval to build a lexical index over the same corpus.

        Returns:
            A snapshot list; mutating it does not affect the store.
        """
        with self._lock:
            return list(self.chunks.values())

    # -- persistence ------------------------------------------------------
    async def apersist(self) -> None:
        """Write the collection to :attr:`persist_path`.

        Raises:
            ProviderError: When no path is configured or the write fails.
        """
        if not self.persist_path:
            raise ProviderError(
                "This store has no persist_path configured.",
                provider="memory",
                hint="Construct it with persist_path='./index' to enable saving.",
            )
        self.save(self._file())

    def save(self, path: str | Path) -> Path:
        """Write the collection to a JSON file.

        Args:
            path: Destination file. Parent directories are created.

        Returns:
            The path written.

        Raises:
            ProviderError: When the file cannot be written.

        Example:
            >>> import tempfile, pathlib
            >>> from windlass.core.types import Chunk
            >>> store = InMemoryVectorStore(dimensions=2)
            >>> _ = store.add([Chunk(content="x", embedding=[1.0, 0.0])])
            >>> out = store.save(pathlib.Path(tempfile.mkdtemp()) / "i.json")
            >>> InMemoryVectorStore().load(out).count()
            1
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "version": 1,
                "collection": self.collection,
                "dimensions": self.dimensions,
                "metric": self.metric,
                "chunks": [c.model_dump() for c in self.chunks.values()],
            }
        try:
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            raise ProviderError(
                f"Could not write the index to {target}: {exc}", provider="memory", original=exc
            ) from exc
        return target

    def load(self, path: str | Path) -> InMemoryVectorStore:
        """Load a collection previously written by :meth:`save`.

        Args:
            path: The JSON file to read.

        Returns:
            ``self``, so the call chains.

        Raises:
            ProviderError: When the file is missing or malformed.
        """
        source = Path(path)
        if not source.is_file():
            raise ProviderError(f"No index file at {source}.", provider="memory")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            chunks = [Chunk.model_validate(item) for item in payload.get("chunks", [])]
        except (OSError, ValueError) as exc:
            raise ProviderError(
                f"Could not read the index at {source}: {exc}", provider="memory", original=exc
            ) from exc

        with self._lock:
            self.chunks = {c.id: c for c in chunks}
            self.dimensions = payload.get("dimensions") or self.dimensions
            self.metric = payload.get("metric") or self.metric
            self.collection = payload.get("collection") or self.collection
        return self

    # -- helpers ----------------------------------------------------------
    def _file(self) -> Path:
        """Resolve :attr:`persist_path` to a concrete file path."""
        base = Path(self.persist_path or ".")
        if base.suffix == ".json":
            return base
        return base / f"{self.collection}.json"

    def _scorer(self) -> Any:
        """Return the similarity function for the configured metric."""
        if self.metric == "cosine":
            return cosine_similarity
        if self.metric == "dot":
            return dot

        def negative_distance(a: Sequence[float], b: Sequence[float]) -> float:
            return -euclidean_distance(a, b)

        return negative_distance

    def __repr__(self) -> str:
        return (
            f"InMemoryVectorStore(collection={self.collection!r}, "
            f"chunks={len(self.chunks)}, metric={self.metric!r})"
        )
