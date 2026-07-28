"""ChromaDB vector store.

Chroma is the pragmatic middle ground: persistent on disk with no server to run,
or client/server when you need it, with native metadata filtering. Windlass uses
Chroma purely as storage — embeddings are always computed by the configured
Windlass embedder, so switching stores never silently switches embedding models.

Install with::

    pip install "windlass[chroma]"

Example:
    >>> from windlass import Windlass                                        # doctest: +SKIP
    >>> store = Windlass.vectordb("chroma", persist_path="./chroma")        # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

from windlass.core.concurrency import to_thread
from windlass.core.exceptions import ProviderError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.core.types import Chunk, ScoredChunk
from windlass.interfaces.vectordb import MetadataFilter, VectorStore

__all__ = ["ChromaVectorStore"]

#: Metadata values Chroma accepts natively; anything else is JSON-encoded.
_SCALARS = (str, int, float, bool)


@register.vectordb(
    "chroma",
    aliases=("chromadb",),
    description="ChromaDB, embedded or client/server, with native metadata filtering.",
)
class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB.

    Args:
        collection: Chroma collection name.
        dimensions: Vector length. Chroma infers it, so this is informational.
        metric: ``cosine``, ``dot`` or ``euclidean``. Set at collection creation
            and immutable afterwards.
        persist_path: Directory for the embedded persistent client. When omitted
            an ephemeral in-memory client is used.
        host: Server host for client/server mode. Takes precedence over
            ``persist_path``.
        port: Server port.
        client: Pass a pre-configured ``chromadb`` client to use instead.
        **config: Forwarded to :class:`~windlass.interfaces.vectordb.VectorStore`.

    Raises:
        MissingDependencyError: When ``chromadb`` is not installed.
        ProviderError: When the client or collection cannot be created.

    Note:
        Chroma stores metadata values as scalars only. Windlass JSON-encodes
        lists and dicts on write and decodes them on read, so nested metadata
        round-trips correctly.
    """

    provider_name = "chroma"
    supports_filters = True

    def __init__(
        self,
        *,
        collection: str = "windlass",
        dimensions: int | None = None,
        metric: str = "cosine",
        persist_path: str | None = None,
        host: str | None = None,
        port: int = 8000,
        client: Any = None,
        **config: Any,
    ) -> None:
        super().__init__(
            collection=collection,
            dimensions=dimensions,
            metric=metric,
            persist_path=persist_path,
            **config,
        )
        chromadb = require("chromadb", extra="chroma", feature="The ChromaDB vector store")
        self._chromadb = chromadb
        try:
            if client is not None:
                self._client = client
            elif host:
                self._client = chromadb.HttpClient(host=host, port=port)
            elif persist_path:
                self._client = chromadb.PersistentClient(path=str(persist_path))
            else:
                self._client = chromadb.EphemeralClient()

            self._collection = self._client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": _space(metric)},
            )
        except Exception as exc:
            raise ProviderError(
                f"Could not open the Chroma collection {collection!r}: {exc}",
                provider="chroma",
                hint="Check the persist_path is writable, or that the server is reachable.",
                original=exc,
            ) from exc

    def native(self) -> Any:
        """Return the underlying ``chromadb`` collection (Level 3 access)."""
        return self._collection

    # -- writes -----------------------------------------------------------
    async def aadd(self, chunks: Sequence[Chunk]) -> int:
        """Upsert chunks into the collection.

        Args:
            chunks: Chunks with embeddings set.

        Returns:
            How many chunks were written.

        Raises:
            ConfigurationError: If a chunk has no embedding.
            ProviderError: When Chroma rejects the batch.
        """
        if not chunks:
            return 0
        self.validate_embeddings(chunks)

        def _upsert() -> int:
            self._collection.upsert(
                ids=[c.id for c in chunks],
                embeddings=[list(c.embedding or []) for c in chunks],
                documents=[c.content for c in chunks],
                metadatas=[_encode_metadata(c) for c in chunks],
            )
            return len(chunks)

        try:
            return await to_thread(_upsert)
        except Exception as exc:
            raise ProviderError(
                f"Chroma rejected a batch of {len(chunks)} chunks: {exc}",
                provider="chroma",
                original=exc,
            ) from exc

    async def adelete(
        self, ids: Sequence[str] | None = None, *, filters: MetadataFilter | None = None
    ) -> int:
        """Delete by id or metadata filter.

        Args:
            ids: Chunk ids to remove.
            filters: Metadata constraints, translated to Chroma's ``where`` syntax.

        Returns:
            How many chunks were removed.

        Raises:
            ValueError: When neither argument is supplied.
        """
        if ids is None and filters is None:
            raise ValueError("Pass ids or filters; use clear() to empty the collection.")

        def _delete() -> int:
            before = self._collection.count()
            kwargs: dict[str, Any] = {}
            if ids:
                kwargs["ids"] = list(ids)
            if filters:
                kwargs["where"] = _to_where(filters)
            self._collection.delete(**kwargs)
            return max(0, before - self._collection.count())

        return await to_thread(_delete)

    async def aclear(self) -> None:
        """Delete and recreate the collection."""

        def _reset() -> None:
            self._client.delete_collection(self.collection)
            self._collection = self._client.get_or_create_collection(
                name=self.collection, metadata={"hnsw:space": _space(self.metric)}
            )

        await to_thread(_reset)

    # -- reads ------------------------------------------------------------
    async def asearch(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Query the collection.

        Args:
            vector: The query embedding.
            k: How many results to return.
            filters: Metadata constraints, pushed down to Chroma.
            **kwargs: Extra keyword arguments for ``collection.query``.

        Returns:
            Ranked hits. Chroma returns distances; they are converted to
            similarity so higher is always better.
        """

        def _query() -> dict[str, Any]:
            payload: dict[str, Any] = {
                "query_embeddings": [list(vector)],
                "n_results": max(1, k),
                "include": ["documents", "metadatas", "distances"],
            }
            if filters:
                payload["where"] = _to_where(filters)
            payload.update(kwargs)
            return self._collection.query(**payload)

        try:
            result = await to_thread(_query)
        except Exception as exc:
            raise ProviderError(
                f"Chroma query failed: {exc}", provider="chroma", original=exc
            ) from exc

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[ScoredChunk] = []
        for cid, text, meta, distance in zip(ids, documents, metadatas, distances, strict=False):
            chunk = _decode_chunk(cid, text or "", meta or {})
            hits.append(
                ScoredChunk(
                    chunk=chunk,
                    score=_to_similarity(float(distance), self.metric),
                    retriever=self.name,
                )
            )
        return self.rank(hits)

    async def aget(self, ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id."""
        if not ids:
            return []

        def _get() -> dict[str, Any]:
            return self._collection.get(ids=list(ids), include=["documents", "metadatas"])

        result = await to_thread(_get)
        return [
            _decode_chunk(cid, text or "", meta or {})
            for cid, text, meta in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
                strict=False,
            )
        ]

    async def acount(self) -> int:
        """Return how many chunks the collection holds."""
        return int(await to_thread(self._collection.count))

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk, for building a companion lexical index.

        Note:
            Loads the whole collection into memory. Fine for the tens of
            thousands of chunks a hybrid retriever needs; not something to call
            on a multi-million-row collection.
        """
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            _decode_chunk(cid, text or "", meta or {})
            for cid, text, meta in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
                strict=False,
            )
        ]

    def __repr__(self) -> str:
        return f"ChromaVectorStore(collection={self.collection!r})"


def _space(metric: str) -> str:
    """Map a Windlass metric onto Chroma's ``hnsw:space`` value."""
    return {"cosine": "cosine", "dot": "ip", "euclidean": "l2", "l2": "l2"}.get(metric, "cosine")


def _to_similarity(distance: float, metric: str) -> float:
    """Convert a Chroma distance into a score where higher is better."""
    if metric == "cosine":
        return 1.0 - distance
    if metric == "dot":
        return -distance
    return -distance


def _encode_metadata(chunk: Chunk) -> dict[str, Any]:
    """Flatten chunk metadata into Chroma-compatible scalars."""
    import json

    encoded: dict[str, Any] = {}
    for key, value in chunk.metadata.items():
        if value is None:
            continue
        encoded[key] = value if isinstance(value, _SCALARS) else json.dumps(value, default=str)
    encoded["_document_id"] = chunk.document_id
    encoded["_index"] = chunk.index
    if chunk.parent_id:
        encoded["_parent_id"] = chunk.parent_id
    return encoded


def _decode_chunk(chunk_id: str, content: str, metadata: dict[str, Any]) -> Chunk:
    """Rebuild a :class:`Chunk` from a Chroma row."""
    import json

    meta = dict(metadata)
    document_id = str(meta.pop("_document_id", ""))
    index = int(meta.pop("_index", 0) or 0)
    parent_id = meta.pop("_parent_id", None)

    for key, value in list(meta.items()):
        if isinstance(value, str) and value[:1] in "[{":
            # Nested metadata was JSON-encoded on write; a value that merely
            # looks like JSON is left exactly as it is.
            with contextlib.suppress(ValueError):
                meta[key] = json.loads(value)

    return Chunk(
        id=chunk_id,
        content=content,
        metadata=meta,
        document_id=document_id,
        index=index,
        parent_id=parent_id,
    )


def _to_where(filters: MetadataFilter) -> dict[str, Any]:
    """Translate a Windlass metadata filter into Chroma's ``where`` syntax.

    Chroma uses the same ``$``-prefixed operator names for the subset it
    supports, so most filters pass through. ``$contains`` and ``$exists`` have no
    Chroma equivalent and are dropped here — the base class re-applies them
    client-side via :meth:`VectorStore.match_filters`.

    Args:
        filters: The Windlass filter.

    Returns:
        A Chroma ``where`` clause. Multiple conditions are wrapped in ``$and``.

    Example:
        >>> _to_where({"year": 2024})
        {'year': {'$eq': 2024}}
    """
    supported = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
    clauses: list[dict[str, Any]] = []
    for key, condition in filters.items():
        if isinstance(condition, dict):
            usable = {op: val for op, val in condition.items() if op in supported}
            if usable:
                clauses.append({key: usable})
        else:
            clauses.append({key: {"$eq": condition}})
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
