"""Pinecone vector store.

Pinecone is the managed option: no index to host, scales past what a single
machine holds, and supports namespaces for multi-tenant isolation. The trade-off
is network latency on every query and eventual consistency on writes — a chunk
you just upserted may not appear in the very next search.

Install with::

    pip install "windlass[pinecone]"

Example:
    >>> from windlass import Windlass                                          # doctest: +SKIP
    >>> store = Windlass.vectordb("pinecone", collection="docs",              # doctest: +SKIP
    ...                          dimensions=1536)                            # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from typing import Any

from windlass.core.concurrency import to_thread
from windlass.core.config import settings
from windlass.core.exceptions import ConfigurationError, ProviderError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.core.types import Chunk, ScoredChunk
from windlass.interfaces.vectordb import MetadataFilter, VectorStore

__all__ = ["PineconeVectorStore"]

#: Pinecone caps metadata at 40 KB per vector; keep well under it.
_MAX_METADATA_BYTES = 30_000

#: Upsert batch size Pinecone's API is comfortable with.
_UPSERT_BATCH = 100


@register.vectordb(
    "pinecone",
    description="Managed, distributed vector search with namespace isolation.",
)
class PineconeVectorStore(VectorStore):
    """Vector store backed by a Pinecone serverless index.

    Args:
        collection: Pinecone index name.
        dimensions: Vector length. **Required** when the index must be created.
        metric: ``cosine``, ``dot`` or ``euclidean``.
        api_key: Credential. Falls back to ``PINECONE_API_KEY``.
        namespace: Namespace within the index — the clean way to isolate tenants
            while sharing one index.
        cloud: Serverless cloud provider for index creation.
        region: Serverless region for index creation.
        create_if_missing: Create the index when it does not exist. Creation is
            not instant; the first write may wait for it to become ready.
        **config: Forwarded to :class:`~windlass.interfaces.vectordb.VectorStore`.

    Raises:
        MissingDependencyError: When ``pinecone`` is not installed.
        ConfigurationError: When the API key or dimensions are missing.
        ProviderError: When the index cannot be reached or created.
    """

    provider_name = "pinecone"
    supports_filters = True

    def __init__(
        self,
        *,
        collection: str = "windlass",
        dimensions: int | None = None,
        metric: str = "cosine",
        api_key: str | None = None,
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
        create_if_missing: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(collection=collection, dimensions=dimensions, metric=metric, **config)
        sdk = require("pinecone", extra="pinecone", feature="The Pinecone vector store")
        key = api_key or settings().secret("pinecone_api_key")
        if not key:
            raise ConfigurationError(
                "No API key configured for Pinecone.",
                hint="Set PINECONE_API_KEY, or pass "
                "Windlass.vectordb('pinecone', api_key='...').",
            )
        self.namespace = namespace
        self._sdk = sdk
        try:
            self._client = sdk.Pinecone(api_key=key)
            existing = {idx["name"] for idx in self._client.list_indexes()}
            if collection not in existing:
                if not create_if_missing:
                    raise ConfigurationError(
                        f"Pinecone index {collection!r} does not exist.",
                        hint="Create it in the console, or pass create_if_missing=True.",
                    )
                if not dimensions:
                    raise ConfigurationError(
                        "Creating a Pinecone index requires dimensions=...",
                        hint="Pass the embedding model's dimensionality.",
                    )
                self._client.create_index(
                    name=collection,
                    dimension=int(dimensions),
                    metric=_metric_name(metric),
                    spec=sdk.ServerlessSpec(cloud=cloud, region=region),
                )
            self._index = self._client.Index(collection)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Could not open the Pinecone index {collection!r}: {exc}",
                provider="pinecone",
                original=exc,
            ) from exc

    def native(self) -> Any:
        """Return the underlying Pinecone ``Index`` handle (Level 3 access)."""
        return self._index

    # -- writes -----------------------------------------------------------
    async def aadd(self, chunks: Sequence[Chunk]) -> int:
        """Upsert chunks in batches.

        Args:
            chunks: Chunks with embeddings set.

        Returns:
            How many chunks were sent.

        Raises:
            ConfigurationError: If a chunk has no embedding.
            ProviderError: When the API rejects a batch.

        Note:
            Pinecone is eventually consistent. A chunk written here may take a
            moment to become visible to :meth:`asearch`.
        """
        if not chunks:
            return 0
        self.validate_embeddings(chunks)

        vectors = [
            {
                "id": chunk.id,
                "values": list(chunk.embedding or []),
                "metadata": _encode_metadata(chunk),
            }
            for chunk in chunks
        ]

        def _upsert() -> int:
            for start in range(0, len(vectors), _UPSERT_BATCH):
                batch = vectors[start : start + _UPSERT_BATCH]
                self._index.upsert(vectors=batch, namespace=self.namespace or None)
            return len(vectors)

        try:
            return await to_thread(_upsert)
        except Exception as exc:
            raise ProviderError(
                f"Pinecone rejected a batch of {len(vectors)} vectors: {exc}",
                provider="pinecone",
                original=exc,
            ) from exc

    async def adelete(
        self, ids: Sequence[str] | None = None, *, filters: MetadataFilter | None = None
    ) -> int:
        """Delete by id or metadata filter.

        Args:
            ids: Chunk ids to remove.
            filters: Metadata constraints, translated to Pinecone's filter syntax.

        Returns:
            The number of ids requested. Pinecone does not report how many rows
            a filtered delete actually matched, so filtered deletes return 0.

        Raises:
            ValueError: When neither argument is supplied.
        """
        if ids is None and filters is None:
            raise ValueError("Pass ids or filters; use clear() to empty the namespace.")

        def _delete() -> int:
            if ids:
                self._index.delete(ids=list(ids), namespace=self.namespace or None)
                return len(list(ids))
            self._index.delete(
                filter=_to_pinecone_filter(filters or {}), namespace=self.namespace or None
            )
            return 0

        try:
            return await to_thread(_delete)
        except Exception as exc:
            raise ProviderError(
                f"Pinecone delete failed: {exc}", provider="pinecone", original=exc
            ) from exc

    async def aclear(self) -> None:
        """Delete every vector in the namespace.

        Clearing a namespace that does not exist yet succeeds. Pinecone creates
        namespaces lazily on first write and returns 404 for a delete-all
        against one that has never been written to, but "make this empty" is
        already satisfied in that case — raising would make teardown code fail
        precisely when there is nothing to tear down.

        Raises:
            ProviderError: For any failure other than an absent namespace.
        """

        def _clear() -> None:
            try:
                self._index.delete(delete_all=True, namespace=self.namespace or None)
            except Exception as exc:
                if _is_missing_namespace(exc):
                    self._log.debug(
                        "Namespace %r does not exist; nothing to clear.", self.namespace
                    )
                    return
                raise ProviderError(
                    f"Pinecone clear failed: {exc}", provider="pinecone", original=exc
                ) from exc

        await to_thread(_clear)

    # -- reads ------------------------------------------------------------
    async def asearch(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Query the index.

        Args:
            vector: The query embedding.
            k: How many results to return.
            filters: Metadata constraints, pushed down to Pinecone.
            **kwargs: Extra keyword arguments for ``index.query``.

        Returns:
            Ranked hits.
        """

        def _query() -> Any:
            payload: dict[str, Any] = {
                "vector": list(vector),
                "top_k": max(1, k),
                "include_metadata": True,
                "namespace": self.namespace or None,
            }
            if filters:
                payload["filter"] = _to_pinecone_filter(filters)
            payload.update(kwargs)
            return self._index.query(**payload)

        try:
            response = await to_thread(_query)
        except Exception as exc:
            raise ProviderError(
                f"Pinecone query failed: {exc}", provider="pinecone", original=exc
            ) from exc

        hits = [
            ScoredChunk(
                chunk=_decode_chunk(match["id"], match.get("metadata") or {}),
                score=float(match.get("score", 0.0)),
                retriever=self.name,
            )
            for match in (response.get("matches") or [])
        ]
        return self.rank(hits)

    async def aget(self, ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id."""
        if not ids:
            return []

        def _fetch() -> Any:
            return self._index.fetch(ids=list(ids), namespace=self.namespace or None)

        response = await to_thread(_fetch)
        vectors = response.get("vectors") or {}
        return [
            _decode_chunk(cid, payload.get("metadata") or {}) for cid, payload in vectors.items()
        ]

    async def acount(self) -> int:
        """Return the namespace's vector count.

        Note:
            Pinecone's statistics lag writes by a few seconds.
        """

        def _stats() -> int:
            stats = self._index.describe_index_stats()
            if self.namespace:
                namespaces = stats.get("namespaces") or {}
                return int((namespaces.get(self.namespace) or {}).get("vector_count", 0))
            return int(stats.get("total_vector_count", 0))

        return await to_thread(_stats)

    def __repr__(self) -> str:
        return f"PineconeVectorStore(index={self.collection!r}, " f"namespace={self.namespace!r})"


def _is_missing_namespace(exc: BaseException) -> bool:
    """Return whether an exception means "that namespace does not exist".

    Matched on status code and message rather than on an exception class,
    because the class moved between Pinecone SDK generations
    (``NotFoundException`` -> ``NotFoundError``) and pinning to either would
    break on the other.

    Args:
        exc: The exception raised by the Pinecone client.

    Returns:
        True when the failure is an absent namespace.

    Example:
        >>> _is_missing_namespace(RuntimeError("[404] Namespace not found"))
        True
        >>> _is_missing_namespace(RuntimeError("[500] Internal error"))
        False
    """
    if getattr(exc, "status_code", None) not in (404, None):
        return False
    message = str(exc).lower()
    return "namespace" in message and ("not found" in message or "does not exist" in message)


def _metric_name(metric: str) -> str:
    """Map a Windlass metric onto Pinecone's name for it."""
    return {
        "cosine": "cosine",
        "dot": "dotproduct",
        "euclidean": "euclidean",
        "l2": "euclidean",
    }.get(metric, "cosine")


def _encode_metadata(chunk: Chunk) -> dict[str, Any]:
    """Pack chunk content and metadata into Pinecone metadata.

    Pinecone has no document payload, so the chunk text rides along in metadata.
    Oversized text is truncated to stay under the per-vector limit; the full text
    remains in your source of record.
    """
    meta: dict[str, Any] = {
        "_content": chunk.content,
        "_document_id": chunk.document_id,
        "_index": chunk.index,
    }
    if chunk.parent_id:
        meta["_parent_id"] = chunk.parent_id
    for key, value in chunk.metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool) or (
            isinstance(value, list) and all(isinstance(v, str) for v in value)
        ):
            meta[key] = value
        else:
            meta[key] = json.dumps(value, default=str)

    encoded = json.dumps(meta, default=str)
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        budget = _MAX_METADATA_BYTES - (len(encoded.encode("utf-8")) - len(chunk.content))
        meta["_content"] = chunk.content[: max(0, budget)]
        meta["_truncated"] = True
    return meta


def _decode_chunk(chunk_id: str, metadata: dict[str, Any]) -> Chunk:
    """Rebuild a :class:`Chunk` from Pinecone metadata."""
    meta = dict(metadata)
    content = str(meta.pop("_content", ""))
    document_id = str(meta.pop("_document_id", ""))
    index = int(meta.pop("_index", 0) or 0)
    parent_id = meta.pop("_parent_id", None)
    meta.pop("_truncated", None)

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


def _to_pinecone_filter(filters: MetadataFilter) -> dict[str, Any]:
    """Translate a Windlass metadata filter into Pinecone's filter syntax.

    Pinecone supports ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``,
    ``$in`` and ``$nin`` with the same names, so most filters pass straight
    through. Unsupported operators are dropped and re-applied client-side.

    Args:
        filters: The Windlass filter.

    Returns:
        A Pinecone filter document.

    Example:
        >>> _to_pinecone_filter({"tenant": "acme"})
        {'tenant': {'$eq': 'acme'}}
    """
    supported = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
    out: dict[str, Any] = {}
    for key, condition in filters.items():
        if isinstance(condition, dict):
            usable = {op: val for op, val in condition.items() if op in supported}
            if usable:
                out[key] = usable
        else:
            out[key] = {"$eq": condition}
    return out
