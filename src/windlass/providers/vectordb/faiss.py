"""FAISS vector store.

FAISS gives you approximate nearest-neighbour search over millions of vectors on
a single machine, with no server to run. Windlass keeps chunk payloads in a
side-table (FAISS stores vectors only) and maps between FAISS's integer ids and
Windlass chunk ids.

Install with::

    pip install "windlass[faiss]"

Example:
    >>> from windlass import Windlass                                    # doctest: +SKIP
    >>> store = Windlass.vectordb("faiss", dimensions=384,              # doctest: +SKIP
    ...                          persist_path="./index")               # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from windlass.core.concurrency import to_thread
from windlass.core.exceptions import ConfigurationError, ProviderError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.core.types import Chunk, ScoredChunk
from windlass.core.vectors import normalize
from windlass.interfaces.vectordb import MetadataFilter, VectorStore

__all__ = ["FaissVectorStore"]

#: How many extra candidates to pull when a metadata filter is active, since
#: filtering happens after the ANN search.
_FILTER_OVERFETCH = 10


@register.vectordb(
    "faiss",
    description="Local approximate nearest-neighbour search over millions of vectors.",
)
class FaissVectorStore(VectorStore):
    """Vector store backed by a FAISS index.

    Args:
        collection: Logical name, used for the persistence filenames.
        dimensions: Vector length. **Required** — FAISS allocates the index up
            front and cannot infer it.
        metric: ``cosine`` (inner product over normalised vectors), ``dot``
            (raw inner product) or ``euclidean``.
        index_type: ``"flat"`` for exact search, ``"ivf"`` for a coarse-quantised
            index (much faster on large corpora, needs training), or ``"hnsw"``
            for a graph index (fast, higher memory, no training).
        nlist: Number of IVF clusters. Ignored for other index types.
        nprobe: How many IVF clusters to scan at query time. Higher means better
            recall and slower search.
        persist_path: Directory to save the index and payload side-table to.
        **config: Forwarded to :class:`~windlass.interfaces.vectordb.VectorStore`.

    Raises:
        MissingDependencyError: When ``faiss-cpu`` is not installed.
        ConfigurationError: When ``dimensions`` is omitted.

    Performance:
        ``flat`` is exact and fast to about a million vectors. ``ivf`` needs at
        least ``39 * nlist`` vectors to train and is trained automatically on the
        first write that has enough data. Searching an untrained IVF index falls
        back to brute force rather than failing.
    """

    provider_name = "faiss"
    supports_filters = True

    def __init__(
        self,
        *,
        collection: str = "windlass",
        dimensions: int | None = None,
        metric: str = "cosine",
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        persist_path: str | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            collection=collection,
            dimensions=dimensions,
            metric=metric,
            persist_path=persist_path,
            **config,
        )
        if not dimensions:
            raise ConfigurationError(
                "FAISS needs to know the vector dimensionality up front.",
                hint="Pass dimensions=..., or let the RAG builder infer it from "
                "the embedding model.",
            )
        self._faiss = require("faiss", extra="faiss", feature="The FAISS vector store")
        self._np = require("numpy", extra="faiss", feature="The FAISS vector store")
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self._lock = threading.RLock()
        self._payloads: dict[int, Chunk] = {}
        self._by_chunk_id: dict[str, int] = {}
        self._next_id = 0
        self._index = self._build_index()

        if persist_path and self._index_file().exists():
            self.load(persist_path)

    # -- index construction -----------------------------------------------
    def _build_index(self) -> Any:
        """Create an empty FAISS index matching the configuration."""
        faiss, dim = self._faiss, int(self.dimensions or 0)
        if self.metric in {"cosine", "dot"}:
            base = faiss.IndexFlatIP(dim)
        else:
            base = faiss.IndexFlatL2(dim)

        if self.index_type == "flat":
            index = base
        elif self.index_type == "ivf":
            metric = (
                faiss.METRIC_INNER_PRODUCT if self.metric in {"cosine", "dot"} else faiss.METRIC_L2
            )
            index = faiss.IndexIVFFlat(base, dim, self.nlist, metric)
            index.nprobe = self.nprobe
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ConfigurationError(
                f"Unknown FAISS index_type {self.index_type!r}.",
                hint="Use 'flat', 'ivf' or 'hnsw'.",
            )
        return faiss.IndexIDMap2(index)

    def native(self) -> Any:
        """Return the underlying ``faiss.Index`` (Level 3 access)."""
        return self._index

    # -- writes -----------------------------------------------------------
    async def aadd(self, chunks: Sequence[Chunk]) -> int:
        """Insert or update chunks.

        Args:
            chunks: Chunks with embeddings set.

        Returns:
            How many chunks were written.

        Raises:
            ConfigurationError: If a chunk has no embedding.
            ProviderError: On dimension mismatch or a FAISS failure.
        """
        if not chunks:
            return 0
        self.validate_embeddings(chunks)
        return await to_thread(self._add_sync, list(chunks))

    def _add_sync(self, chunks: list[Chunk]) -> int:
        """Blocking write path, executed on a worker thread."""
        np, dim = self._np, int(self.dimensions or 0)
        with self._lock:
            vectors: list[list[float]] = []
            ids: list[int] = []
            replacements: list[int] = []

            for chunk in chunks:
                vector = list(chunk.embedding or [])
                if len(vector) != dim:
                    raise ProviderError(
                        f"Chunk {chunk.id!r} has {len(vector)} dimensions, index expects {dim}.",
                        provider="faiss",
                        hint="Every chunk in an index must come from the same embedding model.",
                    )
                if self.metric == "cosine":
                    vector = normalize(vector)

                existing = self._by_chunk_id.get(chunk.id)
                if existing is not None:
                    replacements.append(existing)
                    faiss_id = existing
                else:
                    faiss_id = self._next_id
                    self._next_id += 1
                    self._by_chunk_id[chunk.id] = faiss_id

                ids.append(faiss_id)
                vectors.append(vector)
                self._payloads[faiss_id] = chunk

            matrix = np.asarray(vectors, dtype="float32")
            if replacements:
                # Some index types cannot remove ids; those overwrite in place.
                with contextlib.suppress(RuntimeError):
                    self._index.remove_ids(np.asarray(replacements, dtype="int64"))

            self._train_if_needed(matrix)
            try:
                self._index.add_with_ids(matrix, np.asarray(ids, dtype="int64"))
            except Exception as exc:
                raise ProviderError(
                    f"FAISS rejected the batch: {exc}", provider="faiss", original=exc
                ) from exc
            return len(chunks)

    def _train_if_needed(self, matrix: Any) -> None:
        """Train an IVF index once enough vectors have accumulated."""
        index = self._index.index if hasattr(self._index, "index") else self._index
        if getattr(index, "is_trained", True):
            return
        required = max(self.nlist * 39, self.nlist)
        if len(matrix) >= required:
            index.train(matrix)
        else:
            self._log.debug(
                "IVF index needs %d vectors to train; have %d — using brute force until then.",
                required,
                len(matrix),
            )

    async def adelete(
        self, ids: Sequence[str] | None = None, *, filters: MetadataFilter | None = None
    ) -> int:
        """Delete by chunk id or metadata filter.

        Args:
            ids: Chunk ids to remove.
            filters: Metadata constraints.

        Returns:
            How many chunks were removed.

        Raises:
            ValueError: When neither argument is supplied.
        """
        if ids is None and filters is None:
            raise ValueError("Pass ids or filters; use clear() to empty the index.")
        return await to_thread(self._delete_sync, list(ids or ()), filters)

    def _delete_sync(self, ids: list[str], filters: MetadataFilter | None) -> int:
        """Blocking delete path."""
        with self._lock:
            targets = {self._by_chunk_id[i] for i in ids if i in self._by_chunk_id}
            if filters is not None:
                targets |= {
                    fid
                    for fid, chunk in self._payloads.items()
                    if self.match_filters(chunk.metadata, filters)
                }
            if not targets:
                return 0
            try:
                self._index.remove_ids(self._np.asarray(sorted(targets), dtype="int64"))
            except Exception as exc:
                raise ProviderError(
                    f"This FAISS index does not support deletion: {exc}",
                    provider="faiss",
                    hint="Rebuild the index instead, or use index_type='flat'.",
                    original=exc,
                ) from exc
            for fid in targets:
                chunk = self._payloads.pop(fid, None)
                if chunk is not None:
                    self._by_chunk_id.pop(chunk.id, None)
            return len(targets)

    async def aclear(self) -> None:
        """Drop the index and rebuild it empty."""
        with self._lock:
            self._index = self._build_index()
            self._payloads.clear()
            self._by_chunk_id.clear()
            self._next_id = 0

    # -- reads ------------------------------------------------------------
    async def asearch(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Search the index.

        Args:
            vector: The query embedding.
            k: How many results to return.
            filters: Metadata constraints, applied after the ANN search. Windlass
                over-fetches by a factor of :data:`_FILTER_OVERFETCH` so a
                selective filter still returns ``k`` hits.
            **kwargs: Ignored.

        Returns:
            Ranked hits.
        """
        return await to_thread(self._search_sync, list(vector), k, filters)

    def _search_sync(
        self, vector: list[float], k: int, filters: MetadataFilter | None
    ) -> list[ScoredChunk]:
        """Blocking search path."""
        with self._lock:
            if self._index.ntotal == 0 or k <= 0:
                return []
            query = normalize(vector) if self.metric == "cosine" else vector
            fetch = min(self._index.ntotal, k * (_FILTER_OVERFETCH if filters else 1))
            matrix = self._np.asarray([query], dtype="float32")
            scores, ids = self._index.search(matrix, fetch)

            hits: list[ScoredChunk] = []
            for score, faiss_id in zip(scores[0], ids[0], strict=True):
                if faiss_id < 0:
                    continue
                chunk = self._payloads.get(int(faiss_id))
                if chunk is None:
                    continue
                if not self.match_filters(chunk.metadata, filters):
                    continue
                # L2 returns distances: negate so higher is always better.
                value = -float(score) if self.metric in {"euclidean", "l2"} else float(score)
                hits.append(ScoredChunk(chunk=chunk, score=value, retriever=self.name))
                if len(hits) >= k:
                    break
            return self.rank(hits)

    async def aget(self, ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id."""
        with self._lock:
            return [self._payloads[self._by_chunk_id[i]] for i in ids if i in self._by_chunk_id]

    async def acount(self) -> int:
        """Return how many vectors the index holds."""
        with self._lock:
            return int(self._index.ntotal)

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk, for building a companion lexical index."""
        with self._lock:
            return list(self._payloads.values())

    # -- persistence ------------------------------------------------------
    async def apersist(self) -> None:
        """Save the index and payloads to :attr:`persist_path`."""
        if not self.persist_path:
            raise ProviderError(
                "This store has no persist_path configured.",
                provider="faiss",
                hint="Construct it with persist_path='./index'.",
            )
        await to_thread(self.save, self.persist_path)

    def save(self, directory: str | Path) -> Path:
        """Write the FAISS index and its payload side-table to ``directory``.

        Args:
            directory: Destination directory, created if absent.

        Returns:
            The directory written to.

        Raises:
            ProviderError: When the write fails.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                self._faiss.write_index(self._index, str(target / f"{self.collection}.faiss"))
                payload = {
                    "version": 1,
                    "dimensions": self.dimensions,
                    "metric": self.metric,
                    "index_type": self.index_type,
                    "next_id": self._next_id,
                    "payloads": {
                        str(fid): chunk.model_dump() for fid, chunk in self._payloads.items()
                    },
                }
                (target / f"{self.collection}.meta.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as exc:
                raise ProviderError(
                    f"Could not save the FAISS index to {target}: {exc}",
                    provider="faiss",
                    original=exc,
                ) from exc
        return target

    def load(self, directory: str | Path) -> FaissVectorStore:
        """Load an index previously written by :meth:`save`.

        Args:
            directory: Directory containing the index files.

        Returns:
            ``self``.

        Raises:
            ProviderError: When the files are missing or unreadable.
        """
        target = Path(directory)
        index_file = target / f"{self.collection}.faiss"
        meta_file = target / f"{self.collection}.meta.json"
        if not index_file.is_file() or not meta_file.is_file():
            raise ProviderError(
                f"No saved FAISS index for {self.collection!r} in {target}.", provider="faiss"
            )
        try:
            with self._lock:
                self._index = self._faiss.read_index(str(index_file))
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                self._payloads = {
                    int(fid): Chunk.model_validate(data)
                    for fid, data in meta.get("payloads", {}).items()
                }
                self._by_chunk_id = {c.id: fid for fid, c in self._payloads.items()}
                self._next_id = int(meta.get("next_id", len(self._payloads)))
                self.dimensions = meta.get("dimensions", self.dimensions)
                self.metric = meta.get("metric", self.metric)
        except Exception as exc:
            raise ProviderError(
                f"Could not load the FAISS index from {target}: {exc}",
                provider="faiss",
                original=exc,
            ) from exc
        return self

    def _index_file(self) -> Path:
        """Path of the saved FAISS index for this collection."""
        return Path(self.persist_path or ".") / f"{self.collection}.faiss"

    def __repr__(self) -> str:
        return (
            f"FaissVectorStore(collection={self.collection!r}, "
            f"type={self.index_type!r}, vectors={self._index.ntotal})"
        )
