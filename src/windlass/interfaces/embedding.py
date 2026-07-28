"""The embedding-model interface.

An embedding provider turns text into dense vectors. Windlass distinguishes
*document* embeddings from *query* embeddings because several modern models
(E5, BGE, Nomic) require different instruction prefixes for each — getting that
wrong quietly halves retrieval quality, so the interface makes it explicit.

Implementers override one coroutine, :meth:`Embedder.aembed_texts`.

Example:
    >>> from windlass.providers.embeddings.hash import HashEmbedder
    >>> vectors = HashEmbedder(dimensions=8).embed(["hello", "world"])
    >>> len(vectors), len(vectors[0])
    (2, 8)
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from windlass.core.cache import Cache, NullCache, make_key
from windlass.core.concurrency import batched, gather_bounded, run_sync
from windlass.core.config import settings
from windlass.core.retry import retry_async
from windlass.core.vectors import normalize
from windlass.interfaces.base import Component

__all__ = ["Embedder"]


class Embedder(Component):
    """Abstract text-embedding model.

    Args:
        model: Model identifier passed to the provider.
        dimensions: Output dimensionality. Providers that expose a native value
            should override :meth:`dimension`.
        batch_size: How many texts to send per provider request.
        normalize: L2-normalise every vector. Recommended: it makes cosine
            similarity a plain dot product and matches what FAISS expects.
        cache: Optional cache for embeddings. Embeddings are deterministic, so
            caching them is always safe.
        name: Component name for traces.
        **config: Provider-specific options.

    Attributes:
        model: Configured model identifier.
        batch_size: Configured request batch size.
        normalize: Whether vectors are normalised on the way out.

    Example:
        Implementing a provider takes one method::

            class MyEmbedder(Embedder):
                provider_name = "mine"

                async def aembed_texts(self, texts, *, kind="document"):
                    return [await my_sdk.embed(t) for t in texts]
    """

    kind = "embedding"
    provider_name: str = "embedding"

    #: Some models want an instruction prefix on queries but not documents.
    query_prefix: str = ""
    document_prefix: str = ""

    def __init__(
        self,
        model: str = "",
        *,
        dimensions: int | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
        cache: Cache | None = None,
        name: str | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            name=name or self.provider_name,
            model=model,
            dimensions=dimensions,
            batch_size=batch_size or settings().batch_size,
            normalize=normalize,
            **config,
        )
        self.model = model or self.default_model()
        self.batch_size: int = self.config["batch_size"]
        self.normalize: bool = normalize
        self._dimensions = dimensions
        # `cache or NullCache()` looks equivalent and is not: every Cache
        # implements __len__, so a freshly constructed (empty) cache is falsy
        # and would be silently discarded — caching would appear to be
        # configured and never happen. Test for None explicitly.
        self._cache: Cache = NullCache() if cache is None else cache

    # -- provider hooks ---------------------------------------------------
    @classmethod
    def default_model(cls) -> str:
        """Return the model used when the caller does not name one."""
        return ""

    @abc.abstractmethod
    async def aembed_texts(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        """Embed a batch of texts.

        This is the only method a provider must implement. It receives at most
        :attr:`batch_size` texts, already prefixed for ``kind``.

        Args:
            texts: The texts to embed. Never empty.
            kind: ``"document"`` or ``"query"``.

        Returns:
            One vector per input, in the same order.

        Raises:
            ProviderError: For any provider-side failure.
        """

    def dimension(self) -> int:
        """Return the vector dimensionality.

        Determined from configuration when possible, otherwise by embedding a
        one-character probe and measuring the result. The probe runs at most
        once per instance.

        Returns:
            The number of components in each vector.

        Example:
            >>> from windlass.providers.embeddings.hash import HashEmbedder
            >>> HashEmbedder(dimensions=64).dimension()
            64
        """
        if self._dimensions is None:
            self._dimensions = len(self.embed_one("dimension probe"))
        return self._dimensions

    # -- public API -------------------------------------------------------
    async def aembed(
        self,
        texts: Sequence[str],
        *,
        kind: str = "document",
        concurrency: int | None = None,
    ) -> list[list[float]]:
        """Embed many texts with batching, bounded concurrency and caching.

        Args:
            texts: The texts to embed.
            kind: ``"document"`` for corpus text, ``"query"`` for search input.
            concurrency: Maximum simultaneous provider requests. Defaults to the
                global ``max_concurrency`` setting.

        Returns:
            One vector per input, in input order.

        Raises:
            ProviderError: For any provider-side failure.
            ValueError: If ``kind`` is not ``document`` or ``query``.

        Performance:
            Texts are grouped into :attr:`batch_size` batches and the batches
            run concurrently up to ``concurrency``. Cached texts never reach the
            provider, so re-ingesting a mostly-unchanged corpus is cheap.

        Example:
            >>> import asyncio
            >>> from windlass.providers.embeddings.hash import HashEmbedder
            >>> len(asyncio.run(HashEmbedder(dimensions=8).aembed(["a", "b"])))
            2
        """
        if kind not in {"document", "query"}:
            raise ValueError(f"kind must be 'document' or 'query', got {kind!r}")
        if not texts:
            return []

        prefix = self.query_prefix if kind == "query" else self.document_prefix
        prepared = [f"{prefix}{t}" if prefix else t for t in texts]

        # Split into cache hits and the misses we actually have to compute.
        results: list[list[float] | None] = [None] * len(prepared)
        pending: list[tuple[int, str]] = []
        for i, text in enumerate(prepared):
            hit = await self._cache.aget(self._key(text, kind))
            if hit is not None:
                results[i] = hit
            else:
                pending.append((i, text))

        if pending:
            batches = list(batched(pending, self.batch_size))
            limit = concurrency or settings().max_concurrency
            computed = await gather_bounded(
                [self._embed_batch([t for _, t in batch], kind) for batch in batches],
                limit=limit,
            )
            for batch, vectors in zip(batches, computed, strict=True):
                for (index, text), vector in zip(batch, vectors, strict=True):
                    final = normalize(vector) if self.normalize else list(vector)
                    results[index] = final
                    await self._cache.aset(self._key(text, kind), final)

        return [r for r in results if r is not None]

    async def _embed_batch(self, texts: list[str], kind: str) -> list[list[float]]:
        """Embed one batch with the retry policy applied."""
        return await retry_async(
            lambda: self.aembed_texts(texts, kind=kind),
            description=f"{self.name}.embed",
        )

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        """Blocking :meth:`aembed`."""
        return run_sync(self.aembed(texts, kind=kind))

    async def aembed_one(self, text: str, *, kind: str = "document") -> list[float]:
        """Embed a single text.

        Args:
            text: The text to embed.
            kind: ``"document"`` or ``"query"``.

        Returns:
            One vector.
        """
        vectors = await self.aembed([text], kind=kind)
        return vectors[0]

    def embed_one(self, text: str, *, kind: str = "document") -> list[float]:
        """Blocking :meth:`aembed_one`."""
        return run_sync(self.aembed_one(text, kind=kind))

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a search query, applying any query instruction prefix."""
        return await self.aembed_one(text, kind="query")

    def embed_query(self, text: str) -> list[float]:
        """Blocking :meth:`aembed_query`."""
        return run_sync(self.aembed_query(text))

    # -- helpers ----------------------------------------------------------
    def _key(self, text: str, kind: str) -> str:
        """Build the cache key for one text."""
        return make_key(self.provider_name, self.model, kind, text, namespace="emb")

    def set_cache(self, cache: Cache) -> None:
        """Attach a cache to this embedder.

        Args:
            cache: Any :class:`~windlass.core.cache.Cache` implementation.
        """
        self._cache = cache

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, dim={self._dimensions})"
