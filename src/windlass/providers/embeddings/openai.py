"""OpenAI embedding models.

Works with OpenAI and any OpenAI-compatible embedding endpoint. Install with::

    pip install "windlass[openai]"

Example:
    >>> from windlass import Windlass                                  # doctest: +SKIP
    >>> emb = Windlass.embedding("openai", model="text-embedding-3-small")  # doctest: +SKIP
    >>> len(emb.embed_one("hello"))                                  # doctest: +SKIP
    1536
"""

from __future__ import annotations

from typing import Any

from windlass.core.config import settings
from windlass.core.exceptions import AuthenticationError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.interfaces.embedding import Embedder

__all__ = ["OpenAIEmbedder"]

#: Native dimensionality per model, so ``dimension()`` never needs a probe call.
_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@register.embedding(
    "openai",
    description="OpenAI and OpenAI-compatible embedding models.",
)
class OpenAIEmbedder(Embedder):
    """Embeddings via the official ``openai`` SDK.

    Args:
        model: Model id, e.g. ``text-embedding-3-small``.
        api_key: Credential. Falls back to ``OPENAI_API_KEY``.
        base_url: Endpoint override for compatible gateways.
        dimensions: Request a reduced dimensionality. Supported by the
            ``text-embedding-3-*`` family via Matryoshka truncation — 512 dims
            keeps most of the quality at a third of the storage.
        **config: Forwarded to :class:`~windlass.interfaces.embedding.Embedder`.

    Raises:
        MissingDependencyError: When ``openai`` is not installed.
        AuthenticationError: When no API key can be found.
    """

    provider_name = "openai"

    def __init__(
        self,
        model: str = "",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        **config: Any,
    ) -> None:
        super().__init__(model=model, dimensions=dimensions, **config)
        self._sdk = require("openai", extra="openai", feature="The OpenAI embedding provider")
        cfg = settings()
        key = api_key or cfg.secret("openai_api_key")
        url = base_url or cfg.openai_base_url
        if not key and not url:
            raise AuthenticationError(
                "No API key configured for OpenAI embeddings.",
                provider="openai",
                hint="Set OPENAI_API_KEY, or pass "
                "Windlass.embedding('openai', api_key='sk-...').",
            )
        self._client = self._sdk.AsyncOpenAI(
            api_key=key or "not-needed", base_url=url, timeout=cfg.request_timeout, max_retries=0
        )
        self._requested_dimensions = dimensions
        if self._dimensions is None:
            self._dimensions = _DIMENSIONS.get(self.model)

    @classmethod
    def default_model(cls) -> str:
        """Return ``"text-embedding-3-small"``."""
        return "text-embedding-3-small"

    def native(self) -> Any:
        """Return the underlying ``openai.AsyncOpenAI`` client."""
        return self._client

    async def aembed_texts(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        """Embed a batch via the embeddings endpoint.

        Args:
            texts: The texts to embed.
            kind: Ignored — OpenAI embeddings are symmetric.

        Returns:
            One vector per input, in request order.

        Raises:
            ProviderError: For any API failure, translated from the SDK error.
        """
        from windlass.providers.llm.openai import translate_openai_error

        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self._requested_dimensions:
            payload["dimensions"] = self._requested_dimensions

        try:
            response = await self._client.embeddings.create(**payload)
        except Exception as exc:
            raise translate_openai_error(exc, self._sdk) from exc

        # The API may return items out of order; index is authoritative.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    async def aclose(self) -> None:
        """Close the SDK's HTTP connection pool."""
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()
