"""HuggingFace Inference API embeddings.

The ``huggingface`` provider runs *sentence-transformers locally* — which means
torch, a model download, and a machine with the RAM to hold it. This one calls
the hosted inference endpoint instead: an API token, no local model, no cold
start on your own hardware.

It needs no optional dependency at all. ``httpx`` is already part of the core
install, so this works on a bare ``pip install windlass``::

    embedder = Windlass.embedding("hf_inference", model="BAAI/bge-base-en-v1.5")

BGE and E5 models expect an *instruction prefix on queries but not on
documents*. Getting that asymmetry wrong is a silent quality bug — retrieval
still "works", just worse — so the prefixes are declared through the interface's
own hooks and applied by the base class on the right side of the split.

Example:
    >>> from windlass import Windlass                             # doctest: +SKIP
    >>> emb = Windlass.embedding("hf_inference")                  # doctest: +SKIP
    >>> len(emb.embed_one("hello"))                               # doctest: +SKIP
    768
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from windlass.core.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from windlass.core.registry import register
from windlass.interfaces.embedding import Embedder

__all__ = ["HuggingFaceInferenceEmbedder"]

#: The router endpoint. The legacy ``api-inference.huggingface.co`` host no
#: longer resolves, so pointing at it produces a DNS failure rather than a
#: helpful error — worth pinning explicitly.
_ROUTER = "https://router.huggingface.co/hf-inference/models"

#: Known output geometry, so a vector index can be provisioned before the first
#: call rather than after it.
_KNOWN_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "thenlper/gte-base": 768,
    "intfloat/e5-base-v2": 768,
}

#: Models whose retrieval quality depends on a query-side instruction prefix.
_QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2": "query: ",
}

#: Document-side prefix, for the models that want a symmetric marker.
_DOCUMENT_PREFIXES: dict[str, str] = {
    "intfloat/e5-base-v2": "passage: ",
}


@register.embedding(
    "hf_inference",
    aliases=("huggingface-api", "hf-api", "hf"),
    description="HuggingFace Inference API embeddings (hosted, no local model).",
)
class HuggingFaceInferenceEmbedder(Embedder):
    """Embeddings from the hosted HuggingFace Inference API.

    Args:
        model: Repository id, e.g. ``"BAAI/bge-base-en-v1.5"``.
        api_key: Token. Falls back to ``HUGGINGFACE_API_KEY`` then ``HF_TOKEN``.
        timeout: Per-request timeout. Hosted models cold-start, so this is
            generous by default.
        base_url: Inference router base URL. Override for a dedicated endpoint.
        dimensions: Output width. Inferred for known models, which lets a vector
            store be provisioned before the first call.
        **config: Forwarded to :class:`~windlass.interfaces.embedding.Embedder`.

    Raises:
        AuthenticationError: When no token is configured.

    Example:
        >>> from windlass import Windlass                          # doctest: +SKIP
        >>> emb = Windlass.embedding("hf_inference")               # doctest: +SKIP
        >>> emb.dimension()                                        # doctest: +SKIP
        768
    """

    provider_name = "hf_inference"

    def __init__(
        self,
        model: str = "",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        base_url: str = _ROUTER,
        dimensions: int | None = None,
        **config: Any,
    ) -> None:
        resolved = model or self.default_model()
        super().__init__(
            resolved,
            dimensions=dimensions if dimensions is not None else _KNOWN_DIMENSIONS.get(resolved),
            **config,
        )
        key = api_key or os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
        if not key:
            from windlass.core.config import settings

            key = settings().secret("huggingface_api_key")
        if not key:
            raise AuthenticationError(
                "No API token configured for the HuggingFace Inference API.",
                provider="hf_inference",
                hint=(
                    "Set HUGGINGFACE_API_KEY (or HF_TOKEN) in your environment or .env,\n"
                    "    or pass Windlass.embedding('hf_inference', api_key='hf_...')."
                ),
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.query_prefix = _QUERY_PREFIXES.get(resolved, "")
        self.document_prefix = _DOCUMENT_PREFIXES.get(resolved, "")
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    @classmethod
    def default_model(cls) -> str:
        """Return ``"BAAI/bge-base-en-v1.5"``."""
        return "BAAI/bge-base-en-v1.5"

    def native(self) -> httpx.AsyncClient:
        """Return the underlying ``httpx.AsyncClient`` (Level 3 access)."""
        return self._client

    async def aembed_texts(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        """Embed one batch through the feature-extraction pipeline.

        Args:
            texts: Texts to embed, already carrying any instruction prefix.
            kind: ``"document"`` or ``"query"``; the prefix was applied upstream.

        Returns:
            One vector per input, in input order.

        Raises:
            AuthenticationError: On 401 or 403.
            RateLimitError: On 429.
            ProviderTimeoutError: On a timeout or a cold-starting model (503).
            ProviderError: For any other failure, including a response whose
                shape does not match the request.
        """
        url = f"{self.base_url}/{self.model}/pipeline/feature-extraction"
        try:
            response = await self._client.post(url, json={"inputs": texts})
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"HuggingFace did not respond within {self.timeout:g}s.",
                provider="hf_inference",
                hint="Hosted models cold-start. Raise timeout=... or retry.",
                original=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Could not reach the HuggingFace Inference API: {exc}",
                provider="hf_inference",
                original=exc,
            ) from exc

        self._raise_for_status(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                "HuggingFace returned a non-JSON response.",
                provider="hf_inference",
                context={"body": response.text[:200]},
                original=exc,
            ) from exc

        return self._as_vectors(payload, expected=len(texts))

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Translate an HTTP error into the Windlass exception hierarchy."""
        if response.status_code < 400:
            return
        body = response.text[:300]
        if response.status_code in {401, 403}:
            raise AuthenticationError(
                f"HuggingFace rejected the token: {body}",
                provider="hf_inference",
                hint="Check HUGGINGFACE_API_KEY and that the token has inference access.",
            )
        if response.status_code == 429:
            raise RateLimitError(f"HuggingFace rate limit reached: {body}", provider="hf_inference")
        if response.status_code == 404:
            raise ProviderError(
                f"HuggingFace has no inference endpoint for {self.model!r}.",
                provider="hf_inference",
                hint="Check the repository id, and that the model supports feature-extraction.",
            )
        if response.status_code == 503:
            raise ProviderTimeoutError(
                f"HuggingFace is still loading {self.model!r}: {body}",
                provider="hf_inference",
                hint="The model is cold-starting; retrying usually succeeds.",
            )

        # status_code is what makes a transient 5xx retryable; see
        # windlass.core.retry.is_retryable. Omitting it would fail a whole
        # ingestion run that one retry would have fixed.
        raise ProviderError(
            f"HuggingFace returned HTTP {response.status_code}: {body}",
            provider="hf_inference",
            status_code=response.status_code,
        )

    def _as_vectors(self, payload: Any, *, expected: int) -> list[list[float]]:
        """Normalise the response into one flat vector per input.

        The endpoint returns ``[batch][dim]`` for sentence-transformer models but
        ``[batch][tokens][dim]`` for raw checkpoints with no pooling layer. Mean
        pooling the token axis is the standard recovery, and is far better than
        silently indexing the first token.

        Args:
            payload: The decoded JSON body.
            expected: How many inputs were sent.

        Returns:
            ``expected`` vectors.

        Raises:
            ProviderError: When the response cannot be interpreted, or does not
                have one entry per input.
        """
        if not isinstance(payload, list) or not payload:
            raise ProviderError(
                f"Unexpected embedding response shape: {type(payload).__name__}.",
                provider="hf_inference",
                context={"payload": str(payload)[:200]},
            )

        vectors: list[list[float]] = []
        for entry in payload:
            if not isinstance(entry, list) or not entry:
                raise ProviderError(
                    "Embedding response contained a non-vector entry.",
                    provider="hf_inference",
                )
            if isinstance(entry[0], list):
                width = len(entry[0])
                pooled = [sum(token[i] for token in entry) / len(entry) for i in range(width)]
                vectors.append(pooled)
            else:
                vectors.append([float(v) for v in entry])

        if len(vectors) != expected:
            raise ProviderError(
                f"HuggingFace returned {len(vectors)} vectors for {expected} inputs.",
                provider="hf_inference",
            )
        return vectors

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        await self._client.aclose()
