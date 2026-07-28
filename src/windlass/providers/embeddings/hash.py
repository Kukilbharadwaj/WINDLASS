"""A deterministic, dependency-free embedding model.

``HashEmbedder`` projects text into a fixed-dimensional space using hashed
character n-grams — the classic "hashing trick". It is not a semantic model and
will never beat a trained encoder, but it has three properties that make it the
right default for a framework:

* **Zero dependencies and zero latency.** The full RAG pipeline runs on a bare
  ``pip install windlass``.
* **Deterministic.** The same text always yields the same vector, so tests can
  assert on retrieval order.
* **Genuinely useful for lexical overlap.** Documents sharing character n-grams
  score higher, which makes the smoke tests and tutorials return sensible
  results instead of noise.

Switch to ``huggingface`` or ``openai`` the moment you care about semantics.

Example:
    >>> emb = HashEmbedder(dimensions=64)
    >>> a, b, c = emb.embed(["machine learning", "machine learning models", "sailing"])
    >>> from windlass.core.vectors import cosine_similarity
    >>> cosine_similarity(a, b) > cosine_similarity(a, c)
    True
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from windlass.core.registry import register
from windlass.interfaces.embedding import Embedder

__all__ = ["HashEmbedder"]


@register.embedding(
    "hash",
    aliases=("hashing", "fake", "test"),
    description="Deterministic hashed n-gram embeddings (no dependencies).",
)
class HashEmbedder(Embedder):
    """Hashed character n-gram embeddings.

    Args:
        dimensions: Output dimensionality. Higher values reduce hash collisions;
            256 is a reasonable floor for anything beyond a smoke test.
        ngram: Character n-gram length used for the sub-word signal.
        use_words: Also hash whole words, which sharpens exact-term matching.
        **config: Forwarded to :class:`~windlass.interfaces.embedding.Embedder`.

    Example:
        >>> len(HashEmbedder(dimensions=32).embed_one("hello"))
        32
    """

    provider_name = "hash"

    def __init__(
        self,
        *,
        dimensions: int = 384,
        ngram: int = 3,
        use_words: bool = True,
        **config: Any,
    ) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        if ngram < 1:
            raise ValueError("ngram must be at least 1")
        config.setdefault("model", "hash-ngram")
        super().__init__(dimensions=dimensions, **config)
        self.ngram = ngram
        self.use_words = use_words

    @classmethod
    def default_model(cls) -> str:
        """Return ``"hash-ngram"``."""
        return "hash-ngram"

    def dimension(self) -> int:
        """Return the configured dimensionality without probing."""
        return int(self._dimensions or 384)

    async def aembed_texts(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed.
            kind: Ignored — this model treats queries and documents identically.

        Returns:
            One vector per input.
        """
        return [self._vectorize(text) for text in texts]

    def _vectorize(self, text: str) -> list[float]:
        """Project one text into the hashed feature space.

        Uses signed hashing (each feature contributes ±1) so unrelated features
        cancel rather than accumulate, and sub-linear term weighting so a single
        repeated word cannot dominate the vector.
        """
        size = self.dimension()
        vector = [0.0] * size
        lowered = text.lower()

        counts: dict[str, int] = {}
        if self.use_words:
            for word in lowered.split():
                token = word.strip(".,!?;:\"'()[]{}")
                if token:
                    counts[f"w:{token}"] = counts.get(f"w:{token}", 0) + 1

        padded = f" {lowered} "
        for i in range(max(0, len(padded) - self.ngram + 1)):
            gram = padded[i : i + self.ngram]
            counts[f"g:{gram}"] = counts.get(f"g:{gram}", 0) + 1

        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % size
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        return vector
