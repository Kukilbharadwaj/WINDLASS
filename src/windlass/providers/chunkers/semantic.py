"""Semantic chunking — split where the meaning changes, not where the count runs out.

Fixed-size chunking cuts through the middle of an argument as often as not.
Semantic chunking embeds each sentence, walks the document measuring the
similarity between consecutive sentences, and starts a new chunk wherever that
similarity drops sharply. Boundaries land at topic shifts.

The cost is one embedding call per sentence at ingestion time. That is real, but
it is paid once and it improves every retrieval afterwards.

Example:
    >>> from windlass.providers.embeddings.hash import HashEmbedder
    >>> text = (
    ...     "Cats are mammals. Cats purr when content. Cats groom themselves often. "
    ...     "Rockets burn fuel. Rockets reach orbit. Rockets carry satellites."
    ... )
    >>> chunker = SemanticChunker(embedder=HashEmbedder(dimensions=256), threshold=0.5)
    >>> len(chunker.split_text(text)) >= 1
    True
"""

from __future__ import annotations

import statistics
from typing import Any

from windlass.core.concurrency import run_sync
from windlass.core.exceptions import ConfigurationError
from windlass.core.registry import register
from windlass.core.text import split_sentences
from windlass.core.vectors import cosine_similarity
from windlass.interfaces.chunker import Chunker
from windlass.interfaces.embedding import Embedder

__all__ = ["SemanticChunker"]


@register.chunker(
    "semantic",
    aliases=("meaning",),
    description="Splits at topic shifts detected from sentence embeddings.",
)
class SemanticChunker(Chunker):
    """Embedding-driven topic-boundary splitter.

    Args:
        embedder: Embedding model used to score sentence similarity. Required —
            the RAG builder injects the pipeline's embedder automatically.
        chunk_size: Upper bound in characters. A topic that runs longer than this
            is still split, so no chunk can blow past your context budget.
        overlap: Characters repeated between chunks.
        threshold: Similarity below which a boundary is placed. Ignored when
            ``percentile`` is set.
        percentile: Adaptive alternative to ``threshold``: place a boundary
            wherever similarity falls into the lowest ``percentile`` of the
            document's own distribution. More robust across mixed corpora, since
            "low similarity" means different things in a legal contract and a
            chat log.
        buffer_size: How many neighbouring sentences to average into each
            comparison. ``1`` compares single sentences and is noisy; ``2``-``3``
            smooths it out.
        min_sentences: Minimum sentences per chunk, to avoid one-line chunks.
        **config: Forwarded to :class:`~windlass.interfaces.chunker.Chunker`.

    Raises:
        ConfigurationError: When no embedder is supplied.

    Performance:
        Embeds every sentence once per document. Sentence embeddings are batched
        into a single call, and the embedder's cache (when configured) makes
        re-ingestion nearly free.
    """

    provider_name = "semantic"
    unit = "char"

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        chunk_size: int = 2000,
        overlap: int = 0,
        threshold: float = 0.5,
        percentile: float | None = 20.0,
        buffer_size: int = 2,
        min_sentences: int = 2,
        **config: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, overlap=overlap, **config)
        if embedder is None:
            raise ConfigurationError(
                "Semantic chunking needs an embedding model.",
                hint="Pass one explicitly — Windlass.chunker('semantic', "
                "embedder=Windlass.embedding('huggingface')) — or use the RAG "
                "builder, which injects the pipeline's embedder for you.",
            )
        self.embedder = embedder
        self.threshold = threshold
        self.percentile = percentile
        self.buffer_size = max(1, buffer_size)
        self.min_sentences = max(1, min_sentences)

    def split_text(self, text: str) -> list[str]:
        """Blocking :meth:`asplit_text`."""
        return run_sync(self.asplit_text(text))

    async def asplit_text(self, text: str) -> list[str]:
        """Split ``text`` at detected topic boundaries.

        Args:
            text: The text to split.

        Returns:
            Chunk strings. Short inputs (fewer than ``2 * min_sentences``
            sentences) are returned whole rather than force-split.
        """
        sentences = split_sentences(text)
        if len(sentences) < max(2, self.min_sentences * 2):
            stripped = text.strip()
            return [stripped] if stripped else []

        windows = self._windows(sentences)
        vectors = await self.embedder.aembed(windows)
        similarities = [
            cosine_similarity(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
        ]
        cutoff = self._cutoff(similarities)
        boundaries = self._boundaries(similarities, cutoff)
        return self._assemble(sentences, boundaries)

    # -- internals --------------------------------------------------------
    def _windows(self, sentences: list[str]) -> list[str]:
        """Build the buffered text windows that get embedded.

        Averaging a sentence with its neighbours is what stops a single short
        sentence ("Indeed.") from registering as a topic change.
        """
        if self.buffer_size <= 1:
            return sentences
        windows: list[str] = []
        half = self.buffer_size // 2
        for i in range(len(sentences)):
            start = max(0, i - half)
            end = min(len(sentences), i + half + 1)
            windows.append(" ".join(sentences[start:end]))
        return windows

    def _cutoff(self, similarities: list[float]) -> float:
        """Decide the similarity value below which a boundary is placed."""
        if not similarities:
            return self.threshold
        if self.percentile is None:
            return self.threshold
        ordered = sorted(similarities)
        position = min(
            len(ordered) - 1,
            max(0, int(len(ordered) * (self.percentile / 100.0))),
        )
        adaptive = ordered[position]
        # Guard against degenerate documents where every sentence is identical.
        if len(ordered) > 2 and statistics.pstdev(ordered) < 1e-6:
            return -1.0
        return adaptive

    def _boundaries(self, similarities: list[float], cutoff: float) -> list[int]:
        """Return sentence indices at which a new chunk should start."""
        boundaries: list[int] = []
        last = 0
        for i, similarity in enumerate(similarities):
            if similarity <= cutoff and (i + 1 - last) >= self.min_sentences:
                boundaries.append(i + 1)
                last = i + 1
        return boundaries

    def _assemble(self, sentences: list[str], boundaries: list[int]) -> list[str]:
        """Join sentences into chunks, honouring the size ceiling."""
        chunks: list[str] = []
        start = 0
        for boundary in [*boundaries, len(sentences)]:
            group = sentences[start:boundary]
            start = boundary
            if not group:
                continue
            text = " ".join(group)
            if len(text) <= self.chunk_size:
                chunks.append(text)
            else:
                chunks.extend(self._enforce_size(group))
        return self._merge_undersized(chunks)

    def _enforce_size(self, sentences: list[str]) -> list[str]:
        """Break an over-long topic into size-bounded pieces at sentence edges."""
        out: list[str] = []
        current: list[str] = []
        length = 0
        for sentence in sentences:
            if current and length + len(sentence) + 1 > self.chunk_size:
                out.append(" ".join(current))
                current, length = [], 0
            current.append(sentence)
            length += len(sentence) + 1
        if current:
            out.append(" ".join(current))
        return out
