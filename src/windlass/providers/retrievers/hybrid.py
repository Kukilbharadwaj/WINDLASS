"""Hybrid retrieval — dense and sparse, fused.

Vector search and BM25 fail differently. Embeddings understand "how do I cancel
my plan" ≈ "subscription termination"; BM25 finds the literal string ``E1042``
that no embedding will ever place near your query. Running both and fusing the
results measurably beats either alone on virtually every benchmark.

Windlass fuses with **Reciprocal Rank Fusion**, which combines *ranks* rather than
scores. That matters: a cosine similarity of 0.81 and a BM25 score of 14.3 are
not comparable quantities, and normalising them requires corpus statistics that
change every time you ingest. Ranks need no normalisation at all.

Example:
    >>> from windlass.providers.embeddings.hash import HashEmbedder
    >>> from windlass.providers.vectordb.memory import InMemoryVectorStore
    >>> from windlass.providers.retrievers.vector import VectorRetriever
    >>> from windlass.providers.retrievers.bm25 import BM25Retriever
    >>> from windlass.core.types import Chunk
    >>> emb, store = HashEmbedder(dimensions=128), InMemoryVectorStore(dimensions=128)
    >>> chunks = [Chunk(content="error E1042 socket closed"), Chunk(content="cats purr")]
    >>> for c, v in zip(chunks, emb.embed([c.content for c in chunks])):
    ...     c.embedding = v
    >>> _ = store.add(chunks)
    >>> lexical = BM25Retriever(); _ = lexical.index(chunks)
    >>> hybrid = HybridRetriever(
    ...     retrievers=[VectorRetriever(embedder=emb, vectorstore=store), lexical]
    ... )
    >>> hybrid.retrieve("E1042").hits[0].chunk.content
    'error E1042 socket closed'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from windlass.core.concurrency import gather_bounded
from windlass.core.exceptions import ConfigurationError
from windlass.core.registry import register
from windlass.core.types import Chunk, ScoredChunk
from windlass.core.vectors import reciprocal_rank_fusion
from windlass.interfaces.retriever import Retriever
from windlass.interfaces.vectordb import MetadataFilter

__all__ = ["HybridRetriever"]


@register.retriever(
    "hybrid",
    aliases=("fusion", "rrf", "dense+sparse"),
    description="Fuses several retrievers with Reciprocal Rank Fusion.",
)
class HybridRetriever(Retriever):
    """Fuses any number of retrievers into one ranking.

    Args:
        retrievers: The legs to run. Typically one dense and one sparse, but any
            number of any kind works — add a graph retriever or a remote search
            API and it fuses just the same.
        weights: Per-leg weights, in the same order. Defaults to equal weight.
            Raise the dense weight for conversational corpora, the sparse weight
            for technical ones full of identifiers.
        top_k: Default number of results.
        fetch_k: Candidates pulled from each leg before fusion. Should exceed
            ``top_k`` — fusion can only reorder what it is given.
        rrf_k: RRF smoothing constant. Larger values flatten the influence of
            the top positions.
        mode: ``"rrf"`` (rank fusion, the default) or ``"score"`` (weighted sum
            of min-max normalised scores, for when your legs already produce
            comparable scores).
        **config: Forwarded to :class:`~windlass.interfaces.retriever.Retriever`.

    Raises:
        ConfigurationError: When fewer than two retrievers are supplied, or when
            ``weights`` has the wrong length.

    Performance:
        Legs run concurrently, so hybrid retrieval costs about as much wall-clock
        time as its slowest leg, not the sum.
    """

    provider_name = "hybrid"

    def __init__(
        self,
        *,
        retrievers: Sequence[Retriever] | None = None,
        weights: Sequence[float] | None = None,
        top_k: int = 5,
        fetch_k: int = 20,
        rrf_k: int = 60,
        mode: str = "rrf",
        **config: Any,
    ) -> None:
        legs = list(retrievers or [])
        if len(legs) < 2:
            raise ConfigurationError(
                "Hybrid retrieval needs at least two retrievers.",
                hint="Windlass.rag().retriever('hybrid') builds a vector + BM25 pair "
                "for you; pass retrievers=[...] to choose your own.",
            )
        if weights is not None and len(weights) != len(legs):
            raise ConfigurationError(
                f"Got {len(weights)} weights for {len(legs)} retrievers.",
                hint="Provide one weight per retriever, or omit weights entirely.",
            )
        if mode not in {"rrf", "score"}:
            raise ValueError("mode must be 'rrf' or 'score'")

        super().__init__(top_k=top_k, fetch_k=fetch_k, **config)
        self.retrievers = legs
        self.weights = list(weights) if weights else [1.0] * len(legs)
        self.rrf_k = rrf_k
        self.mode = mode

    async def aindex(self, chunks: Sequence[Chunk]) -> int:
        """Index chunks into every leg that maintains its own index.

        Args:
            chunks: Chunks to index.

        Returns:
            The largest count reported by any leg — legs share one corpus, so
            summing would over-report.
        """
        if not chunks:
            return 0
        counts = await gather_bounded(
            [r.aindex(chunks) for r in self.retrievers], limit=len(self.retrievers)
        )
        return max(counts) if counts else 0

    async def aretrieve_chunks(
        self,
        query: str,
        k: int,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Run every leg concurrently and fuse the rankings.

        A leg that fails is logged and skipped rather than failing the whole
        retrieval — degraded results beat no results.

        Args:
            query: The search query.
            k: How many results to return.
            filters: Metadata constraints, passed to every leg.
            **kwargs: Forwarded to every leg.

        Returns:
            The fused ranking.
        """
        fetch = max(k, self.fetch_k)
        outcomes = await gather_bounded(
            [r.aretrieve(query, fetch, filters=filters, **kwargs) for r in self.retrievers],
            limit=len(self.retrievers),
            return_exceptions=True,
        )

        rankings: list[list[str]] = []
        weights: list[float] = []
        catalogue: dict[str, Chunk] = {}
        raw_scores: list[dict[str, float]] = []
        provenance: dict[str, list[str]] = {}

        for retriever, weight, outcome in zip(self.retrievers, self.weights, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                self._log.warning(
                    "Hybrid leg %r failed and was skipped: %s", retriever.name, outcome
                )
                continue
            ids: list[str] = []
            scores: dict[str, float] = {}
            for hit in outcome.hits:
                catalogue.setdefault(hit.chunk.id, hit.chunk)
                ids.append(hit.chunk.id)
                scores[hit.chunk.id] = hit.score
                provenance.setdefault(hit.chunk.id, []).append(retriever.name)
            rankings.append(ids)
            raw_scores.append(scores)
            weights.append(weight)

        if not rankings:
            return []

        fused = (
            reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=weights)
            if self.mode == "rrf"
            else _weighted_scores(raw_scores, weights)
        )

        hits: list[ScoredChunk] = []
        for rank, (chunk_id, score) in enumerate(list(fused.items())[:k], start=1):
            chunk = catalogue.get(chunk_id)
            if chunk is None:  # pragma: no cover - defensive
                continue
            hits.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retriever="+".join(provenance.get(chunk_id, [self.name])),
                )
            )
        return hits

    def describe(self) -> dict[str, Any]:
        """Return a summary including each leg and its weight."""
        return {
            **super().describe(),
            "mode": self.mode,
            "legs": [
                {"retriever": r.describe(), "weight": w}
                for r, w in zip(self.retrievers, self.weights, strict=True)
            ],
        }

    def __repr__(self) -> str:
        legs = ", ".join(
            f"{r.name}×{w}" for r, w in zip(self.retrievers, self.weights, strict=True)
        )
        return f"HybridRetriever({legs}, mode={self.mode!r})"


def _weighted_scores(score_sets: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    """Fuse by weighted sum of min-max normalised scores.

    An alternative to RRF for legs whose scores are already on comparable
    scales. Each leg's scores are normalised into ``[0, 1]`` first, so one leg
    with a wide score range cannot dominate.

    Args:
        score_sets: One ``{chunk_id: score}`` map per leg.
        weights: Per-leg weight.

    Returns:
        Fused scores, highest first.

    Example:
        >>> _weighted_scores([{"a": 1.0, "b": 0.0}], [1.0])
        {'a': 1.0, 'b': 0.0}
    """
    fused: dict[str, float] = {}
    for scores, weight in zip(score_sets, weights, strict=True):
        if not scores:
            continue
        values = list(scores.values())
        low, high = min(values), max(values)
        span = high - low
        for chunk_id, score in scores.items():
            normalised = (score - low) / span if span else 1.0
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight * normalised
    return dict(sorted(fused.items(), key=lambda kv: kv[1], reverse=True))
