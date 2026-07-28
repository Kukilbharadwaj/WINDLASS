"""BM25 lexical retrieval.

Okapi BM25, implemented from scratch over an inverted index — no ``rank_bm25``,
no dependencies. It earns its place next to vector search because the two fail
in opposite ways: embeddings miss exact identifiers (error codes, product SKUs,
function names) that BM25 nails, and BM25 misses paraphrases that embeddings
handle. Hybrid retrieval combines both.

The scoring function is the standard one::

    score(q, d) = Σ  IDF(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))
                 t∈q

Example:
    >>> from windlass.core.types import Chunk
    >>> r = BM25Retriever()
    >>> r.index([
    ...     Chunk(content="error E1042 occurs when the socket closes"),
    ...     Chunk(content="the network layer handles reconnection"),
    ... ])
    2
    >>> r.retrieve("E1042").hits[0].chunk.content.startswith("error E1042")
    True
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any

from windlass.core.registry import register
from windlass.core.text import tokenize_words
from windlass.core.types import Chunk, ScoredChunk
from windlass.interfaces.retriever import Retriever
from windlass.interfaces.vectordb import MetadataFilter, VectorStore

__all__ = ["STOPWORDS", "BM25Retriever"]

#: A small English stop-word list. Removing these sharpens scoring on short
#: queries, where a single "the" can otherwise dominate the term set.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "these",
        "those",
        "there",
        "their",
        "they",
        "them",
        "then",
        "than",
        "but",
        "not",
        "no",
        "do",
        "does",
        "did",
        "done",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "i",
        "you",
        "your",
        "we",
        "us",
        "our",
        "if",
        "so",
        "such",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "only",
        "own",
        "same",
        "too",
        "very",
    ]
)


@register.retriever(
    "bm25",
    aliases=("keyword", "lexical", "sparse"),
    description="Okapi BM25 keyword search over an in-process inverted index (no dependencies).",
)
class BM25Retriever(Retriever):
    """Sparse lexical retriever.

    Args:
        top_k: Default number of results.
        k1: Term-frequency saturation. Higher values let repeated terms keep
            adding score; ``1.2``-``1.5`` is the usual range.
        b: Length normalisation, in ``[0, 1]``. ``0`` ignores document length,
            ``1`` fully normalises; ``0.75`` is the standard compromise.
        remove_stopwords: Drop :data:`STOPWORDS` from documents and queries.
        min_token_length: Ignore tokens shorter than this.
        vectorstore: Optional store to seed the index from at construction, so a
            hybrid retriever built on an existing collection works immediately.
        **config: Forwarded to :class:`~windlass.interfaces.retriever.Retriever`.

    Attributes:
        chunks: Indexed chunks, keyed by id.

    Performance:
        Indexing is ``O(total tokens)``. Query cost is proportional to the
        number of documents containing the query terms, not the corpus size,
        which makes it fast even on large collections. Memory is roughly
        proportional to the vocabulary.

    Thread safety:
        Indexing and search are guarded by a re-entrant lock.
    """

    provider_name = "bm25"
    requires_index = True

    def __init__(
        self,
        *,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True,
        min_token_length: int = 1,
        vectorstore: VectorStore | None = None,
        **config: Any,
    ) -> None:
        super().__init__(top_k=top_k, **config)
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

        self.chunks: dict[str, Chunk] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._lengths: dict[str, int] = {}
        self._total_length = 0
        self._lock = threading.RLock()

        if vectorstore is not None:
            seed = getattr(vectorstore, "all_chunks", None)
            if callable(seed):
                self.index(seed())

    # -- indexing ---------------------------------------------------------
    async def aindex(self, chunks: Sequence[Chunk]) -> int:
        """Add chunks to the inverted index.

        Re-indexing a chunk id replaces its previous entry, so ingestion is
        idempotent.

        Args:
            chunks: Chunks to index.

        Returns:
            How many chunks were indexed.
        """
        if not chunks:
            return 0
        with self._lock:
            for chunk in chunks:
                if chunk.id in self.chunks:
                    self._remove(chunk.id)
                tokens = self._tokenize(chunk.content)
                if not tokens:
                    self.chunks[chunk.id] = chunk
                    self._lengths[chunk.id] = 0
                    continue
                frequencies: dict[str, int] = {}
                for token in tokens:
                    frequencies[token] = frequencies.get(token, 0) + 1
                for token, count in frequencies.items():
                    self._postings.setdefault(token, {})[chunk.id] = count
                self.chunks[chunk.id] = chunk
                self._lengths[chunk.id] = len(tokens)
                self._total_length += len(tokens)
        return len(chunks)

    def remove(self, chunk_ids: Sequence[str]) -> int:
        """Remove chunks from the index.

        Args:
            chunk_ids: Ids to drop.

        Returns:
            How many were actually removed.
        """
        with self._lock:
            return sum(1 for cid in chunk_ids if self._remove(cid))

    def _remove(self, chunk_id: str) -> bool:
        """Drop one chunk from every posting list. Caller holds the lock."""
        if chunk_id not in self.chunks:
            return False
        self._total_length -= self._lengths.pop(chunk_id, 0)
        del self.chunks[chunk_id]
        emptied: list[str] = []
        for token, postings in self._postings.items():
            postings.pop(chunk_id, None)
            if not postings:
                emptied.append(token)
        for token in emptied:
            self._postings.pop(token, None)
        return True

    def clear(self) -> None:
        """Empty the index."""
        with self._lock:
            self.chunks.clear()
            self._postings.clear()
            self._lengths.clear()
            self._total_length = 0

    # -- retrieval --------------------------------------------------------
    async def aretrieve_chunks(
        self,
        query: str,
        k: int,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Score the corpus against ``query`` using BM25.

        Args:
            query: The search query.
            k: How many results to return.
            filters: Metadata constraints, applied to candidates before ranking.
            **kwargs: Ignored.

        Returns:
            Scored chunks, best first. Chunks matching no query term are omitted
            entirely rather than returned with a zero score.
        """
        terms = self._tokenize(query)
        if not terms:
            return []

        with self._lock:
            corpus_size = len(self.chunks)
            if corpus_size == 0:
                return []
            average_length = (self._total_length / corpus_size) or 1.0

            scores: dict[str, float] = {}
            for term in set(terms):
                postings = self._postings.get(term)
                if not postings:
                    continue
                idf = self._idf(len(postings), corpus_size)
                for chunk_id, frequency in postings.items():
                    length = self._lengths.get(chunk_id, 0)
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / average_length
                    )
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (
                        frequency * (self.k1 + 1) / denominator
                    )

            hits = [
                ScoredChunk(chunk=self.chunks[cid], score=score, retriever=self.name)
                for cid, score in scores.items()
                if cid in self.chunks
                and VectorStore.match_filters(self.chunks[cid].metadata, filters)
            ]

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # -- helpers ----------------------------------------------------------
    def _tokenize(self, text: str) -> list[str]:
        """Tokenise and filter text for indexing or querying."""
        tokens = tokenize_words(text)
        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        if self.remove_stopwords:
            filtered = [t for t in tokens if t not in STOPWORDS]
            # A query of nothing but stop-words should still match something.
            tokens = filtered or tokens
        return tokens

    @staticmethod
    def _idf(document_frequency: int, corpus_size: int) -> float:
        """Return the smoothed inverse document frequency for a term.

        Uses the standard BM25+ smoothing, which keeps the value positive even
        for a term appearing in more than half the corpus (the classic Robertson
        formula can go negative there and quietly penalise common words).
        """
        return math.log(1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5))

    def stats(self) -> dict[str, Any]:
        """Return index statistics.

        Returns:
            A dict with ``documents``, ``vocabulary``, ``total_tokens`` and
            ``average_length``.
        """
        with self._lock:
            count = len(self.chunks)
            return {
                "documents": count,
                "vocabulary": len(self._postings),
                "total_tokens": self._total_length,
                "average_length": (self._total_length / count) if count else 0.0,
            }

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        return f"BM25Retriever(documents={len(self.chunks)}, k1={self.k1}, b={self.b})"
