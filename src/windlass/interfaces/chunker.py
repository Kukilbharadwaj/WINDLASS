"""The chunker interface.

Chunking is the single highest-leverage decision in a RAG system: it determines
what the retriever can find and what the model gets to read. Windlass therefore
treats it as a first-class, swappable strategy rather than a helper function.

Implementers override one method, :meth:`Chunker.split_text`, which works on a
plain string. The base class handles document iteration, metadata propagation,
offset tracking and id assignment — so a custom chunker is genuinely a dozen
lines.

Example:
    >>> from windlass.providers.chunkers.recursive import RecursiveChunker
    >>> chunker = RecursiveChunker(chunk_size=60, overlap=0, min_chunk_size=0)
    >>> len(chunker.chunk_text("word " * 60)) >= 2
    True
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from windlass.core.concurrency import gather_bounded, run_sync
from windlass.core.config import settings
from windlass.core.text import count_tokens
from windlass.core.types import Chunk, Document
from windlass.interfaces.base import Component

__all__ = ["DEFAULT_OVERLAP_RATIO", "Chunker"]

#: Fraction of ``chunk_size`` used when a chunker is given no explicit overlap.
#: 20% is the usual rule of thumb, and scaling means the default is valid for
#: every chunk size rather than only the large ones.
DEFAULT_OVERLAP_RATIO = 0.2


class Chunker(Component):
    """Abstract text splitter.

    Args:
        chunk_size: Target size of each chunk, measured in :attr:`unit`.
        overlap: How much each chunk repeats from the previous one. Overlap
            keeps a fact that straddles a boundary retrievable from either side.
            ``None`` scales it to :data:`DEFAULT_OVERLAP_RATIO` of ``chunk_size``,
            so shrinking ``chunk_size`` alone always produces a working chunker.
        min_chunk_size: Chunks shorter than this are merged into their
            neighbour rather than emitted — stray one-line fragments pollute
            retrieval.
        keep_separator: Whether the split separator stays in the output.
        name: Component name for traces.
        **config: Strategy-specific options.

    Attributes:
        unit: ``"char"`` or ``"token"`` — how ``chunk_size`` is measured.

    Raises:
        ValueError: If an *explicit* ``overlap`` is negative or not smaller than
            ``chunk_size``. A derived default is never invalid.

    Example:
        Implementing a chunker takes one method::

            class LineChunker(Chunker):
                provider_name = "line"

                def split_text(self, text: str) -> list[str]:
                    return [ln for ln in text.splitlines() if ln.strip()]
    """

    kind = "chunker"
    provider_name: str = "chunker"

    #: Unit ``chunk_size`` is expressed in.
    unit: str = "char"

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        overlap: int | None = None,
        min_chunk_size: int = 50,
        keep_separator: bool = True,
        name: str | None = None,
        **config: Any,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap is None:
            # Scale with chunk_size. A fixed default would make the perfectly
            # reasonable `chunker(chunk_size=20)` raise, which is a poor trade:
            # the user changed one obvious knob and got an error about another
            # they never set.
            overlap = int(chunk_size * DEFAULT_OVERLAP_RATIO)
        elif overlap < 0:
            raise ValueError("overlap must not be negative")
        elif overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}); "
                "otherwise chunking cannot make progress."
            )
        super().__init__(
            name=name or self.provider_name,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            keep_separator=keep_separator,
            **config,
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.keep_separator = keep_separator

    # -- provider hooks ---------------------------------------------------
    @abc.abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split raw text into chunk strings.

        The only method a strategy must implement. Return the pieces in reading
        order; the base class turns them into :class:`Chunk` objects with ids,
        offsets and inherited metadata.

        Args:
            text: The text to split.

        Returns:
            Chunk strings in document order.
        """

    async def asplit_text(self, text: str) -> list[str]:
        """Async :meth:`split_text`.

        Override this instead when your strategy needs to await something — the
        semantic chunker calls an embedding model here.

        Args:
            text: The text to split.

        Returns:
            Chunk strings in document order.
        """
        return self.split_text(text)

    # -- public API -------------------------------------------------------
    async def achunk_document(self, document: Document) -> list[Chunk]:
        """Split one document into chunks.

        Args:
            document: The document to split.

        Returns:
            Chunks carrying the document's metadata plus ``chunk_index``,
            ``chunk_total`` and character offsets.
        """
        pieces = await self.asplit_text(document.content)
        pieces = self._merge_undersized(pieces)
        chunks: list[Chunk] = []
        cursor = 0
        for index, piece in enumerate(pieces):
            start = document.content.find(piece, cursor)
            if start < 0:  # strategy rewrote the text (e.g. added a heading path)
                start = cursor
            else:
                cursor = start + len(piece)
            chunks.append(
                Chunk(
                    content=piece,
                    document_id=document.id,
                    index=index,
                    start_char=start,
                    end_char=start + len(piece),
                    metadata={
                        **document.metadata,
                        "document_id": document.id,
                        "chunk_index": index,
                        "chunk_total": len(pieces),
                        "chunker": self.name,
                        **({"source": document.source} if document.source else {}),
                    },
                )
            )
        return chunks

    async def achunk(
        self, documents: Sequence[Document] | Document, *, concurrency: int | None = None
    ) -> list[Chunk]:
        """Split many documents concurrently.

        Args:
            documents: One document or a sequence of them.
            concurrency: Maximum simultaneous splits. Defaults to the global
                ``max_concurrency`` setting.

        Returns:
            All chunks, grouped by source document in input order.

        Performance:
            CPU-bound strategies see little benefit from concurrency; strategies
            that await a model (``semantic``, ``contextual``) see a lot.
        """
        items = [documents] if isinstance(documents, Document) else list(documents)
        if not items:
            return []
        limit = concurrency or settings().max_concurrency
        grouped = await gather_bounded([self.achunk_document(doc) for doc in items], limit=limit)
        return [chunk for group in grouped for chunk in group]

    def chunk(self, documents: Sequence[Document] | Document) -> list[Chunk]:
        """Blocking :meth:`achunk`."""
        return run_sync(self.achunk(documents))

    def chunk_text(self, text: str, *, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split a bare string, without constructing a Document first.

        Args:
            text: The text to split.
            metadata: Metadata copied onto every chunk.

        Returns:
            The resulting chunks.

        Example:
            >>> from windlass.providers.chunkers.recursive import RecursiveChunker
            >>> RecursiveChunker(chunk_size=100, overlap=10).chunk_text("hello")[0].content
            'hello'
        """
        return run_sync(self.achunk(Document(content=text, metadata=dict(metadata or {}))))

    # -- helpers ----------------------------------------------------------
    def _size(self, text: str) -> int:
        """Measure ``text`` in this chunker's :attr:`unit`."""
        return count_tokens(text) if self.unit == "token" else len(text)

    def _merge_undersized(self, pieces: list[str]) -> list[str]:
        """Fold fragments below ``min_chunk_size`` into their neighbour.

        Splitters routinely emit a trailing scrap — a lone heading, half a
        sentence. Those retrieve badly and waste an index slot, so they are
        merged backwards (or forwards, for a leading scrap).
        """
        cleaned = [p.strip() for p in pieces if p and p.strip()]
        if not cleaned or self.min_chunk_size <= 0:
            return cleaned

        merged: list[str] = []
        for piece in cleaned:
            if merged and self._size(piece) < self.min_chunk_size:
                merged[-1] = f"{merged[-1]}\n{piece}"
            else:
                merged.append(piece)
        # A leading scrap has no previous neighbour; fold it forwards instead.
        if len(merged) > 1 and self._size(merged[0]) < self.min_chunk_size:
            merged[1] = f"{merged[0]}\n{merged[1]}"
            merged.pop(0)
        return merged

    def _apply_overlap(self, pieces: list[str]) -> list[str]:
        """Prepend the tail of each piece to the next one.

        Used by strategies that split on hard boundaries (sentences, headings)
        and want overlap applied afterwards.
        """
        if self.overlap <= 0 or len(pieces) < 2:
            return pieces
        out = [pieces[0]]
        for previous, current in pairwise(pieces):
            tail = previous[-self.overlap :]
            out.append(f"{tail}{current}" if tail else current)
        return out

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(chunk_size={self.chunk_size}, "
            f"overlap={self.overlap}, unit={self.unit!r})"
        )
