r"""Recursive and token-based chunking.

``RecursiveChunker`` is the right default for prose. It splits on the largest
natural boundary that produces small-enough pieces — paragraphs first, then
lines, sentences, clauses, words, and finally characters — so chunks end where a
human would end them instead of mid-word.

``TokenChunker`` does the same thing but measures in model tokens, which is what
you want when you must guarantee that ``k`` chunks fit inside a context window.

Neither has any dependencies.

Example:
    >>> text = "First para.\n\nSecond para is longer and keeps going for a while."
    >>> chunks = RecursiveChunker(chunk_size=40, overlap=5).split_text(text)
    >>> all(len(c) <= 60 for c in chunks)
    True
"""

from __future__ import annotations

from typing import Any

from windlass.core.registry import register
from windlass.core.text import count_tokens
from windlass.interfaces.chunker import Chunker

__all__ = ["DEFAULT_SEPARATORS", "RecursiveChunker", "TokenChunker"]

#: Separators tried in order, from the most to the least semantically meaningful.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",  # paragraph
    "\n",  # line
    ". ",  # sentence
    "! ",
    "? ",
    "; ",  # clause
    ", ",
    " ",  # word
    "",  # character (last resort)
)


@register.chunker(
    "recursive",
    aliases=("default", "text"),
    description="Splits on the largest natural boundary that fits (no dependencies).",
)
class RecursiveChunker(Chunker):
    """Character-based recursive splitter.

    Args:
        chunk_size: Target chunk length in characters.
        overlap: Characters repeated between consecutive chunks.
        separators: Boundary strings tried in order. The default walks from
            paragraphs down to individual characters.
        **config: Forwarded to :class:`~windlass.interfaces.chunker.Chunker` —
            notably ``min_chunk_size``, below which a fragment is merged into
            its neighbour rather than emitted on its own.

    Performance:
        Linear in the length of the text. A 10 MB document splits in well under
        a second.

    Example:
        >>> chunker = RecursiveChunker(chunk_size=30, overlap=0)
        >>> parts = chunker.split_text("alpha beta gamma delta epsilon zeta eta")
        >>> len(parts) > 1
        True
    """

    provider_name = "recursive"
    unit = "char"

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        overlap: int | None = None,
        separators: tuple[str, ...] | list[str] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, overlap=overlap, **config)
        self.separators = tuple(separators) if separators else DEFAULT_SEPARATORS

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` recursively.

        Args:
            text: The text to split.

        Returns:
            Chunk strings, each at most ``chunk_size`` in this chunker's unit
            (except where a single indivisible token exceeds it).
        """
        if not text or not text.strip():
            return []
        if self._size(text) <= self.chunk_size:
            return [text.strip()]
        pieces = self._split(text, list(self.separators))
        return self._pack(pieces)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively break ``text`` into pieces below the size limit."""
        if self._size(text) <= self.chunk_size:
            return [text]
        if not separators:
            return self._hard_split(text)

        separator = separators[0]
        rest = separators[1:]
        if separator == "":
            return self._hard_split(text)

        parts = text.split(separator)
        if len(parts) == 1:
            return self._split(text, rest)

        out: list[str] = []
        for position, part in enumerate(parts):
            # Keep the separator attached so re-joined chunks read naturally.
            piece = part if position == len(parts) - 1 else part + separator
            if not piece:
                continue
            if self._size(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._split(piece, rest))
        return out

    def _hard_split(self, text: str) -> list[str]:
        """Cut ``text`` at fixed offsets when no separator helps."""
        if self.unit == "token":
            return _split_by_tokens(text, self.chunk_size)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _pack(self, pieces: list[str]) -> list[str]:
        """Greedily merge pieces up to ``chunk_size``, then apply overlap.

        Splitting produces many small fragments; packing them back up to the
        target size is what keeps chunk counts sane and context windows full.
        """
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        for piece in pieces:
            size = self._size(piece)
            if current and current_size + size > self.chunk_size:
                chunks.append("".join(current))
                current, current_size = self._carry_over(current)
            current.append(piece)
            current_size += size

        if current:
            chunks.append("".join(current))
        return [c.strip() for c in chunks if c.strip()]

    def _carry_over(self, pieces: list[str]) -> tuple[list[str], int]:
        """Return the trailing pieces that seed the next chunk as overlap."""
        if self.overlap <= 0:
            return [], 0
        carried: list[str] = []
        size = 0
        for piece in reversed(pieces):
            piece_size = self._size(piece)
            if size + piece_size > self.overlap and carried:
                break
            carried.insert(0, piece)
            size += piece_size
        return carried, size


@register.chunker(
    "token",
    aliases=("tokens",),
    description="Recursive splitting measured in model tokens rather than characters.",
)
class TokenChunker(RecursiveChunker):
    """Recursive splitter that measures chunks in tokens.

    Use this when a hard context budget matters: ``chunk_size=512`` with
    ``top_k=8`` guarantees roughly 4k tokens of context regardless of how
    verbose the source text is.

    Args:
        chunk_size: Target chunk length in tokens.
        overlap: Tokens repeated between consecutive chunks.
        model: Model name used to select the tokeniser.
        **config: Forwarded to :class:`RecursiveChunker`.

    Note:
        Exact token counts require ``tiktoken`` (installed by
        ``windlass[rag]``). Without it, Windlass falls back to a calibrated
        character-ratio estimate, which is close enough for sizing but not for
        hard limits.

    Example:
        >>> chunks = TokenChunker(chunk_size=8, overlap=2).split_text(
        ...     "one two three four five six seven eight nine ten eleven twelve"
        ... )
        >>> len(chunks) >= 1
        True
    """

    provider_name = "token"
    unit = "token"

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        overlap: int | None = None,
        model: str = "gpt-4o",
        **config: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, overlap=overlap, model=model, **config)
        self.model = model

    def _size(self, text: str) -> int:
        """Measure ``text`` in tokens for the configured model."""
        return count_tokens(text, self.model)


def _split_by_tokens(text: str, limit: int) -> list[str]:
    """Cut ``text`` into pieces of at most ``limit`` tokens.

    Falls back to word-boundary slicing when ``tiktoken`` is unavailable, which
    keeps the output readable even without exact token counts.

    Args:
        text: The text to cut.
        limit: Maximum tokens per piece.

    Returns:
        The pieces, in order.
    """
    from windlass.core.lazy import is_available

    if is_available("tiktoken"):
        try:
            import tiktoken

            encoder = tiktoken.get_encoding("cl100k_base")
            ids = encoder.encode(text, disallowed_special=())
            return [encoder.decode(ids[i : i + limit]) for i in range(0, len(ids), limit)]
        except Exception:
            pass

    words = text.split(" ")
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) >= limit:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces
