"""Parent-child (small-to-big) chunking.

There is a tension at the heart of chunk sizing: **small chunks retrieve better**
(a focused embedding matches a focused query) but **large chunks answer better**
(the model needs surrounding context to reason).

Parent-child chunking refuses the trade-off. It indexes small child chunks for
retrieval, but each child carries a pointer to the larger parent it came from.
At query time the pipeline swaps matched children for their parents before
building the prompt — so you retrieve with precision and generate with context.

Example:
    >>> text = " ".join(f"Sentence number {i}." for i in range(40))
    >>> chunker = ParentChildChunker(parent_size=400, child_size=100)
    >>> children = chunker.chunk_text(text)
    >>> all(c.parent_id for c in children)
    True
    >>> len(chunker.parents) < len(children)
    True
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from windlass.core.registry import register
from windlass.core.types import Chunk, Document
from windlass.interfaces.chunker import Chunker
from windlass.providers.chunkers.recursive import RecursiveChunker

__all__ = ["ParentChildChunker"]


@register.chunker(
    "parent_child",
    aliases=("parent-child", "small-to-big", "hierarchical"),
    description="Indexes small chunks that expand to their larger parent at query time.",
)
class ParentChildChunker(Chunker):
    """Two-level chunker producing indexable children and retrievable parents.

    Args:
        parent_size: Character length of parent chunks — what the model reads.
        child_size: Character length of child chunks — what gets embedded.
        child_overlap: Overlap between children within one parent. ``None``
            derives it from ``child_size``, so shrinking the children alone
            never produces an invalid configuration.
        parent_chunker: Strategy used to cut parents. Defaults to recursive.
        child_chunker: Strategy used to cut children. Defaults to recursive.
        **config: Forwarded to :class:`~windlass.interfaces.chunker.Chunker`.

    Attributes:
        parents: Every parent chunk produced so far, keyed by id. The pipeline
            reads this to expand retrieval hits.

    Raises:
        ValueError: If ``child_size`` is not smaller than ``parent_size``.

    Note:
        Only children are written to the vector store. Parents live in this
        chunker (and are persisted alongside the index by the RAG pipeline), so
        the store holds no redundant copies of your corpus.

    Thread safety:
        :attr:`parents` is guarded by a lock, so concurrent ingestion is safe.
    """

    provider_name = "parent_child"
    unit = "char"

    def __init__(
        self,
        *,
        parent_size: int = 2000,
        child_size: int = 400,
        child_overlap: int | None = None,
        parent_chunker: Chunker | None = None,
        child_chunker: Chunker | None = None,
        **config: Any,
    ) -> None:
        if child_size >= parent_size:
            raise ValueError(
                f"child_size ({child_size}) must be smaller than parent_size ({parent_size}); "
                "otherwise there is nothing to expand."
            )
        config.pop("chunk_size", None)
        config.pop("overlap", None)
        super().__init__(chunk_size=parent_size, overlap=0, **config)
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent_chunker = parent_chunker or RecursiveChunker(
            chunk_size=parent_size, overlap=0, min_chunk_size=0
        )
        self.child_chunker = child_chunker or RecursiveChunker(
            chunk_size=child_size, overlap=child_overlap, min_chunk_size=0
        )
        #: The resolved child overlap — derived from ``child_size`` when the
        #: caller did not supply one.
        self.child_overlap = self.child_chunker.overlap
        self.parents: dict[str, Chunk] = {}
        self._lock = threading.RLock()

    def split_text(self, text: str) -> list[str]:
        """Return the child strings.

        Provided for interface completeness; :meth:`achunk_document` is what
        actually builds the parent links.

        Args:
            text: The text to split.

        Returns:
            Child chunk strings.
        """
        children: list[str] = []
        for parent in self.parent_chunker.split_text(text):
            children.extend(self.child_chunker.split_text(parent))
        return children

    async def achunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into children, recording their parents.

        Args:
            document: The document to split.

        Returns:
            Child chunks. Each has ``parent_id`` set and carries
            ``parent_content`` in its metadata for stores that cannot look
            parents up later.
        """
        parents_text = self.parent_chunker.split_text(document.content)
        children: list[Chunk] = []
        cursor = 0
        child_index = 0

        for parent_index, parent_text in enumerate(parents_text):
            start = document.content.find(parent_text, cursor)
            if start < 0:
                start = cursor
            else:
                cursor = start + len(parent_text)

            parent = Chunk(
                content=parent_text,
                document_id=document.id,
                index=parent_index,
                start_char=start,
                end_char=start + len(parent_text),
                metadata={
                    **document.metadata,
                    "document_id": document.id,
                    "chunk_index": parent_index,
                    "level": "parent",
                    "chunker": self.name,
                    **({"source": document.source} if document.source else {}),
                },
            )
            with self._lock:
                self.parents[parent.id] = parent

            for child_text in self.child_chunker.split_text(parent_text):
                offset = parent_text.find(child_text)
                child_start = start + (offset if offset >= 0 else 0)
                children.append(
                    Chunk(
                        content=child_text,
                        document_id=document.id,
                        index=child_index,
                        parent_id=parent.id,
                        start_char=child_start,
                        end_char=child_start + len(child_text),
                        metadata={
                            **document.metadata,
                            "document_id": document.id,
                            "chunk_index": child_index,
                            "parent_id": parent.id,
                            "parent_content": parent_text,
                            "level": "child",
                            "chunker": self.name,
                            **({"source": document.source} if document.source else {}),
                        },
                    )
                )
                child_index += 1

        return children

    def parent_of(self, chunk: Chunk) -> Chunk | None:
        """Return a child's parent chunk.

        Args:
            chunk: The child chunk.

        Returns:
            The parent, or ``None`` when the chunk has no parent or the parent
            is not held by this chunker (for example after a process restart —
            in that case the pipeline falls back to ``parent_content`` metadata).
        """
        if not chunk.parent_id:
            return None
        with self._lock:
            return self.parents.get(chunk.parent_id)

    def expand(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Swap child chunks for their parents, de-duplicated.

        Two children of the same parent collapse to a single parent, which is
        exactly what you want in a prompt — the same paragraph should not appear
        twice.

        Args:
            chunks: Retrieved chunks, typically children.

        Returns:
            Parents where available, originals otherwise, preserving input order.

        Example:
            >>> from windlass.core.types import Chunk
            >>> chunker = ParentChildChunker(parent_size=200, child_size=50)
            >>> kids = chunker.chunk_text("word " * 100)
            >>> expanded = chunker.expand(kids[:3])
            >>> len(expanded) <= 3
            True
        """
        seen: set[str] = set()
        expanded: list[Chunk] = []
        for chunk in chunks:
            parent = self.parent_of(chunk)
            if parent is None and chunk.metadata.get("parent_content"):
                parent = chunk.model_copy(
                    update={
                        "content": str(chunk.metadata["parent_content"]),
                        "id": chunk.parent_id or chunk.id,
                    }
                )
            target = parent or chunk
            if target.id in seen:
                continue
            seen.add(target.id)
            expanded.append(target)
        return expanded

    def clear(self) -> None:
        """Forget every recorded parent."""
        with self._lock:
            self.parents.clear()

    def __repr__(self) -> str:
        return (
            f"ParentChildChunker(parent_size={self.parent_size}, "
            f"child_size={self.child_size}, parents={len(self.parents)})"
        )
