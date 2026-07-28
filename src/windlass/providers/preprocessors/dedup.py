"""Deduplication and table extraction.

Real corpora are full of duplicates: the same policy PDF exported three times,
boilerplate footers repeated on every page, near-identical release notes.
Duplicates waste embedding spend, and worse, they crowd the top-k so a single
document can fill the entire context window.

``DeduplicatePreprocessor`` removes exact duplicates by content hash and
near-duplicates by MinHash-style shingle similarity — no dependencies, one pass.

Example:
    >>> from windlass.core.types import Document
    >>> docs = [Document(content="same text here"), Document(content="Same  text here")]
    >>> len(DeduplicatePreprocessor().process(docs))
    1
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from windlass.core.registry import register
from windlass.core.text import text_hash
from windlass.core.types import Document
from windlass.interfaces.preprocessor import Preprocessor

__all__ = ["DeduplicatePreprocessor", "TableExtractPreprocessor", "jaccard", "shingles"]

_TABLE_RE = re.compile(r"(?:^\|.*\|\s*$\n?){2,}", re.MULTILINE)


def shingles(text: str, size: int = 5) -> set[int]:
    """Return the set of hashed word n-grams in ``text``.

    Shingling is what makes near-duplicate detection work: two documents that
    differ by a sentence share almost all their shingles, while two unrelated
    documents share almost none.

    Args:
        text: The text to shingle.
        size: Words per shingle. Smaller catches looser similarity; 5 is a good
            default for prose.

    Returns:
        Hashed shingles. Short texts return one shingle covering everything.

    Example:
        >>> len(shingles("a b c d e f", size=3)) > 0
        True
    """
    words = text.lower().split()
    if len(words) < size:
        return {hash(" ".join(words))} if words else set()
    return {
        int.from_bytes(
            hashlib.blake2b(" ".join(words[i : i + size]).encode(), digest_size=8).digest(),
            "big",
        )
        for i in range(len(words) - size + 1)
    }


def jaccard(left: set[int], right: set[int]) -> float:
    """Return the Jaccard similarity of two sets, in ``[0, 1]``.

    Args:
        left: First set.
        right: Second set.

    Returns:
        ``|A ∩ B| / |A ∪ B|``, or ``0.0`` when both are empty.

    Example:
        >>> jaccard({1, 2, 3}, {2, 3, 4})
        0.5
    """
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


@register.preprocessor(
    "dedup",
    aliases=("deduplicate", "unique"),
    description="Removes exact and near-duplicate documents (no dependencies).",
)
class DeduplicatePreprocessor(Preprocessor):
    """Removes duplicate and near-duplicate documents.

    Args:
        threshold: Jaccard similarity at or above which two documents are
            considered duplicates. ``1.0`` disables near-duplicate detection and
            keeps only exact-hash matching, which is much faster.
        shingle_size: Words per shingle for near-duplicate comparison.
        keep: ``"first"`` keeps the earliest occurrence, ``"longest"`` keeps the
            most complete version — usually what you want when the same document
            was exported at different truncation levels.
        min_words: Documents shorter than this skip near-duplicate comparison,
            since short texts collide spuriously.
        **config: Forwarded to :class:`~windlass.interfaces.preprocessor.Preprocessor`.

    Raises:
        ValueError: For an unknown ``keep`` policy.

    Performance:
        Exact deduplication is ``O(n)``. Near-duplicate detection compares each
        document against the survivors, which is ``O(n·k)`` where ``k`` is the
        number of unique documents — fine for tens of thousands, and the reason
        ``threshold=1.0`` exists for much larger corpora.

    Note:
        Unlike most preprocessors this one is *corpus-level*: it must see the
        whole batch at once, so it overrides ``aprocess`` rather than
        ``aprocess_one``.
    """

    provider_name = "dedup"

    def __init__(
        self,
        *,
        threshold: float = 0.9,
        shingle_size: int = 5,
        keep: str = "first",
        min_words: int = 20,
        **config: Any,
    ) -> None:
        if keep not in {"first", "longest"}:
            raise ValueError("keep must be 'first' or 'longest'")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        super().__init__(**config)
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.keep = keep
        self.min_words = min_words

    async def aprocess_one(self, document: Document) -> list[Document]:
        """Pass a single document through — deduplication needs the batch."""
        return [document]

    async def aprocess(
        self, documents: Sequence[Document], *, concurrency: int | None = None
    ) -> list[Document]:
        """Remove duplicates from a batch.

        Args:
            documents: The documents to deduplicate.
            concurrency: Ignored — this step is inherently sequential.

        Returns:
            The surviving documents, in input order, each tagged with how many
            duplicates it absorbed.
        """
        if not documents:
            return []

        ordered = list(documents)
        if self.keep == "longest":
            ordered.sort(key=lambda d: len(d.content), reverse=True)

        survivors: list[Document] = []
        signatures: list[set[int]] = []
        seen_hashes: dict[str, int] = {}
        duplicate_counts: dict[int, int] = {}

        for document in ordered:
            digest = text_hash(document.content)
            if digest in seen_hashes:
                duplicate_counts[seen_hashes[digest]] = (
                    duplicate_counts.get(seen_hashes[digest], 0) + 1
                )
                continue

            if self.threshold < 1.0 and len(document.content.split()) >= self.min_words:
                signature = shingles(document.content, self.shingle_size)
                match = next(
                    (
                        index
                        for index, other in enumerate(signatures)
                        if jaccard(signature, other) >= self.threshold
                    ),
                    None,
                )
                if match is not None:
                    duplicate_counts[match] = duplicate_counts.get(match, 0) + 1
                    continue
                signatures.append(signature)
            else:
                signatures.append(set())

            seen_hashes[digest] = len(survivors)
            survivors.append(document)

        removed = len(ordered) - len(survivors)
        if removed:
            self._log.info(
                "Deduplication removed %d of %d documents (%.0f%%).",
                removed,
                len(ordered),
                100 * removed / len(ordered),
            )

        return [
            document.model_copy(
                update={
                    "metadata": {
                        **document.metadata,
                        "duplicates_removed": duplicate_counts.get(index, 0),
                    }
                }
            )
            for index, document in enumerate(survivors)
        ]


@register.preprocessor(
    "tables",
    aliases=("table-extract",),
    description="Splits Markdown tables into their own documents.",
)
class TableExtractPreprocessor(Preprocessor):
    r"""Separates tables from surrounding prose.

    Tables and prose want different chunk sizes and retrieve on different
    signals. Splitting them lets a table stay whole (a half table is useless)
    while the prose around it is chunked normally.

    Args:
        mode: ``"split"`` emits tables as separate documents and leaves the
            prose behind; ``"annotate"`` keeps one document and records how many
            tables it contains; ``"drop"`` removes tables entirely.
        min_rows: Ignore tables with fewer rows than this.
        **config: Forwarded to :class:`~windlass.interfaces.preprocessor.Preprocessor`.

    Raises:
        ValueError: For an unknown ``mode``.

    Note:
        Operates on Markdown-style pipe tables, which is what the XLSX, DOCX and
        HTML loaders emit. Tables in a PDF's raw text layer are not reliably
        detectable and are left alone.

    Example:
        >>> from windlass.core.types import Document
        >>> md = "Intro.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nOutro."
        >>> docs = TableExtractPreprocessor().process([Document(content=md)])
        >>> sum(1 for d in docs if d.metadata.get("content_type") == "table")
        1
    """

    provider_name = "tables"

    def __init__(self, *, mode: str = "split", min_rows: int = 2, **config: Any) -> None:
        if mode not in {"split", "annotate", "drop"}:
            raise ValueError("mode must be 'split', 'annotate' or 'drop'")
        super().__init__(**config)
        self.mode = mode
        self.min_rows = min_rows

    async def aprocess_one(self, document: Document) -> list[Document]:
        """Find tables and apply the configured mode.

        Args:
            document: The document to inspect.

        Returns:
            One or more documents depending on the mode.
        """
        tables = [
            m
            for m in _TABLE_RE.finditer(document.content)
            if m.group(0).strip().count("\n") + 1 >= self.min_rows
        ]
        if not tables:
            return [document]

        if self.mode == "annotate":
            return [
                document.model_copy(
                    update={"metadata": {**document.metadata, "table_count": len(tables)}}
                )
            ]

        prose = _TABLE_RE.sub("\n", document.content).strip()
        outputs: list[Document] = []
        if prose:
            outputs.append(
                document.model_copy(
                    update={
                        "content": prose,
                        "metadata": {
                            **document.metadata,
                            "content_type": "prose",
                            "table_count": len(tables),
                        },
                    }
                )
            )

        if self.mode == "split":
            for index, match in enumerate(tables):
                outputs.append(
                    Document(
                        content=match.group(0).strip(),
                        metadata={
                            **document.metadata,
                            "content_type": "table",
                            "table_index": index,
                            "parent_document_id": document.id,
                        },
                        source=document.source,
                        mimetype="text/markdown",
                    )
                )
        return outputs or [document]
