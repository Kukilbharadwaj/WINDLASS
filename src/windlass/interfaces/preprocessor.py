r"""The preprocessor interface.

Preprocessors sit between loading and chunking. They clean, enrich, filter or
redact documents, and they compose: a pipeline runs them in order, each seeing
what the previous one produced.

A preprocessor may return **zero** documents (dropping the input — that is how
deduplication and language filtering work), **one** (the common case), or
**many** (splitting a spreadsheet into per-sheet documents).

Implementers override one coroutine, :meth:`Preprocessor.aprocess_one`.

Example:
    >>> from windlass.providers.preprocessors.clean import CleanPreprocessor
    >>> from windlass.core.types import Document
    >>> clean = CleanPreprocessor(min_length=0)
    >>> docs = clean.process([Document(content="a  \t b")])
    >>> docs[0].content
    'a b'
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from windlass.core.concurrency import gather_bounded, run_sync
from windlass.core.config import settings
from windlass.core.types import Document
from windlass.interfaces.base import Component

__all__ = ["Preprocessor", "PreprocessorChain"]


class Preprocessor(Component):
    """Abstract document preprocessor.

    Args:
        name: Component name for traces.
        **config: Preprocessor-specific options.

    Example:
        Implementing a preprocessor takes one method::

            class Shout(Preprocessor):
                provider_name = "shout"

                async def aprocess_one(self, document):
                    return [document.model_copy(
                        update={"content": document.content.upper()}
                    )]
    """

    kind = "preprocessor"
    provider_name: str = "preprocessor"

    @abc.abstractmethod
    async def aprocess_one(self, document: Document) -> list[Document]:
        """Transform a single document.

        Args:
            document: The document to process.

        Returns:
            Zero, one or many documents. Return ``[]`` to drop the input.

        Raises:
            IngestionError: When the document cannot be processed and dropping
                it silently would hide a real problem.
        """

    async def aprocess(
        self, documents: Sequence[Document], *, concurrency: int | None = None
    ) -> list[Document]:
        """Process a batch of documents concurrently.

        Args:
            documents: The documents to process.
            concurrency: Maximum simultaneous invocations. Defaults to the
                global ``max_concurrency`` setting.

        Returns:
            The flattened output, preserving input order.

        Performance:
            Order is preserved even though execution is concurrent, so a
            preprocessor that calls an LLM (metadata extraction, contextual
            enrichment) still yields a deterministic corpus.
        """
        if not documents:
            return []
        limit = concurrency or settings().max_concurrency
        batches = await gather_bounded([self.aprocess_one(doc) for doc in documents], limit=limit)
        return [doc for batch in batches for doc in batch]

    def process(self, documents: Sequence[Document] | Document) -> list[Document]:
        """Blocking :meth:`aprocess`.

        Args:
            documents: One document or a sequence of them.

        Returns:
            The processed documents.
        """
        items = [documents] if isinstance(documents, Document) else list(documents)
        return run_sync(self.aprocess(items))

    def __or__(self, other: Preprocessor) -> PreprocessorChain:
        """Compose two preprocessors with ``|``.

        Args:
            other: The preprocessor to run after this one.

        Returns:
            A :class:`PreprocessorChain` running both in order.

        Example:
            >>> from windlass.providers.preprocessors.clean import CleanPreprocessor
            >>> from windlass.providers.preprocessors.dedup import DeduplicatePreprocessor
            >>> chain = CleanPreprocessor() | DeduplicatePreprocessor()
            >>> len(chain.steps)
            2
        """
        left = self.steps if isinstance(self, PreprocessorChain) else [self]
        right = other.steps if isinstance(other, PreprocessorChain) else [other]
        return PreprocessorChain([*left, *right])


class PreprocessorChain(Preprocessor):
    """Runs several preprocessors in sequence.

    Each step sees the full output of the previous one, which matters for
    corpus-level steps like deduplication that need to compare documents against
    each other rather than in isolation.

    Args:
        steps: The preprocessors to run, in order.
        name: Component name for traces.

    Attributes:
        steps: The configured steps.

    Example:
        >>> from windlass.providers.preprocessors.clean import CleanPreprocessor
        >>> chain = PreprocessorChain([CleanPreprocessor(min_length=0)])
        >>> from windlass.core.types import Document
        >>> chain.process([Document(content="  hi  ")])[0].content
        'hi'
    """

    provider_name = "chain"

    def __init__(self, steps: Sequence[Preprocessor] = (), *, name: str | None = None) -> None:
        super().__init__(name=name or "chain")
        self.steps: list[Preprocessor] = list(steps)

    def add(self, step: Preprocessor) -> PreprocessorChain:
        """Append a step and return ``self`` for chaining."""
        self.steps.append(step)
        return self

    async def aprocess_one(self, document: Document) -> list[Document]:
        """Run every step against a single document."""
        return await self.aprocess([document])

    async def aprocess(
        self, documents: Sequence[Document], *, concurrency: int | None = None
    ) -> list[Document]:
        """Run every step in order over the whole batch.

        Args:
            documents: The documents to process.
            concurrency: Forwarded to each step.

        Returns:
            The output of the final step. Short-circuits to ``[]`` as soon as a
            step drops everything.
        """
        current = list(documents)
        for step in self.steps:
            if not current:
                break
            current = await step.aprocess(current, concurrency=concurrency)
        return current

    def describe(self) -> dict[str, Any]:
        """Return a summary including each step."""
        return {**super().describe(), "steps": [s.describe() for s in self.steps]}

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"PreprocessorChain({' | '.join(s.name for s in self.steps)})"
