"""Retrieval-augmented generation.

Start with :func:`windlass.Windlass.rag`, which returns a
:class:`~windlass.rag.builder.RAGBuilder`::

    from windlass import Windlass

    rag = Windlass.rag().chunker("semantic").retriever("hybrid")
    rag.ingest("./docs")
    print(rag.ask("How does authentication work?"))

The pieces:

* :class:`~windlass.rag.builder.RAGBuilder` — the fluent API, and the place where
  cross-component wiring is resolved.
* :class:`~windlass.rag.pipeline.RAGPipeline` — the assembled pipeline that does
  the work.
* :func:`~windlass.rag.loading.loader_for` — format auto-detection.
"""

from __future__ import annotations

from windlass.rag.builder import RAGBuilder
from windlass.rag.loading import AutoLoader, loader_for
from windlass.rag.pipeline import DEFAULT_PROMPT, NO_CONTEXT_ANSWER, RAGPipeline

__all__ = [
    "DEFAULT_PROMPT",
    "NO_CONTEXT_ANSWER",
    "AutoLoader",
    "RAGBuilder",
    "RAGPipeline",
    "loader_for",
]
