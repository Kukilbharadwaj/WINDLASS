"""Contextual retrieval and query transformation.

Two techniques that attack retrieval failure from opposite ends:

**Contextual chunk enrichment** (index time). A chunk that reads "revenue grew
3%" is unfindable — grew from what, for whom, when? Before embedding, a small
model writes one situating sentence using the surrounding document, and that
sentence is prepended. Anthropic's published results put the retrieval-failure
reduction at roughly 35-50%.

**Query transformation** (query time). ``HyDE`` has the model write a
*hypothetical answer* and embeds that instead of the question, because answers
live closer to answers in embedding space than questions do. Multi-query
expansion runs several rephrasings and fuses the results.

Both cost model calls. Enrichment pays once at ingestion; transformation pays on
every query, so measure before enabling it in a latency-sensitive path.

Example:
    >>> from windlass.providers.llm.fake import FakeLLM
    >>> from windlass.providers.retrievers.bm25 import BM25Retriever
    >>> from windlass.core.types import Chunk
    >>> base = BM25Retriever()
    >>> r = ContextualRetriever(retriever=base, llm=FakeLLM(responses=["Q3 revenue."]))
    >>> r.index([Chunk(content="revenue grew 3%", metadata={"source": "10-K"})])
    1
    >>> "Q3 revenue." in base.chunks[next(iter(base.chunks))].content
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from windlass.core.concurrency import gather_bounded
from windlass.core.config import settings
from windlass.core.exceptions import ConfigurationError
from windlass.core.registry import register
from windlass.core.types import Chunk, ScoredChunk
from windlass.core.vectors import reciprocal_rank_fusion
from windlass.interfaces.llm import LLM
from windlass.interfaces.retriever import Retriever
from windlass.interfaces.vectordb import MetadataFilter

__all__ = ["CONTEXT_PROMPT", "HYDE_PROMPT", "MULTI_QUERY_PROMPT", "ContextualRetriever"]

CONTEXT_PROMPT = """\
Here is a document:
<document>
{document}
</document>

Here is a chunk taken from it:
<chunk>
{chunk}
</chunk>

Write one short sentence that situates this chunk within the document, so that a
search engine could find it. State the subject, the time period and the section
it belongs to where those are known. Answer with the sentence only — no preamble.
"""

HYDE_PROMPT = """\
Write a short, factual passage that would answer this question. Invent plausible
specifics if you must; the passage is used only as a search probe, never shown to
anyone.

Question: {query}

Passage:"""

MULTI_QUERY_PROMPT = """\
Rewrite the search query below in {n} different ways. Vary the vocabulary and
phrasing so that together they cover synonyms and related terminology. Answer
with one rewrite per line and nothing else.

Query: {query}
"""


@register.retriever(
    "contextual",
    aliases=("contextual-retrieval", "hyde"),
    description="LLM-enriched chunks at index time, optional query rewriting at search time.",
)
class ContextualRetriever(Retriever):
    """Wraps another retriever with LLM-driven enrichment and query rewriting.

    Args:
        retriever: The retriever doing the actual search. Required.
        llm: Model used for enrichment and rewriting. Required. Use a small,
            cheap model — this is a summarisation task, not reasoning.
        enrich: Prepend a generated situating sentence to each chunk at index
            time.
        transform: Query-time strategy — ``"none"``, ``"hyde"`` or
            ``"multi_query"``.
        n_queries: How many rewrites to generate for ``multi_query``.
        context_window: Characters of surrounding document shown to the model
            when enriching. Larger gives better context and costs more.
        top_k: Default number of results.
        **config: Forwarded to :class:`~windlass.interfaces.retriever.Retriever`.

    Raises:
        ConfigurationError: When ``retriever`` or ``llm`` is missing.
        ValueError: For an unknown ``transform``.

    Performance:
        Enrichment issues one model call per chunk, run with bounded concurrency.
        For a 10k-chunk corpus that is a real but one-off cost; the embedder's
        cache makes re-ingestion cheap.
    """

    provider_name = "contextual"

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        llm: LLM | None = None,
        enrich: bool = True,
        transform: str = "none",
        n_queries: int = 3,
        context_window: int = 4000,
        top_k: int = 5,
        **config: Any,
    ) -> None:
        if retriever is None:
            raise ConfigurationError(
                "Contextual retrieval wraps another retriever.",
                hint="Pass retriever=Windlass.retriever('vector', ...).",
            )
        if llm is None:
            raise ConfigurationError(
                "Contextual retrieval needs a model to write the context.",
                hint="Pass llm=Windlass.llm('openai', model='gpt-4o-mini').",
            )
        if transform not in {"none", "hyde", "multi_query"}:
            raise ValueError("transform must be 'none', 'hyde' or 'multi_query'")

        super().__init__(top_k=top_k, **config)
        self.retriever = retriever
        self.llm = llm
        self.enrich = enrich
        self.transform = transform
        self.n_queries = max(2, n_queries)
        self.context_window = context_window

    def native(self) -> Any:
        """Return the wrapped retriever's native handle."""
        return self.retriever.native()

    # -- indexing ---------------------------------------------------------
    async def aindex(self, chunks: Sequence[Chunk]) -> int:
        """Enrich chunks and hand them to the wrapped retriever.

        Args:
            chunks: Chunks to index.

        Returns:
            How many chunks were indexed.
        """
        if not chunks:
            return 0
        prepared = await self._enrich(list(chunks)) if self.enrich else list(chunks)
        return await self.retriever.aindex(prepared)

    async def _enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        """Prepend a generated situating sentence to each chunk."""
        documents = _group_by_document(chunks)

        async def _context(chunk: Chunk) -> str:
            document = documents.get(chunk.document_id, chunk.content)
            prompt = CONTEXT_PROMPT.format(
                document=document[: self.context_window], chunk=chunk.content
            )
            completion = await self.llm.acomplete(prompt, max_tokens=120)
            return completion.content.strip()

        contexts = await gather_bounded(
            [_context(c) for c in chunks],
            limit=settings().max_concurrency,
            return_exceptions=True,
        )

        enriched: list[Chunk] = []
        for chunk, context in zip(chunks, contexts, strict=True):
            if isinstance(context, BaseException) or not context:
                if isinstance(context, BaseException):
                    self._log.warning("Context generation failed for %s: %s", chunk.id, context)
                enriched.append(chunk)
                continue
            enriched.append(
                chunk.model_copy(
                    update={
                        "content": f"{context}\n\n{chunk.content}",
                        "metadata": {
                            **chunk.metadata,
                            "generated_context": context,
                            "original_content": chunk.content,
                        },
                    }
                )
            )
        return enriched

    # -- retrieval --------------------------------------------------------
    async def aretrieve_chunks(
        self,
        query: str,
        k: int,
        *,
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Optionally rewrite the query, then delegate to the wrapped retriever.

        Args:
            query: The search query.
            k: How many results to return.
            filters: Metadata constraints.
            **kwargs: Forwarded to the wrapped retriever.

        Returns:
            Ranked hits.
        """
        if self.transform == "none":
            result = await self.retriever.aretrieve(query, k, filters=filters, **kwargs)
            return result.hits

        if self.transform == "hyde":
            probe = await self._hyde(query)
            result = await self.retriever.aretrieve(probe, k, filters=filters, **kwargs)
            return result.hits

        return await self._multi_query(query, k, filters, kwargs)

    async def _hyde(self, query: str) -> str:
        """Generate a hypothetical answer to use as the search probe."""
        try:
            completion = await self.llm.acomplete(HYDE_PROMPT.format(query=query), max_tokens=200)
            passage = completion.content.strip()
        except Exception as exc:
            self._log.warning("HyDE generation failed, using the raw query: %s", exc)
            return query
        # Keep the original terms too: the hypothesis may drift off-topic.
        return f"{query}\n\n{passage}" if passage else query

    async def _multi_query(
        self,
        query: str,
        k: int,
        filters: MetadataFilter | None,
        kwargs: dict[str, Any],
    ) -> list[ScoredChunk]:
        """Search several rewrites of the query and fuse the results."""
        queries = [query, *await self._rewrites(query)]
        results = await gather_bounded(
            [self.retriever.aretrieve(q, k * 2, filters=filters, **kwargs) for q in queries],
            limit=settings().max_concurrency,
            return_exceptions=True,
        )

        rankings: list[list[str]] = []
        catalogue: dict[str, Chunk] = {}
        for result in results:
            if isinstance(result, BaseException):
                continue
            ids = []
            for hit in result.hits:
                catalogue.setdefault(hit.chunk.id, hit.chunk)
                ids.append(hit.chunk.id)
            rankings.append(ids)

        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings)
        return [
            ScoredChunk(chunk=catalogue[cid], score=score, rank=rank, retriever=self.name)
            for rank, (cid, score) in enumerate(list(fused.items())[:k], start=1)
            if cid in catalogue
        ]

    async def _rewrites(self, query: str) -> list[str]:
        """Ask the model for alternative phrasings of the query."""
        try:
            completion = await self.llm.acomplete(
                MULTI_QUERY_PROMPT.format(query=query, n=self.n_queries), max_tokens=300
            )
        except Exception as exc:
            self._log.warning("Query expansion failed: %s", exc)
            return []
        lines = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in completion.content.splitlines()
        ]
        return [line for line in lines if line][: self.n_queries]

    def __repr__(self) -> str:
        return (
            f"ContextualRetriever(enrich={self.enrich}, transform={self.transform!r}, "
            f"base={self.retriever.name})"
        )


def _group_by_document(chunks: list[Chunk]) -> dict[str, str]:
    """Reconstruct each document's text from its chunks.

    Enrichment needs the surrounding document, but the retriever only ever sees
    chunks. Concatenating a document's chunks in index order is a faithful
    enough reconstruction for a one-sentence summary.

    Args:
        chunks: The chunks being indexed.

    Returns:
        A ``{document_id: text}`` mapping.
    """
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)
    return {
        document_id: "\n\n".join(c.content for c in sorted(items, key=lambda c: c.index))
        for document_id, items in grouped.items()
    }
