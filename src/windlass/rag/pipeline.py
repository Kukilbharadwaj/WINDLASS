"""The RAG pipeline.

A :class:`RAGPipeline` wires the eight components a retrieval-augmented system
needs — loader, preprocessor, chunker, embedder, vector store, retriever,
guardrail, model — and exposes two operations:

* :meth:`RAGPipeline.aingest` — load, clean, chunk, embed, index.
* :meth:`RAGPipeline.aask` — guard, retrieve, prompt, generate, guard.

You rarely construct one directly; :class:`~windlass.rag.builder.RAGBuilder`
assembles it and injects the cross-component dependencies (the semantic chunker
needs the embedder, the vector retriever needs both the embedder and the store)
so you never wire them by hand.

Every stage is traced, so ``.observe('console')`` shows you exactly where the
time went.

Example:
    >>> from windlass import Windlass
    >>> rag = Windlass.rag()
    >>> rag.ingest_text("Windlass is a modular AI application framework.")
    1
    >>> answer = rag.ask("What is Windlass?")
    >>> len(answer.contexts) >= 1
    True
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

from windlass.core.concurrency import iter_sync, run_sync
from windlass.core.exceptions import ConfigurationError, IngestionError, PipelineError
from windlass.core.logging import get_logger
from windlass.core.text import truncate_tokens
from windlass.core.types import Chunk, Document, RAGAnswer, ScoredChunk, SearchResult
from windlass.interfaces.chunker import Chunker
from windlass.interfaces.embedding import Embedder
from windlass.interfaces.guardrail import Guardrail
from windlass.interfaces.llm import LLM
from windlass.interfaces.loader import Loader
from windlass.interfaces.memory import Memory
from windlass.interfaces.preprocessor import Preprocessor
from windlass.interfaces.retriever import Retriever
from windlass.interfaces.tracer import NullTracer, Tracer
from windlass.interfaces.vectordb import MetadataFilter, VectorStore

__all__ = ["DEFAULT_PROMPT", "NO_CONTEXT_ANSWER", "RAGPipeline"]

_log = get_logger(__name__)

DEFAULT_PROMPT = """\
Answer the question using only the context below. The context is authoritative;
your own knowledge is not.

If the context does not contain the answer, say so plainly rather than guessing.
Cite the sources you used with their [n] markers.

<context>
{context}
</context>

Question: {question}

Answer:"""

#: Returned when retrieval finds nothing, instead of letting the model invent one.
NO_CONTEXT_ANSWER = "I could not find anything relevant in the indexed documents to answer that."


class RAGPipeline:
    """A fully wired retrieval-augmented generation pipeline.

    Args:
        llm: Model used to generate answers.
        embedder: Model used to embed chunks and queries.
        vectorstore: Where chunk vectors are stored.
        retriever: Retrieval strategy. Defaults to dense search over
            ``vectorstore``.
        chunker: Chunking strategy. Defaults to recursive.
        loader: Loader used by :meth:`aingest`. Defaults to format auto-detection.
        preprocessor: Optional cleaning/enrichment stage.
        guardrail: Optional input/output policy.
        memory: Optional conversation memory, making ``ask`` multi-turn aware.
        tracer: Observability backend.
        prompt: Prompt template. Must contain ``{context}`` and ``{question}``.
        top_k: How many chunks to retrieve per question.
        max_context_tokens: Ceiling on context placed in the prompt. Chunks are
            dropped from the tail until it fits, so the highest-ranked context
            always survives.
        include_sources: Number each context block and ask the model to cite them.
        answer_without_context: When False, retrieval finding nothing returns
            :data:`NO_CONTEXT_ANSWER` instead of calling the model. This is the
            single most effective anti-hallucination setting in a RAG system.
        batch_size: Chunks embedded and indexed per batch during ingestion.

    Attributes:
        llm: The generation model.
        retriever: The retrieval strategy.
        vectorstore: The vector store.

    Raises:
        ConfigurationError: When a required component is missing or the prompt
            template lacks its placeholders.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        embedder: Embedder,
        vectorstore: VectorStore,
        retriever: Retriever | None = None,
        chunker: Chunker | None = None,
        loader: Loader | None = None,
        preprocessor: Preprocessor | None = None,
        guardrail: Guardrail | None = None,
        memory: Memory | None = None,
        tracer: Tracer | None = None,
        prompt: str = DEFAULT_PROMPT,
        top_k: int = 5,
        max_context_tokens: int = 6000,
        include_sources: bool = True,
        answer_without_context: bool = False,
        batch_size: int = 64,
    ) -> None:
        if llm is None or embedder is None or vectorstore is None:
            raise ConfigurationError(
                "A RAG pipeline needs an llm, an embedder and a vector store.",
                hint="Build it with Windlass.rag(), which supplies working defaults.",
            )
        if "{context}" not in prompt or "{question}" not in prompt:
            raise ConfigurationError(
                "The RAG prompt template must contain {context} and {question}.",
                context={"prompt": prompt[:200]},
            )

        self.llm = llm
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.chunker = chunker
        self.loader = loader
        self.preprocessor = preprocessor
        self.guardrail = guardrail
        self.memory = memory
        self.tracer = tracer or NullTracer()
        self.prompt = prompt
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.include_sources = include_sources
        self.answer_without_context = answer_without_context
        self.batch_size = batch_size

        if retriever is None:
            from windlass.providers.retrievers.vector import VectorRetriever

            retriever = VectorRetriever(embedder=embedder, vectorstore=vectorstore, top_k=top_k)
        self.retriever = retriever

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    async def aingest(
        self,
        source: Any,
        *,
        metadata: dict[str, Any] | None = None,
        show_progress: bool = False,
    ) -> int:
        """Load, preprocess, chunk, embed and index a source.

        Args:
            source: A path, directory, URL, glob, byte payload, or a sequence of
                any of those. Directories are walked and each file routed to the
                right loader by extension.
            metadata: Extra metadata attached to every resulting chunk — a
                tenant id, a corpus name, an ingestion date.
            show_progress: Log a line per stage. Useful on long runs.

        Returns:
            How many chunks were indexed.

        Raises:
            IngestionError: When nothing could be loaded, or a stage fails.

        Performance:
            Loading, embedding and indexing are all batched and concurrent.
            Re-ingesting unchanged content is cheap: chunk ids are content
            hashes, so the store upserts rather than duplicating, and a
            configured embedding cache skips the model entirely.

        Example:
            >>> import asyncio, tempfile, pathlib
            >>> from windlass import Windlass
            >>> folder = pathlib.Path(tempfile.mkdtemp())
            >>> _ = (folder / "a.txt").write_text("Retrieval augmented generation.")
            >>> asyncio.run(Windlass.rag().aingest(folder)) >= 1
            True
        """
        started = time.perf_counter()
        with self.tracer.span("ingest", kind="ingestion", inputs=str(source)) as span:
            documents = await self._load(source, metadata)
            if show_progress:
                _log.info("Loaded %d documents", len(documents))

            documents = await self._preprocess(documents)
            if not documents:
                raise IngestionError(
                    "Every document was dropped during preprocessing.",
                    hint="Relax the preprocessor — a min_length or language filter "
                    "is probably too strict.",
                )
            if show_progress:
                _log.info("Preprocessed to %d documents", len(documents))

            chunks = await self._chunk(documents)
            if show_progress:
                _log.info("Produced %d chunks", len(chunks))

            indexed = await self._index(chunks)
            span.set_output({"documents": len(documents), "chunks": indexed})
            span.set_metadata(elapsed_ms=round((time.perf_counter() - started) * 1000, 1))

        if show_progress:
            _log.info("Indexed %d chunks in %.1fs", indexed, time.perf_counter() - started)
        return indexed

    def ingest(
        self, source: Any, *, metadata: dict[str, Any] | None = None, show_progress: bool = False
    ) -> int:
        """Blocking :meth:`aingest`."""
        return run_sync(self.aingest(source, metadata=metadata, show_progress=show_progress))

    async def aingest_documents(self, documents: Sequence[Document]) -> int:
        """Ingest already-loaded documents, skipping the loader.

        Args:
            documents: Documents to preprocess, chunk, embed and index.

        Returns:
            How many chunks were indexed.
        """
        processed = await self._preprocess(list(documents))
        chunks = await self._chunk(processed)
        return await self._index(chunks)

    def ingest_documents(self, documents: Sequence[Document]) -> int:
        """Blocking :meth:`aingest_documents`."""
        return run_sync(self.aingest_documents(documents))

    async def aingest_text(
        self, text: str, *, source: str | None = None, metadata: dict[str, Any] | None = None
    ) -> int:
        """Ingest a raw string.

        Args:
            text: The text to index.
            source: Label recorded as the document's source, used for citation.
            metadata: Extra metadata.

        Returns:
            How many chunks were indexed.

        Example:
            >>> import asyncio
            >>> from windlass import Windlass
            >>> asyncio.run(Windlass.rag().aingest_text("Some knowledge.")) >= 1
            True
        """
        document = Document(content=text, source=source, metadata=dict(metadata or {}))
        return await self.aingest_documents([document])

    def ingest_text(
        self, text: str, *, source: str | None = None, metadata: dict[str, Any] | None = None
    ) -> int:
        """Blocking :meth:`aingest_text`."""
        return run_sync(self.aingest_text(text, source=source, metadata=metadata))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    async def asearch(
        self, query: str, k: int | None = None, *, filters: MetadataFilter | None = None
    ) -> SearchResult:
        """Retrieve without generating an answer.

        Useful for building a search UI, for debugging why an answer was wrong,
        and for evaluating retrieval quality independently of generation.

        Args:
            query: The search query.
            k: Override for :attr:`top_k`.
            filters: Metadata constraints.

        Returns:
            The ranked results.
        """
        with self.tracer.span("retrieve", kind="retriever", inputs=query) as span:
            result = await self.retriever.aretrieve(query, k or self.top_k, filters=filters)
            span.set_output([h.chunk.content[:200] for h in result.hits])
            span.set_metadata(hits=len(result.hits))
        return result

    def search(
        self, query: str, k: int | None = None, *, filters: MetadataFilter | None = None
    ) -> SearchResult:
        """Blocking :meth:`asearch`."""
        return run_sync(self.asearch(query, k, filters=filters))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    async def aask(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: MetadataFilter | None = None,
        thread_id: str = "default",
        **llm_kwargs: Any,
    ) -> RAGAnswer:
        """Answer a question from the indexed corpus.

        The full path: input guardrail, retrieval, context assembly, generation,
        output guardrail, memory update.

        Args:
            question: The user's question.
            k: Override for :attr:`top_k`.
            filters: Metadata constraints on retrieval.
            thread_id: Conversation thread, when memory is configured.
            **llm_kwargs: Per-call model overrides (``temperature``, ...).

        Returns:
            The answer, the contexts it was based on, token usage and latency.

        Raises:
            GuardrailViolation: When a guardrail blocks the question or answer.
            PipelineError: When generation fails.

        Performance:
            One retrieval (embedding + store query) plus one model call. Context
            is truncated to :attr:`max_context_tokens` from the tail, so a large
            ``top_k`` never blows the context window.

        Example:
            >>> import asyncio
            >>> from windlass import Windlass
            >>> rag = Windlass.rag()
            >>> _ = rag.ingest_text("The Eiffel Tower is in Paris.")
            >>> asyncio.run(rag.aask("Where is the Eiffel Tower?")).question
            'Where is the Eiffel Tower?'
        """
        started = time.perf_counter()
        with self.tracer.span("rag.ask", kind="chain", inputs=question) as span:
            safe_question = await self._guard(question, "input")
            result = await self.asearch(safe_question, k, filters=filters)

            if not result.hits and not self.answer_without_context:
                answer = RAGAnswer(
                    answer=NO_CONTEXT_ANSWER,
                    question=question,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    metadata={"no_context": True},
                )
                span.set_output(answer.answer)
                return answer

            context, used = self._build_context(result.hits)
            messages = await self._build_messages(safe_question, context, thread_id)

            with self.tracer.span("generate", kind="llm", inputs=safe_question) as llm_span:
                try:
                    completion = await self.llm.acomplete(messages, **llm_kwargs)
                except Exception as exc:
                    raise PipelineError(
                        f"Answer generation failed: {exc}",
                        context={"question": question[:200]},
                    ) from exc
                llm_span.set_output(completion.content)
                llm_span.set_usage(completion.usage)
                llm_span.set_metadata(model=completion.model)

            safe_answer = await self._guard(completion.content, "output")
            await self._remember(question, safe_answer, thread_id)

            answer = RAGAnswer(
                answer=safe_answer,
                question=question,
                contexts=used,
                usage=completion.usage,
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata={
                    "model": completion.model,
                    "retrieved": len(result.hits),
                    "used_contexts": len(used),
                    "retrieval_ms": round(result.latency_ms, 1),
                },
            )
            span.set_output(safe_answer)
            span.set_usage(completion.usage)
        return answer

    def ask(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: MetadataFilter | None = None,
        thread_id: str = "default",
        **llm_kwargs: Any,
    ) -> RAGAnswer:
        """Blocking :meth:`aask`."""
        return run_sync(
            self.aask(question, k=k, filters=filters, thread_id=thread_id, **llm_kwargs)
        )

    async def astream_ask(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: MetadataFilter | None = None,
        thread_id: str = "default",
        **llm_kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream an answer token by token.

        Retrieval completes first (it must — the context shapes the prompt), then
        generation streams. Guardrails run on the assembled answer after the
        stream finishes, so a blocking output policy cannot un-send text that has
        already reached the user; use ``redact`` semantics or non-streaming
        generation when output blocking must be authoritative.

        Args:
            question: The user's question.
            k: Override for :attr:`top_k`.
            filters: Metadata constraints.
            thread_id: Conversation thread.
            **llm_kwargs: Per-call model overrides.

        Yields:
            Text fragments as the model produces them.
        """
        safe_question = await self._guard(question, "input")
        result = await self.asearch(safe_question, k, filters=filters)

        if not result.hits and not self.answer_without_context:
            yield NO_CONTEXT_ANSWER
            return

        context, _ = self._build_context(result.hits)
        messages = await self._build_messages(safe_question, context, thread_id)

        collected: list[str] = []
        async for event in self.llm.astream(messages, **llm_kwargs):
            if event.type == "text" and event.delta:
                collected.append(event.delta)
                yield event.delta

        answer = "".join(collected)
        if self.guardrail is not None:
            await self.guardrail.avalidate(answer, stage="output")
        await self._remember(question, answer, thread_id)

    def stream_ask(self, question: str, **kwargs: Any) -> Iterator[str]:
        """Blocking :meth:`astream_ask`.

        Args:
            question: The user's question.
            **kwargs: Forwarded to :meth:`astream_ask`.

        Yields:
            Text fragments.
        """
        return iter_sync(self.astream_ask(question, **kwargs))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    async def aevaluate(
        self,
        questions: Sequence[str] | Sequence[dict[str, Any]],
        *,
        evaluator: Any = None,
        metrics: Sequence[str] | None = None,
    ) -> Any:
        """Answer a question set and score the results.

        Args:
            questions: Question strings, or dicts with ``question`` and an
                optional ``reference`` ground-truth answer.
            evaluator: An :class:`~windlass.interfaces.evaluator.Evaluator`. When
                omitted, the built-in evaluator is used with this pipeline's
                model as the judge.
            metrics: Metric names, passed to the default evaluator.

        Returns:
            An :class:`~windlass.core.types.EvaluationReport`.

        Example:
            >>> import asyncio
            >>> from windlass import Windlass
            >>> rag = Windlass.rag()
            >>> _ = rag.ingest_text("Paris is the capital of France.")
            >>> report = asyncio.run(rag.aevaluate(
            ...     [{"question": "Capital of France?", "reference": "Paris"}],
            ...     metrics=["answer_relevancy_lexical"],
            ... ))
            >>> report.samples
            1
        """
        from windlass.interfaces.evaluator import EvalSample

        if evaluator is None:
            from windlass.providers.evaluation.builtin import BuiltinEvaluator

            evaluator = BuiltinEvaluator(metrics=metrics, llm=self.llm)

        samples: list[EvalSample] = []
        for item in questions:
            if isinstance(item, str):
                question, reference = item, ""
            else:
                question = str(item.get("question", ""))
                reference = str(item.get("reference", "") or item.get("ground_truth", ""))
            answer = await self.aask(question)
            samples.append(EvalSample.from_answer(answer, reference=reference))

        with self.tracer.span("evaluate", kind="evaluation") as span:
            report = await evaluator.aevaluate(samples)
            span.set_output(report.summary)
        return report

    def evaluate(
        self,
        questions: Sequence[str] | Sequence[dict[str, Any]],
        *,
        evaluator: Any = None,
        metrics: Sequence[str] | None = None,
    ) -> Any:
        """Blocking :meth:`aevaluate`."""
        return run_sync(self.aevaluate(questions, evaluator=evaluator, metrics=metrics))

    # ------------------------------------------------------------------
    # Level 3: native access
    # ------------------------------------------------------------------
    def native_llm(self) -> Any:
        """Return the model provider's own client."""
        return self.llm.native()

    def native_store(self) -> Any:
        """Return the vector store's own handle (a FAISS index, a Chroma collection)."""
        return self.vectorstore.native()

    def native_retriever(self) -> Any:
        """Return the retriever's underlying object."""
        return self.retriever.native()

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description of every wired component.

        Useful for logging what a deployment is actually running, and for
        diffing two environments that behave differently.

        Returns:
            A dict keyed by component role.
        """
        parts: dict[str, Any] = {
            "llm": self.llm.describe(),
            "embedder": self.embedder.describe(),
            "vectorstore": self.vectorstore.describe(),
            "retriever": self.retriever.describe(),
            "top_k": self.top_k,
            "max_context_tokens": self.max_context_tokens,
        }
        for label, component in (
            ("chunker", self.chunker),
            ("loader", self.loader),
            ("preprocessor", self.preprocessor),
            ("guardrail", self.guardrail),
            ("memory", self.memory),
            ("tracer", self.tracer),
        ):
            if component is not None:
                parts[label] = component.describe()
        return parts

    async def acount(self) -> int:
        """Return how many chunks are indexed.

        Reads the vector store, falling back to the retriever's own index for
        purely lexical pipelines where no vectors are stored.
        """
        count = await self.vectorstore.acount()
        if count:
            return count
        own = getattr(self.retriever, "chunks", None)
        return len(own) if own is not None else 0

    def count(self) -> int:
        """Blocking :meth:`acount`."""
        return run_sync(self.acount())

    async def aclose(self) -> None:
        """Release every component's resources."""
        for component in (
            self.llm,
            self.embedder,
            self.vectorstore,
            self.retriever,
            self.tracer,
        ):
            close = getattr(component, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception as exc:
                    _log.debug("Closing %s failed: %s", component, exc)

    def close(self) -> None:
        """Blocking :meth:`aclose`."""
        run_sync(self.aclose())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _load(self, source: Any, metadata: dict[str, Any] | None) -> list[Document]:
        """Route a source to the right loader and read it."""
        if isinstance(source, Document):
            return [source]
        if isinstance(source, Sequence) and source and isinstance(source[0], Document):
            return list(source)

        loader = self.loader
        if loader is None:
            from windlass.rag.loading import loader_for

            loader = loader_for(source, metadata=metadata)
        elif metadata:
            loader = type(loader)(**{**loader.config, "metadata": metadata})

        documents = await loader.aload(source)
        if not documents:
            raise IngestionError(
                f"No documents were produced from {source!r}.",
                hint="Check the path exists and that a loader for that format is "
                'installed (pip install "windlass[loaders]").',
            )
        if metadata:
            for document in documents:
                document.metadata.update(metadata)
        return documents

    async def _preprocess(self, documents: list[Document]) -> list[Document]:
        """Run the preprocessing stage, if configured."""
        if self.preprocessor is None:
            return documents
        with self.tracer.span("preprocess", kind="ingestion") as span:
            processed = await self.preprocessor.aprocess(documents)
            span.set_metadata(before=len(documents), after=len(processed))
        return processed

    async def _chunk(self, documents: list[Document]) -> list[Chunk]:
        """Run the chunking stage."""
        if self.chunker is None:
            from windlass.providers.chunkers.recursive import RecursiveChunker

            self.chunker = RecursiveChunker()
        with self.tracer.span("chunk", kind="ingestion") as span:
            chunks = await self.chunker.achunk(documents)
            span.set_metadata(documents=len(documents), chunks=len(chunks))
        if not chunks:
            raise IngestionError(
                "Chunking produced nothing.",
                hint="The documents may be empty, or min_chunk_size may be larger "
                "than the documents themselves.",
            )
        return chunks

    async def _index(self, chunks: list[Chunk]) -> int:
        """Hand chunks to the retriever, which owns indexing.

        Indexing belongs to the retriever, not the pipeline: a lexical retriever
        builds an inverted index, a dense one embeds and writes vectors, a
        contextual one rewrites chunk text *before* either happens. Centralising
        it here would force the pipeline to know which is which.

        Args:
            chunks: Chunks to index.

        Returns:
            How many chunks were indexed.
        """
        from windlass.core.concurrency import batched

        total = 0
        with self.tracer.span("index", kind="ingestion") as span:
            for batch in batched(chunks, self.batch_size):
                total += await self.retriever.aindex(batch)
            span.set_metadata(indexed=total)
        return total

    def _build_context(self, hits: list[ScoredChunk]) -> tuple[str, list[ScoredChunk]]:
        """Assemble the context block, respecting the token budget.

        Chunks are added highest-ranked first and the tail is dropped when the
        budget is reached, so the best context is never the part that gets cut.

        Args:
            hits: Ranked retrieval hits.

        Returns:
            A ``(context_text, used_hits)`` pair.
        """
        blocks: list[str] = []
        used: list[ScoredChunk] = []
        remaining = self.max_context_tokens

        for position, hit in enumerate(hits, start=1):
            source = hit.chunk.metadata.get("source") or hit.chunk.metadata.get("url") or ""
            page = hit.chunk.metadata.get("page")
            label = f"[{position}]"
            if self.include_sources and source:
                citation = f"{source}" + (f" p.{page}" if page else "")
                header = f"{label} {citation}"
            else:
                header = label

            block = f"{header}\n{hit.chunk.content}"
            cost = len(block) // 4 + 8
            if used and cost > remaining:
                break
            blocks.append(block)
            used.append(hit)
            remaining -= cost

        context = "\n\n".join(blocks)
        if not used and hits:
            # A single chunk larger than the whole budget: truncate rather than
            # answer with nothing.
            context = truncate_tokens(
                hits[0].chunk.content, self.max_context_tokens, suffix="\n[…]"
            )
            used = [hits[0]]
        return context, used

    async def _build_messages(self, question: str, context: str, thread_id: str) -> Any:
        """Render the prompt, prepending conversation memory when configured."""
        rendered = self.prompt.format(context=context, question=question)
        if self.memory is None:
            return rendered

        from windlass.core.types import Message

        history = await self.memory.aget(thread_id=thread_id, query=question)
        return [*history, Message.user(rendered)] if history else rendered

    async def _guard(self, content: str, stage: str) -> str:
        """Apply the guardrail policy for one stage."""
        if self.guardrail is None:
            return content
        with self.tracer.span(f"guardrail.{stage}", kind="guardrail") as span:
            result = await self.guardrail.avalidate(content, stage=stage)
            span.set_metadata(modified=result != content)
        return result

    async def _remember(self, question: str, answer: str, thread_id: str) -> None:
        """Record the exchange in memory, if configured."""
        if self.memory is None:
            return
        from windlass.core.types import Message

        await self.memory.aadd(
            [Message.user(question), Message.assistant(answer)], thread_id=thread_id
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist the index to disk, when the store supports it.

        Args:
            directory: Where to write. Created if absent.

        Returns:
            The directory written to.

        Raises:
            PipelineError: When the configured store cannot persist.

        Example:
            >>> import tempfile
            >>> from windlass import Windlass
            >>> rag = Windlass.rag()
            >>> _ = rag.ingest_text("something worth keeping")
            >>> rag.save(tempfile.mkdtemp()).is_dir()
            True
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        saver = getattr(self.vectorstore, "save", None)
        if not callable(saver):
            raise PipelineError(
                f"{type(self.vectorstore).__name__} does not support saving to disk.",
                hint="Chroma and Pinecone persist automatically; use "
                "vectordb('memory', persist_path=...) or 'faiss' for explicit saves.",
            )
        # The in-memory store saves to a single JSON file; FAISS writes a
        # directory of index plus sidecar.
        if self.vectorstore.provider_name == "memory":
            saver(target / f"{self.vectorstore.collection}.json")
        else:
            saver(target)
        return target

    def load(self, directory: str | Path) -> RAGPipeline:
        """Restore an index previously written by :meth:`save`.

        Args:
            directory: Directory containing the saved index.

        Returns:
            ``self``, so the call chains.

        Raises:
            PipelineError: When the configured store cannot load.
        """
        source = Path(directory)
        loader = getattr(self.vectorstore, "load", None)
        if not callable(loader):
            raise PipelineError(
                f"{type(self.vectorstore).__name__} does not support loading from disk."
            )
        if self.vectorstore.provider_name == "memory":
            loader(source / f"{self.vectorstore.collection}.json")
        else:
            loader(source)

        # Rebuild any lexical index from the restored corpus.
        rebuild = getattr(self.vectorstore, "all_chunks", None)
        if callable(rebuild) and getattr(self.retriever, "requires_index", False):
            run_sync(self.retriever.aindex(rebuild()))
        return self

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(llm={self.llm.name!r}, retriever={self.retriever.name!r}, "
            f"store={self.vectorstore.name!r}, top_k={self.top_k})"
        )
