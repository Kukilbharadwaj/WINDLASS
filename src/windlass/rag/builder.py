"""The fluent RAG builder.

This is the API most people will actually use::

    rag = (
        Windlass.rag()
        .loader("pdf")
        .preprocessor()
        .chunker("semantic")
        .embedding("huggingface")
        .vectordb("pinecone")
        .retriever("hybrid")
        .guardrails()
        .observe("langfuse")
    )

    rag.ingest("./documents")
    print(rag.ask("Explain transformers"))

Three properties make it work:

**Nothing is constructed until it is needed.** Each call records intent; the
pipeline is assembled on the first ``ingest`` or ``ask``. So you can reconfigure
freely, and a builder that names Pinecone does not import Pinecone until you use
it.

**Cross-component wiring is automatic.** The semantic chunker needs the
embedder, the vector retriever needs the embedder *and* the store, hybrid search
needs a BM25 leg over the same corpus, FAISS needs the embedding
dimensionality. :meth:`RAGBuilder.build` resolves that graph so you never wire
it by hand.

**Every argument accepts three forms.** A registry name (``"semantic"``), a
configured instance (``MyChunker(...)``), or a factory. That is the whole
extensibility story: your component is indistinguishable from a built-in one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Self

from windlass.core.container import Container, root_container
from windlass.core.exceptions import ConfigurationError
from windlass.core.logging import get_logger
from windlass.core.types import RAGAnswer, SearchResult
from windlass.interfaces.vectordb import MetadataFilter
from windlass.rag.pipeline import DEFAULT_PROMPT, RAGPipeline

__all__ = ["RAGBuilder"]

_log = get_logger(__name__)


class RAGBuilder:
    """Fluent builder for a :class:`~windlass.rag.pipeline.RAGPipeline`.

    Every configuration method returns ``self``, and the pipeline methods
    (:meth:`ingest`, :meth:`ask`, :meth:`search`, ...) are forwarded, so a
    builder *is* a usable pipeline — no explicit ``.build()`` required.

    Args:
        container: Dependency container. Defaults to a child of the process root,
            so application-wide bindings are inherited.

    Attributes:
        container: The container backing this builder.

    Example:
        Level 1 — defaults all the way down, no dependencies, no API keys::

            rag = Windlass.rag()
            rag.ingest_text("Windlass unifies the AI ecosystem.")
            print(rag.ask("What does Windlass do?"))

        Level 2 — configured::

            rag = (Windlass.rag()
                   .llm("openai", model="gpt-4o-mini", temperature=0)
                   .embedding("huggingface", model="BAAI/bge-small-en-v1.5")
                   .chunker("recursive", chunk_size=800, overlap=120)
                   .vectordb("faiss", persist_path="./index")
                   .retriever("hybrid", weights=[0.6, 0.4])
                   .top_k(8))

        Level 3 — reach past the abstraction::

            index = rag.native_store()      # the raw faiss.Index
            client = rag.native_llm()       # the raw openai.AsyncOpenAI
    """

    def __init__(self, container: Container | None = None) -> None:
        self.container = container or root_container().scope()
        self._specs: dict[str, tuple[Any, dict[str, Any]]] = {}
        self._options: dict[str, Any] = {
            "prompt": DEFAULT_PROMPT,
            "top_k": 5,
            "max_context_tokens": 6000,
            "include_sources": True,
            "answer_without_context": False,
            "batch_size": 64,
        }
        self._score_threshold: float | None = None
        self._pipeline: RAGPipeline | None = None

    # ------------------------------------------------------------------
    # Component selection
    # ------------------------------------------------------------------
    def llm(self, spec: Any = None, /, **config: Any) -> Self:
        """Choose the generation model.

        Args:
            spec: Registry name (``"openai"``, ``"anthropic"``, ``"ollama"``,
                ``"fake"``), an :class:`~windlass.interfaces.llm.LLM` instance, or
                a factory.
            **config: Provider options — ``model``, ``temperature``,
                ``max_tokens``, ``api_key``, ``base_url``, ...

        Returns:
            ``self``.

        Example:
            >>> from windlass import Windlass
            >>> _ = Windlass.rag().llm("fake", responses=["hi"])
        """
        return self._set("llm", spec, config)

    def embedding(self, spec: Any = None, /, **config: Any) -> Self:
        """Choose the embedding model.

        The same model must embed your corpus and your queries — Windlass uses
        this one for both, which is why swapping it means re-ingesting.

        Args:
            spec: Registry name (``"huggingface"``, ``"openai"``, ``"hash"``),
                an :class:`~windlass.interfaces.embedding.Embedder`, or a factory.
            **config: Provider options — ``model``, ``batch_size``, ``device``, ...

        Returns:
            ``self``.
        """
        return self._set("embedding", spec, config)

    def vectordb(self, spec: Any = None, /, **config: Any) -> Self:
        """Choose the vector store.

        Args:
            spec: Registry name (``"memory"``, ``"faiss"``, ``"chroma"``,
                ``"pinecone"``), a :class:`~windlass.interfaces.vectordb.VectorStore`,
                or a factory.
            **config: Store options — ``collection``, ``persist_path``,
                ``metric``, ``namespace``, ... ``dimensions`` is filled in from
                the embedding model when you omit it.

        Returns:
            ``self``.
        """
        return self._set("vectordb", spec, config)

    def chunker(self, spec: Any = None, /, **config: Any) -> Self:
        """Choose the chunking strategy.

        The highest-leverage knob in a RAG system: it decides what the retriever
        can find and what the model gets to read.

        Args:
            spec: Registry name (``"recursive"``, ``"semantic"``, ``"token"``,
                ``"markdown"``, ``"code"``, ``"parent_child"``), a
                :class:`~windlass.interfaces.chunker.Chunker`, or a factory.
            **config: Strategy options — ``chunk_size``, ``overlap``,
                ``threshold``, ``parent_size``, ...

        Returns:
            ``self``.

        Example:
            >>> from windlass import Windlass
            >>> _ = Windlass.rag().chunker("recursive", chunk_size=1000, overlap=200)
        """
        return self._set("chunker", spec, config)

    def retriever(self, spec: Any = None, /, **config: Any) -> Self:
        """Choose the retrieval strategy.

        Args:
            spec: Registry name (``"vector"``, ``"bm25"``, ``"hybrid"``,
                ``"contextual"``), a :class:`~windlass.interfaces.retriever.Retriever`,
                or a factory.
            **config: Strategy options — ``top_k``, ``diversity``, ``weights``,
                ``rerank``, ...

        Returns:
            ``self``.

        Note:
            ``"hybrid"`` builds a dense leg plus a BM25 leg over the same corpus
            automatically. Pass ``retrievers=[...]`` to fuse your own set.
        """
        return self._set("retriever", spec, config)

    def loader(self, spec: Any = None, /, **config: Any) -> Self:
        """Pin a document loader.

        Args:
            spec: Registry name (``"pdf"``, ``"docx"``, ``"web"``, ...), a
                :class:`~windlass.interfaces.loader.Loader`, or a factory.
            **config: Loader options.

        Returns:
            ``self``.

        Note:
            Optional. With no loader configured, Windlass picks one per file from
            the extension, which is what you want for a mixed corpus.
        """
        return self._set("loader", spec, config)

    def preprocessor(self, spec: Any = None, /, **config: Any) -> Self:
        """Add a preprocessing stage.

        Called repeatedly, stages compose into a chain in call order.

        Args:
            spec: Registry name (``"clean"``, ``"pii"``, ``"dedup"``,
                ``"language"``, ``"metadata"``, ``"tables"``, ``"ocr"``), a
                :class:`~windlass.interfaces.preprocessor.Preprocessor`, or a
                factory. Omit it for the default cleaner.
            **config: Preprocessor options.

        Returns:
            ``self``.

        Example:
            >>> from windlass import Windlass
            >>> _ = (Windlass.rag()
            ...      .preprocessor("clean", min_length=50)
            ...      .preprocessor("dedup", threshold=0.95)
            ...      .preprocessor("pii", action="redact"))
        """
        chain = self._specs.setdefault("preprocessor_chain", ([], {}))[0]
        chain.append((spec if spec is not None else "clean", config))
        return self

    def memory(self, spec: Any = None, /, **config: Any) -> Self:
        """Attach conversation memory, making ``ask`` multi-turn aware.

        Args:
            spec: Registry name (``"window"``, ``"buffer"``, ``"summary"``,
                ``"vector"``), a :class:`~windlass.interfaces.memory.Memory`, or a
                factory. Omit it for a sliding window.
            **config: Memory options.

        Returns:
            ``self``.
        """
        return self._set("memory", spec if spec is not None else "window", config)

    def guardrails(self, spec: Any = None, /, **config: Any) -> Self:
        """Enable input and output guardrails.

        Args:
            spec: Registry name (``"rules"``, ``"nemo"``), a
                :class:`~windlass.interfaces.guardrail.Guardrail`, or a factory.
                Omit it for the dependency-free rule guardrail.
            **config: Policy options — ``pii``, ``injection``, ``banned_words``,
                ``on_violation``, ...

        Returns:
            ``self``.

        Example:
            >>> from windlass import Windlass
            >>> _ = Windlass.rag().guardrails(pii=True, on_violation="redact")
        """
        return self._set("guardrail", spec if spec is not None else "rules", config)

    def observe(self, spec: Any = None, /, **config: Any) -> Self:
        """Enable tracing.

        Args:
            spec: Registry name (``"console"``, ``"langfuse"``, ``"langsmith"``,
                ``"memory"``), a :class:`~windlass.interfaces.tracer.Tracer`, or a
                factory. Omit it for console tracing.
            **config: Tracer options — ``project``, ``tags``, ``show_io``, ...

        Returns:
            ``self``.
        """
        return self._set("tracer", spec if spec is not None else "console", config)

    # ------------------------------------------------------------------
    # Behaviour options
    # ------------------------------------------------------------------
    def prompt(self, template: str) -> Self:
        """Override the answer prompt template.

        Args:
            template: A format string containing ``{context}`` and ``{question}``.

        Returns:
            ``self``.

        Raises:
            ConfigurationError: When a placeholder is missing.
        """
        if "{context}" not in template or "{question}" not in template:
            raise ConfigurationError("The prompt template must contain {context} and {question}.")
        self._options["prompt"] = template
        return self._invalidate()

    def top_k(self, k: int) -> Self:
        """Set how many chunks to retrieve per question.

        Args:
            k: Number of chunks. Must be positive.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``k`` is not positive.
        """
        if k <= 0:
            raise ValueError("top_k must be positive")
        self._options["top_k"] = k
        return self._invalidate()

    def max_context_tokens(self, tokens: int) -> Self:
        """Cap how much retrieved context reaches the prompt.

        Args:
            tokens: Token ceiling. Lower-ranked chunks are dropped first.

        Returns:
            ``self``.
        """
        self._options["max_context_tokens"] = tokens
        return self._invalidate()

    def strict(self, enabled: bool = True) -> Self:
        """Refuse to answer when retrieval finds nothing.

        Removes the model's opportunity to fill an empty context with plausible
        invention, which is more effective than asking it not to.

        Args:
            enabled: True to refuse, False to answer from parametric knowledge.

        Returns:
            ``self``.

        Note:
            Dense retrieval returns the *nearest* vectors, however far away they
            are — so it is almost never empty, and strict mode alone will rarely
            fire. Pair it with :meth:`min_score` to make "found nothing relevant"
            mean what you expect:

            ```python
            Windlass.rag().strict().min_score(0.25)
            ```
        """
        self._options["answer_without_context"] = not enabled
        return self._invalidate()

    def min_score(self, threshold: float | None) -> Self:
        """Discard retrieval hits scoring below ``threshold``.

        This is what turns strict mode into a real relevance gate. Without it,
        dense retrieval hands back its ``k`` nearest neighbours no matter how
        unrelated they are, and the model is asked to answer from noise.

        Args:
            threshold: Minimum score to keep, or ``None`` to keep everything.
                The right value depends on the retriever: cosine similarity is
                bounded in ``[-1, 1]`` (0.2-0.4 is a common floor), BM25 is
                unbounded, and hybrid fusion produces small reciprocal-rank
                scores. Measure on your own corpus rather than guessing.

        Returns:
            ``self``.

        Example:
            >>> from windlass import Windlass
            >>> rag = Windlass.rag().retriever("vector").strict().min_score(0.3)
            >>> rag.ingest_text("Paris is the capital of France.")
            1
            >>> rag.ask("Explain quantum chromodynamics.").metadata["no_context"]
            True
        """
        self._score_threshold = threshold
        return self._invalidate()

    def batch_size(self, size: int) -> Self:
        """Set the ingestion batch size.

        Args:
            size: Chunks embedded and indexed per batch.

        Returns:
            ``self``.
        """
        self._options["batch_size"] = size
        return self._invalidate()

    def bind(self, key: str, factory: Any) -> Self:
        """Bind an arbitrary dependency into this builder's container.

        The escape hatch for wiring something Windlass has no named slot for.

        Args:
            key: Binding key.
            factory: A factory callable, or a ready-made instance.

        Returns:
            ``self``.
        """
        if callable(factory):
            self.container.bind(key, factory)
        else:
            self.container.bind_instance(key, factory)
        return self._invalidate()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self) -> RAGPipeline:
        """Resolve every component and construct the pipeline.

        Called automatically on first use. Call it explicitly when you want
        construction errors (a missing API key, an uninstalled extra) to surface
        at startup rather than on the first request.

        Returns:
            The assembled :class:`~windlass.rag.pipeline.RAGPipeline`.

        Raises:
            ConfigurationError: When a component cannot be constructed.
            MissingDependencyError: When a chosen provider's extra is not installed.

        Example:
            >>> from windlass import Windlass
            >>> pipeline = Windlass.rag().build()
            >>> pipeline.top_k
            5
        """
        if self._pipeline is not None:
            return self._pipeline

        from windlass.core.config import settings

        cfg = settings()

        llm = self._resolve("llm", cfg.default_llm)
        embedder = self._resolve("embedding", cfg.default_embedding)
        vectorstore = self._resolve_store(embedder, cfg.default_vectordb)
        chunker = self._resolve_chunker(embedder, cfg.default_chunker)
        retriever = self._resolve_retriever(
            embedder, vectorstore, llm, chunker, cfg.default_retriever
        )

        self._pipeline = RAGPipeline(
            llm=llm,
            embedder=embedder,
            vectorstore=vectorstore,
            retriever=retriever,
            chunker=chunker,
            loader=self._resolve_optional("loader"),
            preprocessor=self._resolve_preprocessors(llm),
            guardrail=self._resolve_optional("guardrail"),
            memory=self._resolve_memory(embedder),
            tracer=self._resolve_optional("tracer"),
            **self._options,
        )
        return self._pipeline

    @property
    def pipeline(self) -> RAGPipeline:
        """The built pipeline, constructing it on first access."""
        return self.build()

    # ------------------------------------------------------------------
    # Pipeline delegation — a builder behaves like the thing it builds
    # ------------------------------------------------------------------
    def ingest(self, source: Any, **kwargs: Any) -> int:
        """Ingest a source. See :meth:`~windlass.rag.pipeline.RAGPipeline.ingest`."""
        return self.build().ingest(source, **kwargs)

    async def aingest(self, source: Any, **kwargs: Any) -> int:
        """Async :meth:`ingest`."""
        return await self.build().aingest(source, **kwargs)

    def ingest_text(self, text: str, **kwargs: Any) -> int:
        """Ingest a raw string. See :meth:`~windlass.rag.pipeline.RAGPipeline.ingest_text`."""
        return self.build().ingest_text(text, **kwargs)

    async def aingest_text(self, text: str, **kwargs: Any) -> int:
        """Async :meth:`ingest_text`."""
        return await self.build().aingest_text(text, **kwargs)

    def ingest_documents(self, documents: Sequence[Any]) -> int:
        """Ingest pre-loaded documents."""
        return self.build().ingest_documents(documents)

    def ask(self, question: str, **kwargs: Any) -> RAGAnswer:
        """Answer a question. See :meth:`~windlass.rag.pipeline.RAGPipeline.ask`."""
        return self.build().ask(question, **kwargs)

    async def aask(self, question: str, **kwargs: Any) -> RAGAnswer:
        """Async :meth:`ask`."""
        return await self.build().aask(question, **kwargs)

    def stream_ask(self, question: str, **kwargs: Any) -> Iterator[str]:
        """Stream an answer. See :meth:`~windlass.rag.pipeline.RAGPipeline.stream_ask`."""
        return self.build().stream_ask(question, **kwargs)

    def astream_ask(self, question: str, **kwargs: Any) -> AsyncIterator[str]:
        """Async :meth:`stream_ask`."""
        return self.build().astream_ask(question, **kwargs)

    def search(
        self, query: str, k: int | None = None, *, filters: MetadataFilter | None = None
    ) -> SearchResult:
        """Retrieve without generating. See :meth:`~windlass.rag.pipeline.RAGPipeline.search`."""
        return self.build().search(query, k, filters=filters)

    async def asearch(
        self, query: str, k: int | None = None, *, filters: MetadataFilter | None = None
    ) -> SearchResult:
        """Async :meth:`search`."""
        return await self.build().asearch(query, k, filters=filters)

    def evaluate(self, questions: Sequence[Any], **kwargs: Any) -> Any:
        """Evaluate the pipeline. See :meth:`~windlass.rag.pipeline.RAGPipeline.evaluate`."""
        return self.build().evaluate(questions, **kwargs)

    async def aevaluate(self, questions: Sequence[Any], **kwargs: Any) -> Any:
        """Async :meth:`evaluate`."""
        return await self.build().aevaluate(questions, **kwargs)

    def count(self) -> int:
        """Return how many chunks are indexed."""
        return self.build().count()

    def save(self, directory: str) -> Any:
        """Persist the index. See :meth:`~windlass.rag.pipeline.RAGPipeline.save`."""
        return self.build().save(directory)

    def load(self, directory: str) -> Self:
        """Restore a persisted index and return ``self``."""
        self.build().load(directory)
        return self

    def close(self) -> None:
        """Release every component's resources."""
        if self._pipeline is not None:
            self._pipeline.close()

    # -- Level 3 ----------------------------------------------------------
    def native_llm(self) -> Any:
        """Return the model provider's own client."""
        return self.build().native_llm()

    def native_store(self) -> Any:
        """Return the vector store's own handle."""
        return self.build().native_store()

    def native_retriever(self) -> Any:
        """Return the retriever's underlying object."""
        return self.build().native_retriever()

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description of the wired pipeline."""
        return self.build().describe()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _set(self, kind: str, spec: Any, config: dict[str, Any]) -> Self:
        """Record a component choice and invalidate any built pipeline."""
        self._specs[kind] = (spec, config)
        return self._invalidate()

    def _invalidate(self) -> Self:
        """Discard the built pipeline so the next use rebuilds it."""
        self._pipeline = None
        return self

    def _resolve(self, kind: str, default: str) -> Any:
        """Resolve a required component.

        Precedence is explicit builder call, then a container binding, then the
        configured default. The binding has to be consulted here: passing the
        default name straight to ``component()`` would resolve it from the
        registry and silently ignore whatever the container was told to use.
        """
        spec, config = self._specs.get(kind, (None, {}))
        if spec is None and not config and self.container.has(kind):
            return self.container.component(kind)
        return self.container.component(kind, spec if spec is not None else default, **config)

    def _resolve_optional(self, kind: str) -> Any:
        """Resolve a component the user asked for, or one the container supplies."""
        if kind not in self._specs:
            return self.container.component(kind) if self.container.has(kind) else None
        spec, config = self._specs[kind]
        return self.container.component(kind, spec, **config)

    def _resolve_store(self, embedder: Any, default: str) -> Any:
        """Resolve the vector store, supplying the embedding dimensionality.

        FAISS and Pinecone must know the dimensionality before the first write.
        Asking the embedder saves the user from repeating a number that is
        already determined by their model choice.
        """
        spec, config = self._specs.get("vectordb", (None, {}))
        name = spec if spec is not None else default
        if isinstance(name, str) and "dimensions" not in config:
            try:
                config = {**config, "dimensions": embedder.dimension()}
            except Exception as exc:
                _log.debug("Could not determine embedding dimensionality: %s", exc)
        return self.container.component("vectordb", name, **config)

    def _resolve_chunker(self, embedder: Any, default: str) -> Any:
        """Resolve the chunker, injecting the embedder for semantic chunking."""
        spec, config = self._specs.get("chunker", (None, {}))
        name = spec if spec is not None else default
        if name in {"semantic", "meaning"} and "embedder" not in config:
            config = {**config, "embedder": embedder}
        return self.container.component("chunker", name, **config)

    def _resolve_retriever(
        self, embedder: Any, store: Any, llm: Any, chunker: Any, default: str
    ) -> Any:
        """Resolve the retriever and wire its dependencies.

        This is where most of the builder's value lives. ``"hybrid"`` needs a
        dense leg and a lexical leg over the same corpus; ``"contextual"`` needs
        a base retriever and a model; parent-child chunking needs the retriever
        to expand children into parents. Doing that by hand is exactly the
        integration work Windlass exists to remove.
        """
        from windlass.providers.chunkers.hierarchical import ParentChildChunker
        from windlass.providers.retrievers.bm25 import BM25Retriever
        from windlass.providers.retrievers.vector import VectorRetriever

        spec, config = self._specs.get("retriever", (None, {}))
        name = spec if spec is not None else default
        top_k = self._options["top_k"]

        if not isinstance(name, str):
            # An already-constructed retriever cannot take construction config,
            # so the threshold is applied to the live object instead. Without
            # this, .min_score() combined with .retriever(instance) fails at
            # build time with an error that never mentions min_score.
            retriever = self.container.component("retriever", name, **config)
            if self._score_threshold is not None:
                retriever.score_threshold = self._score_threshold
            return retriever

        if self._score_threshold is not None:
            config = {"score_threshold": self._score_threshold, **config}

        hierarchical = isinstance(chunker, ParentChildChunker)

        def _dense(**extra: Any) -> VectorRetriever:
            return VectorRetriever(
                embedder=embedder,
                vectorstore=store,
                top_k=top_k,
                expand_parents=hierarchical,
                parent_source=chunker if hierarchical else None,
                **extra,
            )

        if name in {"vector", "dense", "semantic", "similarity"}:
            options = {"top_k": top_k, **config}
            options.setdefault("embedder", embedder)
            options.setdefault("vectorstore", store)
            options.setdefault("expand_parents", hierarchical)
            if hierarchical:
                options.setdefault("parent_source", chunker)
            return self.container.component("retriever", name, **options)

        if name in {"bm25", "keyword", "lexical", "sparse"}:
            return self.container.component("retriever", name, top_k=top_k, **config)

        if name in {"hybrid", "fusion", "rrf", "dense+sparse"}:
            options = {"top_k": top_k, **config}
            if "retrievers" not in options:
                options["retrievers"] = [
                    _dense(),
                    BM25Retriever(top_k=top_k, vectorstore=store),
                ]
            return self.container.component("retriever", name, **options)

        if name in {"contextual", "contextual-retrieval", "hyde"}:
            options = {"top_k": top_k, **config}
            options.setdefault("llm", llm)
            if "retriever" not in options:
                options["retriever"] = _dense()
            return self.container.component("retriever", name, **options)

        return self.container.component("retriever", name, top_k=top_k, **config)

    def _resolve_preprocessors(self, llm: Any) -> Any:
        """Build the preprocessor chain, injecting the model where needed."""
        entries = self._specs.get("preprocessor_chain", ([], {}))[0]
        if not entries:
            return None

        from windlass.interfaces.preprocessor import PreprocessorChain

        steps = []
        for spec, config in entries:
            options = dict(config)
            if spec in {"llm_metadata", "extract", "auto-metadata"}:
                options.setdefault("llm", llm)
            steps.append(self.container.component("preprocessor", spec, **options))
        return steps[0] if len(steps) == 1 else PreprocessorChain(steps)

    def _resolve_memory(self, embedder: Any) -> Any:
        """Resolve memory, injecting the embedder for semantic recall."""
        if "memory" not in self._specs:
            return None
        spec, config = self._specs["memory"]
        options = dict(config)
        if spec in {"vector", "long-term", "longterm", "semantic"}:
            options.setdefault("embedder", embedder)
        return self.container.component("memory", spec, **options)

    def __repr__(self) -> str:
        chosen = ", ".join(
            f"{kind}={spec!r}"
            for kind, (spec, _) in self._specs.items()
            if kind != "preprocessor_chain" and isinstance(spec, str)
        )
        state = "built" if self._pipeline is not None else "unbuilt"
        return f"RAGBuilder({chosen or 'defaults'}, {state})"
