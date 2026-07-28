"""The ``Windlass`` facade — one entry point for the whole framework.

Everything starts here::

    from windlass import Windlass

    rag = Windlass.rag()                 # a RAG pipeline builder
    agent = Windlass.agent()             # an agent builder
    llm = Windlass.llm("openai")         # a single component
    Windlass.list("chunker")             # what is available

The class is a namespace of static factories, not something you instantiate. Its
job is discoverability: one import, and your editor's autocomplete shows you the
entire framework.
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import Any

from windlass.core.config import WindlassSettings, configure, settings
from windlass.core.container import Container, root_container
from windlass.core.registry import REGISTRY, ComponentSpec

__all__ = ["Windlass"]


class Windlass:
    """Factory namespace for every Windlass component and builder.

    Example:
        >>> from windlass import Windlass
        >>> rag = Windlass.rag()
        >>> rag.ingest_text("Windlass unifies the AI ecosystem behind one API.")
        1
        >>> "Windlass" in str(rag.ask("What does Windlass do?").contexts[0].content)
        True
    """

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    @staticmethod
    def rag(container: Container | None = None) -> Any:
        """Start building a RAG pipeline.

        Args:
            container: Dependency container. Defaults to a child of the process
                root, so application-wide bindings are inherited.

        Returns:
            A :class:`~windlass.rag.builder.RAGBuilder`.

        Example:
            >>> from windlass import Windlass
            >>> rag = (Windlass.rag()
            ...        .chunker("recursive", chunk_size=800)
            ...        .retriever("hybrid")
            ...        .top_k(8))
            >>> rag.ingest_text("Vector databases store embeddings.")
            1
        """
        from windlass.rag.builder import RAGBuilder

        return RAGBuilder(container)

    @staticmethod
    def agent(container: Container | None = None) -> Any:
        """Start building an agent.

        Args:
            container: Dependency container.

        Returns:
            An :class:`~windlass.agent.builder.AgentBuilder`.

        Example:
            >>> from windlass import Windlass
            >>> agent = Windlass.agent().llm("fake", responses=["Hello!"])
            >>> agent.run("Say hello").output
            'Hello!'
        """
        from windlass.agent.builder import AgentBuilder

        return AgentBuilder(container)

    @staticmethod
    def supervisor(
        agents: dict[str, Any],
        *,
        llm: Any = None,
        descriptions: dict[str, str] | None = None,
        **config: Any,
    ) -> Any:
        """Build a multi-agent supervisor.

        Args:
            agents: Specialists keyed by the name the supervisor will use.
            llm: The coordinating model. Defaults to the configured provider.
            descriptions: What each specialist is for. The supervisor routes on
                this text, so it is worth writing carefully.
            **config: Forwarded to :class:`~windlass.agent.supervisor.Supervisor`.

        Returns:
            A :class:`~windlass.agent.supervisor.Supervisor`.

        Example:
            >>> from windlass import Windlass
            >>> boss = Windlass.supervisor(
            ...     {"writer": Windlass.agent().llm("fake", responses=["Drafted."])},
            ...     llm=Windlass.llm("fake", responses=["Done."]),
            ... )
            >>> boss.broadcast("write something")["writer"].output
            'Drafted.'
        """
        from windlass.agent.supervisor import Supervisor

        return Supervisor(
            llm=llm or Windlass.llm(),
            agents={k: (v.build() if hasattr(v, "build") else v) for k, v in agents.items()},
            descriptions=descriptions,
            **config,
        )

    # ------------------------------------------------------------------
    # Individual components
    # ------------------------------------------------------------------
    @staticmethod
    def llm(name: str | None = None, /, **config: Any) -> Any:
        """Construct a language model.

        Args:
            name: Registry name, or a bare model id like ``"gpt-4o"``. Defaults
                to the configured provider.
            **config: Provider options.

        Returns:
            An :class:`~windlass.interfaces.llm.LLM`.

        Raises:
            ComponentNotFoundError: For an unknown provider.
            MissingDependencyError: When the provider's extra is not installed.

        Example:
            >>> from windlass import Windlass
            >>> Windlass.llm("fake", responses=["hi"]).complete("x").content
            'hi'
        """
        from windlass.agent.builder import _split_model

        if name:
            provider, model = _split_model(name)
            if model and "model" not in config:
                config["model"] = model
            name = provider
        return REGISTRY.create("llm", name or settings().default_llm, **config)

    @staticmethod
    def embedding(name: str | None = None, /, **config: Any) -> Any:
        """Construct an embedding model.

        Args:
            name: Registry name. Defaults to the configured provider.
            **config: Provider options.

        Returns:
            An :class:`~windlass.interfaces.embedding.Embedder`.
        """
        return REGISTRY.create("embedding", name or settings().default_embedding, **config)

    @staticmethod
    def vectordb(name: str | None = None, /, **config: Any) -> Any:
        """Construct a vector store.

        Args:
            name: Registry name. Defaults to the configured store.
            **config: Store options.

        Returns:
            A :class:`~windlass.interfaces.vectordb.VectorStore`.
        """
        return REGISTRY.create("vectordb", name or settings().default_vectordb, **config)

    @staticmethod
    def chunker(name: str | None = None, /, **config: Any) -> Any:
        """Construct a chunker.

        Args:
            name: Registry name. Defaults to the configured strategy.
            **config: Strategy options.

        Returns:
            A :class:`~windlass.interfaces.chunker.Chunker`.

        Example:
            >>> from windlass import Windlass
            >>> len(Windlass.chunker("recursive", chunk_size=20).split_text("a " * 40)) > 1
            True
        """
        return REGISTRY.create("chunker", name or settings().default_chunker, **config)

    @staticmethod
    def retriever(name: str | None = None, /, **config: Any) -> Any:
        """Construct a retriever.

        Args:
            name: Registry name. Defaults to the configured strategy.
            **config: Strategy options. Dense strategies need ``embedder`` and
                ``vectorstore``.

        Returns:
            A :class:`~windlass.interfaces.retriever.Retriever`.
        """
        return REGISTRY.create("retriever", name or settings().default_retriever, **config)

    @staticmethod
    def loader(name: str | None = None, /, **config: Any) -> Any:
        """Construct a document loader.

        Args:
            name: Registry name. Omit it for automatic format detection.
            **config: Loader options.

        Returns:
            A :class:`~windlass.interfaces.loader.Loader`.
        """
        if name is None:
            from windlass.rag.loading import AutoLoader

            return AutoLoader(**config)
        return REGISTRY.create("loader", name, **config)

    @staticmethod
    def preprocessor(name: str = "clean", /, **config: Any) -> Any:
        """Construct a preprocessor.

        Args:
            name: Registry name.
            **config: Preprocessor options.

        Returns:
            A :class:`~windlass.interfaces.preprocessor.Preprocessor`.
        """
        return REGISTRY.create("preprocessor", name, **config)

    @staticmethod
    def memory(name: str = "window", /, **config: Any) -> Any:
        """Construct a memory backend.

        Args:
            name: Registry name.
            **config: Memory options.

        Returns:
            A :class:`~windlass.interfaces.memory.Memory`.
        """
        return REGISTRY.create("memory", name, **config)

    @staticmethod
    def guardrail(name: str = "rules", /, **config: Any) -> Any:
        """Construct a guardrail.

        Args:
            name: Registry name.
            **config: Policy options.

        Returns:
            A :class:`~windlass.interfaces.guardrail.Guardrail`.

        Example:
            >>> from windlass import Windlass
            >>> Windlass.guardrail(on_violation="redact").validate("mail a@b.com")
            'mail [EMAIL]'
        """
        return REGISTRY.create("guardrail", name, **config)

    @staticmethod
    def evaluator(name: str = "builtin", /, **config: Any) -> Any:
        """Construct an evaluator.

        Args:
            name: Registry name — ``"builtin"``, ``"ragas"`` or ``"deepeval"``.
            **config: Evaluator options, including ``metrics`` and ``llm``.

        Returns:
            An :class:`~windlass.interfaces.evaluator.Evaluator`.
        """
        return REGISTRY.create("evaluator", name, **config)

    @staticmethod
    def tracer(name: str = "console", /, **config: Any) -> Any:
        """Construct a tracer.

        Args:
            name: Registry name.
            **config: Tracer options.

        Returns:
            A :class:`~windlass.interfaces.tracer.Tracer`.
        """
        return REGISTRY.create("tracer", name, **config)

    @staticmethod
    def mcp(name: str = "fastmcp", /, **config: Any) -> Any:
        """Construct an MCP client.

        Args:
            name: Registry name.
            **config: Transport options.

        Returns:
            An :class:`~windlass.interfaces.mcp.MCPClient`.
        """
        return REGISTRY.create("mcp", name, **config)

    @staticmethod
    def checkpointer(name: str = "memory", /, **config: Any) -> Any:
        """Construct a checkpointer.

        Args:
            name: Registry name — ``"memory"`` or ``"sqlite"``.
            **config: Checkpointer options.

        Returns:
            A :class:`~windlass.agent.checkpoint.Checkpointer`.
        """
        return REGISTRY.create("checkpointer", name, **config)

    @staticmethod
    def create(kind: str, name: str, /, **config: Any) -> Any:
        """Construct any registered component, including custom kinds.

        Args:
            kind: Component kind.
            name: Registry name.
            **config: Constructor options.

        Returns:
            The component.
        """
        return REGISTRY.create(kind, name, **config)

    # ------------------------------------------------------------------
    # Introspection & configuration
    # ------------------------------------------------------------------
    @staticmethod
    def list(kind: str | None = None) -> Any:
        """List what is registered.

        Args:
            kind: A component kind, or ``None`` for everything grouped by kind.

        Returns:
            Sorted names for one kind, or a ``{kind: [names]}`` mapping.

        Example:
            >>> from windlass import Windlass
            >>> "recursive" in Windlass.list("chunker")
            True
            >>> "llm" in Windlass.list()
            True
        """
        if kind is not None:
            return REGISTRY.names(kind)
        return {k: REGISTRY.names(k) for k in REGISTRY.kinds()}

    @staticmethod
    def catalog(kind: str | None = None) -> builtins.list[ComponentSpec]:
        """Return full registry entries, including descriptions and extras.

        Args:
            kind: A component kind, or ``None`` for everything.

        Returns:
            The matching :class:`~windlass.core.registry.ComponentSpec` objects.
        """
        if kind is not None:
            return REGISTRY.specs(kind)
        return [spec for k in REGISTRY.kinds() for spec in REGISTRY.specs(k)]

    @staticmethod
    def kinds() -> builtins.list[str]:
        """Return every component kind Windlass knows about."""
        return REGISTRY.kinds()

    @staticmethod
    def configure(**overrides: Any) -> WindlassSettings:
        """Update process-wide settings.

        Args:
            **overrides: Any field of
                :class:`~windlass.core.config.WindlassSettings`.

        Returns:
            The new settings.

        Example:
            >>> from windlass import Windlass
            >>> Windlass.configure(temperature=0.0).temperature
            0.0
        """
        return configure(**overrides)

    @staticmethod
    def settings() -> WindlassSettings:
        """Return the active settings."""
        return settings()

    @staticmethod
    def container() -> Container:
        """Return the process-wide root dependency container.

        Bindings placed here are inherited by every builder created afterwards,
        which makes it the right place for application-wide wiring::

            Windlass.container().bind_instance("tracer", MyTracer())

        Returns:
            The root :class:`~windlass.core.container.Container`.
        """
        return root_container()

    @staticmethod
    def plugins(*, strict: bool = False) -> builtins.list[str]:
        """Discover entry-point plugins.

        Runs automatically on first registry lookup; call it explicitly to see
        what loaded, or with ``strict=True`` to surface plugin errors.

        Args:
            strict: Raise on the first plugin that fails.

        Returns:
            The names that loaded successfully.
        """
        return REGISTRY.load_plugins(strict=strict)

    @staticmethod
    def tools(*functions: Any) -> Sequence[Any]:
        """Wrap plain functions as tools.

        A convenience for binding several existing functions at once.

        Args:
            *functions: Functions to wrap.

        Returns:
            The resulting tools.

        Example:
            >>> from windlass import Windlass
            >>> def ping() -> str:
            ...     '''Return pong.'''
            ...     return "pong"
            >>> [t.name for t in Windlass.tools(ping)]
            ['ping']
        """
        from windlass.interfaces.tool import Tool
        from windlass.tools import FunctionTool

        return [f if isinstance(f, Tool) else FunctionTool(f) for f in functions]

    def __init__(self) -> None:  # pragma: no cover - guard against misuse
        raise TypeError(
            "Windlass is a namespace of factories, not something to instantiate. "
            "Call Windlass.rag() or Windlass.agent() instead."
        )
