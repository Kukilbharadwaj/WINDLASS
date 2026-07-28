"""Testing helpers for applications built on Windlass.

Testing an AI application is hard for a specific reason: the model is
non-deterministic, slow and costs money. The fix is to make every component
replaceable with a deterministic double — which is exactly what the framework's
architecture already provides.

This module packages that into helpers you can use directly::

    from windlass.testing import scripted_llm, isolated_registry, capture_spans

    def test_agent_calls_the_search_tool():
        llm = scripted_llm(["", "Found it."], tool_calls=[[call("search", q="x")], []])
        agent = Windlass.agent().llm(llm).tool(search)
        with capture_spans() as spans:
            assert agent.run("find x").output == "Found it."
        assert spans.count("tool") == 1

Everything here is dependency-free and offline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from windlass.core.registry import REGISTRY, ComponentSpec
from windlass.core.types import Chunk, Document, Message, ToolCall

__all__ = [
    "RecordingTool",
    "assert_answer_uses_context",
    "call",
    "capture_spans",
    "fake_agent",
    "fake_rag",
    "isolated_registry",
    "make_chunks",
    "make_documents",
    "scripted_llm",
]


def scripted_llm(
    responses: Sequence[str] | str,
    *,
    tool_calls: Sequence[Sequence[ToolCall]] | None = None,
    handler: Callable[[list[Message], list[dict[str, Any]] | None], Any] | None = None,
    **config: Any,
) -> Any:
    """Build a model that returns a fixed script.

    Args:
        responses: Texts returned in order, one per call.
        tool_calls: Tool calls emitted per turn. Use this to drive an agent
            through a specific tool-use path.
        handler: Callable receiving ``(messages, tools)``, for assertions on what
            the model was actually asked.
        **config: Forwarded to :class:`~windlass.providers.llm.fake.FakeLLM`.

    Returns:
        A :class:`~windlass.providers.llm.fake.FakeLLM`. Its ``calls`` attribute
        records every prompt it received.

    Example:
        >>> llm = scripted_llm(["first", "second"])
        >>> llm.complete("a").content, llm.complete("b").content
        ('first', 'second')
        >>> llm.calls[0][-1].content
        'a'
    """
    from windlass.providers.llm.fake import FakeLLM

    return FakeLLM(responses, tool_calls=tool_calls, handler=handler, **config)


def call(name: str, **arguments: Any) -> ToolCall:
    """Build a :class:`~windlass.core.types.ToolCall` for a scripted model.

    Args:
        name: Tool name.
        **arguments: Tool arguments.

    Returns:
        The tool call.

    Example:
        >>> call("search", query="rag").arguments
        {'query': 'rag'}
    """
    return ToolCall(name=name, arguments=arguments)


def fake_rag(
    documents: Sequence[str] | None = None,
    *,
    answers: Sequence[str] | None = None,
    **config: Any,
) -> Any:
    """Build a fully offline RAG pipeline, optionally pre-loaded.

    Args:
        documents: Texts to ingest immediately.
        answers: Scripted answers for the model.
        **config: Forwarded to :meth:`~windlass.rag.builder.RAGBuilder.llm`.

    Returns:
        A :class:`~windlass.rag.builder.RAGBuilder` using the fake model, hash
        embeddings and the in-memory store — no network, no keys, deterministic.

    Example:
        >>> rag = fake_rag(["Paris is the capital of France."], answers=["Paris."])
        >>> rag.ask("What is the capital of France?").answer
        'Paris.'
    """
    from windlass.api import Windlass

    builder = Windlass.rag().llm(
        "fake", responses=list(answers or ["A scripted answer."]), **config
    )
    for text in documents or ():
        builder.ingest_text(text)
    return builder


def fake_agent(
    answers: Sequence[str] | None = None,
    *,
    tools: Sequence[Any] = (),
    tool_calls: Sequence[Sequence[ToolCall]] | None = None,
    **config: Any,
) -> Any:
    """Build a fully offline agent.

    Args:
        answers: Scripted model responses, in order.
        tools: Tools to bind.
        tool_calls: Tool calls emitted per turn.
        **config: Forwarded to :meth:`~windlass.agent.builder.AgentBuilder.llm`.

    Returns:
        An :class:`~windlass.agent.builder.AgentBuilder`.

    Example:
        >>> agent = fake_agent(["All done."])
        >>> agent.run("do the thing").output
        'All done.'
    """
    from windlass.api import Windlass

    builder = Windlass.agent().llm(
        "fake", responses=list(answers or ["Done."]), tool_calls=tool_calls, **config
    )
    if tools:
        builder.tool(*tools)
    return builder


@contextmanager
def isolated_registry() -> Iterator[Any]:
    """Run a block against a private copy of the component registry.

    Registering a component is a global side effect. Tests that register custom
    components would otherwise leak into each other; this snapshots the registry
    on entry and restores it on exit.

    Yields:
        The global :class:`~windlass.core.registry.Registry`, safe to mutate.

    Example:
        >>> from windlass import register
        >>> with isolated_registry() as reg:
        ...     @register.chunker("temporary")
        ...     class Temp:
        ...         pass
        ...     reg.has("chunker", "temporary")
        True
        >>> REGISTRY.has("chunker", "temporary")
        False
    """
    snapshot: dict[str, dict[str, ComponentSpec]] = REGISTRY.snapshot()
    try:
        yield REGISTRY
    finally:
        REGISTRY.restore(snapshot)


@contextmanager
def capture_spans() -> Iterator[Any]:
    """Collect the traces emitted inside the block.

    Binds a :class:`~windlass.providers.observability.console.MemoryTracer` into
    the root container, so every builder created inside the block traces into it.

    Yields:
        The tracer. Assert with ``.count(kind)``, ``.by_kind(kind)``,
        ``.errors()`` and ``.total_usage()``.

    Example:
        >>> from windlass import Windlass
        >>> with capture_spans() as spans:
        ...     agent = Windlass.agent().llm("fake", responses=["hi"]).observe("memory")
        ...     _ = agent.run("hello")
        >>> spans.count() >= 0
        True
    """
    from windlass.core.container import root_container
    from windlass.providers.observability.console import MemoryTracer

    tracer = MemoryTracer()
    container = root_container()
    had_binding = "tracer" in container
    previous = container.resolve("tracer", None) if had_binding else None
    container.bind_instance("tracer", tracer)
    try:
        yield tracer
    finally:
        if had_binding and previous is not None:
            container.bind_instance("tracer", previous)
        else:
            container.unbind("tracer")


def make_documents(*texts: str, **metadata: Any) -> list[Document]:
    """Build documents from plain strings.

    Args:
        *texts: Document contents.
        **metadata: Metadata applied to every document.

    Returns:
        The documents, each with a ``source`` of ``test-<n>``.

    Example:
        >>> docs = make_documents("one", "two", corpus="unit-test")
        >>> len(docs), docs[0].metadata["corpus"]
        (2, 'unit-test')
    """
    return [
        Document(content=text, source=f"test-{index}", metadata=dict(metadata))
        for index, text in enumerate(texts)
    ]


def make_chunks(*texts: str, embed: bool = True, dimensions: int = 64) -> list[Chunk]:
    """Build chunks from plain strings, optionally embedded.

    Args:
        *texts: Chunk contents.
        embed: Attach deterministic hash embeddings, so the chunks can be written
            straight to a vector store.
        dimensions: Embedding dimensionality.

    Returns:
        The chunks.

    Example:
        >>> chunks = make_chunks("alpha", "beta", dimensions=8)
        >>> len(chunks), len(chunks[0].embedding)
        (2, 8)
    """
    chunks = [
        Chunk(content=text, document_id="test-doc", index=index) for index, text in enumerate(texts)
    ]
    if embed:
        from windlass.providers.embeddings.hash import HashEmbedder

        embedder = HashEmbedder(dimensions=dimensions)
        for chunk, vector in zip(chunks, embedder.embed([c.content for c in chunks]), strict=True):
            chunk.embedding = vector
    return chunks


def assert_answer_uses_context(answer: Any, expected: str) -> None:
    """Assert that a RAG answer's retrieved context mentions ``expected``.

    Testing the *answer* text is testing the model, which is not what a
    pipeline test should assert on. Testing the retrieved context is testing
    your retrieval — which is yours, and is deterministic.

    Args:
        answer: A :class:`~windlass.core.types.RAGAnswer`.
        expected: A substring that should appear in the retrieved context.

    Raises:
        AssertionError: When no retrieved chunk contains ``expected``.

    Example:
        >>> rag = fake_rag(["The tower is 330 metres tall."])
        >>> assert_answer_uses_context(rag.ask("How tall is the tower?"), "330")
    """
    contexts = [hit.chunk.content for hit in answer.contexts]
    if not any(expected in text for text in contexts):
        raise AssertionError(
            f"No retrieved context contains {expected!r}.\n"
            f"Retrieved {len(contexts)} chunk(s): " + "; ".join(t[:80] for t in contexts[:5])
        )


class RecordingTool:
    """A tool that records its invocations.

    Use it to assert that an agent called what you expected, with the arguments
    you expected.

    Args:
        name: Tool name.
        result: Value returned on every call.
        description: Model-facing description.

    Attributes:
        calls: One dict of arguments per invocation.

    Example:
        >>> recorder = RecordingTool("lookup", result={"found": True})
        >>> agent = fake_agent(
        ...     ["", "Found."],
        ...     tools=[recorder.as_tool()],
        ...     tool_calls=[[call("lookup", key="abc")], []],
        ... )
        >>> agent.run("look it up").output
        'Found.'
        >>> recorder.calls
        [{'key': 'abc'}]
    """

    def __init__(self, name: str, *, result: Any = "ok", description: str = "") -> None:
        self.name = name
        self.result = result
        self.description = description or f"A recording stub for {name!r}."
        self.calls: list[dict[str, Any]] = []

    def as_tool(self) -> Any:
        """Return a bindable :class:`~windlass.interfaces.tool.Tool`.

        Returns:
            A tool accepting arbitrary keyword arguments and returning
            :attr:`result`.
        """
        from windlass.interfaces.tool import Tool

        recorder = self

        class _Recording(Tool):
            provider_name = "recording"

            async def acall(self, **kwargs: Any) -> Any:
                recorder.calls.append(dict(kwargs))
                return recorder.result

        return _Recording(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}},
        )

    def assert_called_with(self, **expected: Any) -> None:
        """Assert that the tool was called with exactly these arguments at least once.

        Args:
            **expected: The expected arguments.

        Raises:
            AssertionError: When no recorded call matches.
        """
        if not any(recorded == expected for recorded in self.calls):
            raise AssertionError(
                f"{self.name!r} was never called with {expected!r}. "
                f"Recorded calls: {self.calls!r}"
            )

    @property
    def call_count(self) -> int:
        """How many times the tool was invoked."""
        return len(self.calls)
