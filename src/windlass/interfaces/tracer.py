"""The observability interface.

Windlass instruments every meaningful operation — a model call, a retrieval, a
tool execution, an ingestion run — as a :class:`Span`. A :class:`Tracer` decides
what to do with those spans: drop them, print them, or ship them to LangSmith or
Langfuse.

Instrumentation is always on and always cheap. With no tracer configured, spans
resolve to :class:`NullTracer`, whose ``span()`` is a context manager that does
nothing measurable.

Implementers override one method, :meth:`Tracer.start_span`.

Example:
    >>> from windlass.providers.observability.console import ConsoleTracer
    >>> tracer = ConsoleTracer(enabled=False)
    >>> with tracer.span("demo", kind="llm") as span:
    ...     _ = span.set_output("done")
    >>> span.name
    'demo'
"""

from __future__ import annotations

import abc
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from windlass.core.types import Usage
from windlass.interfaces.base import Component

__all__ = ["SPAN_KINDS", "NullTracer", "Span", "Tracer"]

#: Recognised span kinds. Backends map these onto their own taxonomies.
SPAN_KINDS = (
    "chain",
    "llm",
    "embedding",
    "retriever",
    "tool",
    "agent",
    "guardrail",
    "ingestion",
    "evaluation",
)


class Span:
    """One timed, attributed unit of work.

    Spans are created by :meth:`Tracer.span` and finished automatically when the
    context manager exits. Exceptions are recorded before propagating, so a
    failed run is still fully traced.

    Args:
        name: Human readable operation name.
        kind: One of :data:`SPAN_KINDS`.
        parent_id: Enclosing span's id, when nested.
        metadata: Static attributes attached at creation.
        trace_id: Groups spans belonging to one top-level request.

    Attributes:
        id: Unique span id.
        started_at: Monotonic start time.
        ended_at: Monotonic end time, set on :meth:`end`.
        error: Error message, when the span failed.
        usage: Token accounting, for model spans.
    """

    __slots__ = (
        "_native",
        "ended_at",
        "error",
        "id",
        "inputs",
        "kind",
        "metadata",
        "name",
        "outputs",
        "parent_id",
        "started_at",
        "trace_id",
        "usage",
    )

    def __init__(
        self,
        name: str,
        *,
        kind: str = "chain",
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:16]
        self.trace_id = trace_id or self.id
        self.parent_id = parent_id
        self.name = name
        self.kind = kind
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.inputs: Any = None
        self.outputs: Any = None
        self.usage: Usage | None = None
        self.error: str | None = None
        self.started_at = time.perf_counter()
        self.ended_at: float | None = None
        self._native: Any = None

    @property
    def duration_ms(self) -> float:
        """Elapsed time in milliseconds — live until the span ends."""
        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return (end - self.started_at) * 1000

    def set_input(self, value: Any) -> Span:
        """Record the operation's input. Returns ``self`` for chaining."""
        self.inputs = value
        return self

    def set_output(self, value: Any) -> Span:
        """Record the operation's output. Returns ``self`` for chaining."""
        self.outputs = value
        return self

    def set_usage(self, usage: Usage) -> Span:
        """Record token accounting. Returns ``self`` for chaining."""
        self.usage = usage
        return self

    def set_metadata(self, **fields: Any) -> Span:
        """Merge extra attributes. Returns ``self`` for chaining."""
        self.metadata.update(fields)
        return self

    def set_error(self, error: BaseException | str) -> Span:
        """Mark the span as failed. Returns ``self`` for chaining."""
        self.error = str(error)
        return self

    def end(self) -> Span:
        """Stop the clock. Idempotent."""
        if self.ended_at is None:
            self.ended_at = time.perf_counter()
        return self

    def attach_native(self, obj: Any) -> Span:
        """Associate the backend's own span object, for Level 3 access."""
        self._native = obj
        return self

    def native(self) -> Any:
        """Return the backend's span object, if the tracer created one."""
        return self._native

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the span."""
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
            "inputs": _clip(self.inputs),
            "outputs": _clip(self.outputs),
            "usage": self.usage.model_dump() if self.usage else None,
            "error": self.error,
        }

    def __repr__(self) -> str:
        status = "error" if self.error else "ok"
        return f"<Span {self.kind}:{self.name} {self.duration_ms:.1f}ms {status}>"


class Tracer(Component):
    """Abstract tracing backend.

    Args:
        enabled: Master switch. Disabled tracers still create spans (so code
            paths are identical) but never export them.
        project: Project / session name shown in the backend's UI.
        tags: Tags applied to every span.
        name: Component name.
        **config: Backend-specific options.

    Example:
        Implementing a tracer takes one method::

            class PrintTracer(Tracer):
                provider_name = "print"

                def start_span(self, span): ...
                def end_span(self, span):
                    print(span.to_dict())
    """

    kind = "tracer"
    provider_name: str = "tracer"

    def __init__(
        self,
        *,
        enabled: bool = True,
        project: str | None = None,
        tags: tuple[str, ...] = (),
        name: str | None = None,
        **config: Any,
    ) -> None:
        from windlass.core.config import settings

        super().__init__(
            name=name or self.provider_name,
            enabled=enabled,
            project=project or settings().project,
            tags=tags,
            **config,
        )
        self.enabled = enabled
        self.project = project or settings().project
        self.tags = tuple(tags)
        self._stack: list[Span] = []

    # -- provider hooks ---------------------------------------------------
    @abc.abstractmethod
    def start_span(self, span: Span) -> None:
        """Called when a span begins.

        Args:
            span: The span that just started. Attach a backend object with
                :meth:`Span.attach_native` if the backend has one.
        """

    def end_span(self, span: Span) -> None:
        """Called when a span ends, successfully or not.

        Args:
            span: The finished span.
        """

    def flush(self) -> None:
        """Force any buffered spans to be exported.

        Backends that batch should override this; a process that exits without
        flushing loses its last traces.
        """

    # -- public API -------------------------------------------------------
    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "chain",
        inputs: Any = None,
        **metadata: Any,
    ) -> Iterator[Span]:
        """Open a span around a block of work.

        Spans nest automatically: a span opened inside another becomes its
        child, which is what produces a readable trace tree from plain code.

        Args:
            name: Operation name.
            kind: One of :data:`SPAN_KINDS`.
            inputs: The operation's input, recorded immediately.
            **metadata: Extra attributes.

        Yields:
            The live :class:`Span`.

        Raises:
            Exception: Anything the block raises, after recording it.

        Example:
            >>> from windlass.providers.observability.console import ConsoleTracer
            >>> with ConsoleTracer(enabled=False).span("work", kind="tool") as s:
            ...     s.set_output(42)
            <Span tool:work ...>
        """
        parent = self._stack[-1] if self._stack else None
        current = Span(
            name,
            kind=kind,
            parent_id=parent.id if parent else None,
            trace_id=parent.trace_id if parent else None,
            metadata={**dict.fromkeys(self.tags, True), **metadata},
        )
        if inputs is not None:
            current.set_input(inputs)

        self._stack.append(current)
        if self.enabled:
            try:
                self.start_span(current)
            except Exception as exc:
                self._log.debug("Tracer %s failed to start a span: %s", self.name, exc)
        try:
            yield current
        except BaseException as exc:
            current.set_error(exc)
            raise
        finally:
            current.end()
            self._stack.pop()
            if self.enabled:
                try:
                    self.end_span(current)
                except Exception as exc:
                    self._log.debug("Tracer %s failed to end a span: %s", self.name, exc)

    def current_span(self) -> Span | None:
        """Return the innermost open span, if any."""
        return self._stack[-1] if self._stack else None

    def event(self, name: str, **fields: Any) -> None:
        """Record a zero-duration event on the current span.

        Args:
            name: Event name.
            **fields: Event attributes.
        """
        span = self.current_span()
        if span is not None:
            span.metadata.setdefault("events", []).append({"name": name, **fields})

    def __repr__(self) -> str:
        return f"{type(self).__name__}(project={self.project!r}, enabled={self.enabled})"


class NullTracer(Tracer):
    """A tracer that discards everything.

    Used whenever observability is not configured, so instrumented code paths
    never need a ``if tracer is not None:`` guard.

    Example:
        >>> with NullTracer().span("x") as span:
        ...     span.set_output(1)
        <Span chain:x ...>
    """

    provider_name = "null"

    def __init__(self, **config: Any) -> None:
        super().__init__(enabled=False, **config)

    def start_span(self, span: Span) -> None:
        """Discards the span."""

    def end_span(self, span: Span) -> None:
        """Discards the span."""


def _clip(value: Any, limit: int = 2000) -> Any:
    """Truncate large payloads so traces stay cheap to ship and read."""
    if isinstance(value, str) and len(value) > limit:
        return f"{value[:limit]}… ({len(value)} chars)"
    if isinstance(value, list | tuple) and len(value) > 20:
        return [_clip(v) for v in value[:20]] + [f"… ({len(value)} items)"]
    if isinstance(value, dict):
        return {k: _clip(v, limit) for k, v in list(value.items())[:20]}
    return value
