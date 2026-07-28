"""Console and in-memory tracers — dependency-free observability.

:class:`ConsoleTracer` prints an indented trace tree as your pipeline runs. It
is the fastest way to answer "what is this thing actually doing?" without
signing up for anything::

    Windlass.rag().observe("console")

:class:`MemoryTracer` keeps spans in a list, which is what you want in tests:
assert that retrieval ran once, that the guardrail fired, that no LLM call was
made on the cached path.

Example:
    >>> tracer = MemoryTracer()
    >>> with tracer.span("retrieve", kind="retriever") as span:
    ...     span.set_output(["a", "b"])
    <Span retriever:retrieve ...>
    >>> [s.name for s in tracer.spans]
    ['retrieve']
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, TextIO

from windlass.core.registry import register
from windlass.interfaces.tracer import Span, Tracer

__all__ = ["ConsoleTracer", "MemoryTracer"]

#: Glyph per span kind, so a trace tree is scannable at a glance.
_ICONS = {
    "chain": "▸",
    "llm": "◆",
    "embedding": "◇",
    "retriever": "⌕",
    "tool": "⚒",
    "agent": "◈",
    "guardrail": "⛨",
    "ingestion": "↓",
    "evaluation": "✓",
}


@register.tracer(
    "console",
    aliases=("stdout", "print", "debug"),
    description="Prints an indented trace tree to the console (no dependencies).",
)
class ConsoleTracer(Tracer):
    """Prints spans as an indented tree.

    Args:
        stream: Where to write. Defaults to ``sys.stderr``, so traces do not
            contaminate a program's real output.
        show_io: Print each span's inputs and outputs.
        show_usage: Print token usage for model spans.
        max_value_length: Truncate printed values at this length.
        colour: Emit ANSI colour. Auto-detected from the stream when ``None``.
        enabled: Master switch.
        **config: Forwarded to :class:`~windlass.interfaces.tracer.Tracer`.

    Example:
        >>> import io
        >>> buffer = io.StringIO()
        >>> tracer = ConsoleTracer(stream=buffer, colour=False)
        >>> with tracer.span("work", kind="tool"):
        ...     pass
        >>> "work" in buffer.getvalue()
        True
    """

    provider_name = "console"

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        show_io: bool = False,
        show_usage: bool = True,
        max_value_length: int = 160,
        colour: bool | None = None,
        enabled: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(enabled=enabled, **config)
        self.stream = stream or sys.stderr
        self.show_io = show_io
        self.show_usage = show_usage
        self.max_value_length = max_value_length
        self.colour = colour if colour is not None else _supports_colour(self.stream)
        self._depth = 0
        self._lock = threading.RLock()

    def start_span(self, span: Span) -> None:
        """Print the span's opening line and indent."""
        with self._lock:
            icon = _ICONS.get(span.kind, "•")
            self._write(f"{icon} {self._paint(span.name, '1;36')}  ({span.kind})")
            if self.show_io and span.inputs is not None:
                self._depth += 1
                self._write(f"in:  {self._render(span.inputs)}", dim=True)
                self._depth -= 1
            self._depth += 1

    def end_span(self, span: Span) -> None:
        """Print the span's result line and dedent."""
        with self._lock:
            self._depth = max(0, self._depth - 1)
            self._depth += 1
            if span.error:
                self._write(f"✗ {self._paint(span.error, '1;31')}")
            else:
                if self.show_io and span.outputs is not None:
                    self._write(f"out: {self._render(span.outputs)}", dim=True)
                bits = [f"{span.duration_ms:.0f}ms"]
                if self.show_usage and span.usage and span.usage.total_tokens:
                    bits.append(f"{span.usage.total_tokens} tok")
                    if span.usage.cost_usd:
                        bits.append(f"${span.usage.cost_usd:.4f}")
                self._write(self._paint("· " + "  ".join(bits), "2"))
            self._depth = max(0, self._depth - 1)

    def _write(self, text: str, *, dim: bool = False) -> None:
        """Write one indented line."""
        prefix = "  " * self._depth
        line = self._paint(text, "2") if dim else text
        try:
            self.stream.write(f"{prefix}{line}\n")
            self.stream.flush()
        except (ValueError, OSError):  # pragma: no cover - closed stream
            pass

    def _render(self, value: Any) -> str:
        """Render a value for display, truncated."""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                text = repr(value)
        text = " ".join(text.split())
        if len(text) > self.max_value_length:
            return f"{text[: self.max_value_length]}…"
        return text

    def _paint(self, text: str, code: str) -> str:
        """Wrap ``text`` in an ANSI escape when colour is enabled."""
        return f"\033[{code}m{text}\033[0m" if self.colour else text


@register.tracer(
    "memory",
    aliases=("collect", "test"),
    description="Collects spans in a list for assertions in tests.",
)
class MemoryTracer(Tracer):
    """Records every span in memory.

    Args:
        max_spans: Ceiling on retained spans; the oldest are dropped. Prevents a
            long-running process from growing without bound.
        enabled: Master switch.
        **config: Forwarded to :class:`~windlass.interfaces.tracer.Tracer`.

    Attributes:
        spans: Finished spans in completion order.

    Example:
        >>> tracer = MemoryTracer()
        >>> with tracer.span("llm-call", kind="llm"):
        ...     pass
        >>> tracer.count("llm")
        1
    """

    provider_name = "memory"

    def __init__(self, *, max_spans: int = 10_000, enabled: bool = True, **config: Any) -> None:
        super().__init__(enabled=enabled, **config)
        self.max_spans = max_spans
        self.spans: list[Span] = []
        self._lock = threading.RLock()

    def start_span(self, span: Span) -> None:
        """No-op: spans are recorded when they finish."""

    def end_span(self, span: Span) -> None:
        """Record the finished span."""
        with self._lock:
            self.spans.append(span)
            if len(self.spans) > self.max_spans:
                del self.spans[: len(self.spans) - self.max_spans]

    def count(self, kind: str | None = None) -> int:
        """Return how many spans were recorded.

        Args:
            kind: Restrict the count to one span kind.

        Returns:
            The number of matching spans.
        """
        with self._lock:
            if kind is None:
                return len(self.spans)
            return sum(1 for s in self.spans if s.kind == kind)

    def by_kind(self, kind: str) -> list[Span]:
        """Return every recorded span of one kind."""
        with self._lock:
            return [s for s in self.spans if s.kind == kind]

    def errors(self) -> list[Span]:
        """Return every span that failed."""
        with self._lock:
            return [s for s in self.spans if s.error]

    def total_usage(self) -> dict[str, int]:
        """Return summed token usage across every recorded span.

        Returns:
            A dict with ``prompt_tokens``, ``completion_tokens`` and
            ``total_tokens``.
        """
        with self._lock:
            prompt = sum(s.usage.prompt_tokens for s in self.spans if s.usage)
            completion = sum(s.usage.completion_tokens for s in self.spans if s.usage)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def clear(self) -> None:
        """Discard every recorded span."""
        with self._lock:
            self.spans.clear()

    def to_list(self) -> list[dict[str, Any]]:
        """Return every span as a JSON-safe dict."""
        with self._lock:
            return [s.to_dict() for s in self.spans]


def _supports_colour(stream: TextIO) -> bool:
    """Return whether ``stream`` is an interactive terminal that accepts ANSI."""
    import os

    if os.getenv("NO_COLOR"):
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False
