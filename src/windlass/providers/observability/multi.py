"""Fan-out tracing — send one trace to several backends at once.

Teams rarely have exactly one place they want traces to land. A platform team
standardises on LangSmith while a product team already reads Langfuse; during a
migration you want both; while debugging you want the console too. Without this
you would have to pick one, or write the fan-out by hand in every application.

::

    rag = Windlass.rag().observe("multi", backends=["langfuse", "langsmith"])

    agent = Windlass.agent().observe(
        "multi", backends=["console", Windlass.tracer("langfuse")]
    )

Backends are named exactly as they are in the registry, or passed as live
:class:`~windlass.interfaces.tracer.Tracer` instances when they need their own
configuration.

**A tracer is never allowed to break the application it observes**, and that
guarantee gets harder with several backends, not easier: one misconfigured
exporter must not take down the others or the run. Every call into a backend is
therefore isolated — a backend that raises is logged once and skipped for the
rest of that call, never propagated to the caller.

Example:
    >>> from windlass import Windlass
    >>> tracer = Windlass.tracer("multi", backends=["memory", "memory"])
    >>> with tracer.span("work", kind="chain"):
    ...     pass
    >>> [len(b.spans) for b in tracer.backends]
    [1, 1]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from windlass.core.registry import register
from windlass.interfaces.tracer import Span, Tracer

__all__ = ["MultiTracer"]


@register.tracer(
    "multi",
    aliases=("fanout", "fan-out", "tee"),
    description="Sends every span to several tracing backends at once.",
)
class MultiTracer(Tracer):
    """A tracer that forwards every span to several other tracers.

    Args:
        backends: Registry names (``"langfuse"``, ``"console"``, ...) or live
            :class:`~windlass.interfaces.tracer.Tracer` instances.
        project: Project/session name passed to any backend built by name.
        tags: Tags passed to any backend built by name.
        enabled: Master switch.
        **config: Forwarded to :class:`~windlass.interfaces.tracer.Tracer`.

    Attributes:
        backends: The live backend tracers, in call order.

    Raises:
        ConfigurationError: When a named backend cannot be constructed. A
            backend you asked for by name and which cannot be built is a
            configuration mistake worth failing on — unlike a *runtime* export
            failure, which is isolated and logged.

    Example:
        >>> from windlass import Windlass
        >>> tracer = Windlass.tracer("multi", backends=["memory"])
        >>> len(tracer.backends)
        1
    """

    provider_name = "multi"

    def __init__(
        self,
        *,
        backends: Sequence[Any] = (),
        project: str | None = None,
        tags: tuple[str, ...] = (),
        enabled: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(enabled=enabled, project=project, tags=tags, **config)
        from windlass.core.registry import REGISTRY

        resolved: list[Tracer] = []
        for backend in backends:
            if isinstance(backend, str):
                options: dict[str, Any] = {}
                if project is not None:
                    options["project"] = project
                if tags:
                    options["tags"] = tags
                resolved.append(REGISTRY.create("tracer", backend, **options))
            else:
                resolved.append(backend)
        self.backends: list[Tracer] = resolved
        self._broken: set[int] = set()

    def _each(self, method: str, *args: Any) -> None:
        """Call ``method`` on every healthy backend, isolating failures."""
        for index, backend in enumerate(self.backends):
            if index in self._broken:
                continue
            try:
                getattr(backend, method)(*args)
            except Exception as exc:
                # Warn once per backend, then stop calling it. A backend that
                # fails on start_span will fail on every subsequent span too;
                # logging each one would bury the application's own output.
                self._broken.add(index)
                self._log.warning(
                    "Tracing backend %s failed on %s and will be skipped: %s",
                    getattr(backend, "name", type(backend).__name__),
                    method,
                    exc,
                )

    def start_span(self, span: Span) -> None:
        """Open the span on every backend."""
        self._each("start_span", span)

    def end_span(self, span: Span) -> None:
        """Close the span on every backend."""
        self._each("end_span", span)

    def flush(self) -> None:
        """Flush every backend."""
        self._each("flush")

    async def aclose(self) -> None:
        """Close every backend, ignoring individual failures."""
        for backend in self.backends:
            close = getattr(backend, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:  # pragma: no cover - shutdown best effort
                self._log.debug("Closing tracing backend failed: %s", exc)

    def native(self) -> list[Any]:
        """Return the underlying client of each backend (Level 3 access)."""
        return [b.native() for b in self.backends]

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe summary naming every backend."""
        info = super().describe()
        info["backends"] = [getattr(b, "name", type(b).__name__) for b in self.backends]
        return info

    def __repr__(self) -> str:
        names = ", ".join(getattr(b, "name", type(b).__name__) for b in self.backends)
        return f"MultiTracer({names})"
