"""The base every Windlass component shares.

:class:`Component` gives all thirteen component kinds the same three things:

1. **Identity** — a ``name`` used in traces, errors and registry lookups.
2. **Configuration** — a validated ``config`` mapping, echoed in ``repr`` and
   traces so a misbehaving pipeline is debuggable from its logs alone.
3. **Native access (Level 3)** — :meth:`Component.native` hands back the
   underlying SDK object. Windlass simplifies libraries; it never hides them.

Subclasses implement async methods. The sync variants are derived once, here and
in each interface, so the two APIs can never drift.
"""

from __future__ import annotations

import abc
from typing import Any

from windlass.core.exceptions import ConfigurationError
from windlass.core.logging import get_logger

__all__ = ["Component", "SupportsNative"]


class SupportsNative(abc.ABC):
    """Mixin declaring the Level 3 escape hatch.

    Any adapter wrapping a third-party object should return it from
    :meth:`native` so advanced users can reach past Windlass without forking it.
    """

    @abc.abstractmethod
    def native(self) -> Any:
        """Return the underlying provider object.

        Returns:
            The wrapped SDK client, index, graph or handle. Adapters with no
            third-party object behind them return ``self``.

        Example:
            >>> from windlass.providers.llm.fake import FakeLLM
            >>> FakeLLM().native() is not None
            True
        """


class Component(SupportsNative):
    """Abstract base for every registrable Windlass component.

    Args:
        name: Identifier used in traces and error messages. Defaults to the
            class's registered name, or its class name.
        **config: Arbitrary component configuration, kept on :attr:`config`.

    Attributes:
        name: The component's identifier.
        config: The configuration it was constructed with.

    Example:
        >>> class Upper(Component):
        ...     def apply(self, text: str) -> str:
        ...         return text.upper()
        >>> Upper(name="upper").name
        'upper'
    """

    #: Registry kind this component implements. Set by each interface.
    kind: str = "component"

    #: Extras group needed for this component, surfaced in error messages.
    requires_extra: str | None = None

    def __init__(self, *, name: str | None = None, **config: Any) -> None:
        self.name = name or getattr(type(self), "provider_name", None) or type(self).__name__
        self.config: dict[str, Any] = dict(config)
        self._log = get_logger(f"{self.kind}.{self.name}")

    # -- configuration ----------------------------------------------------
    def option(self, key: str, default: Any = None) -> Any:
        """Read a configuration value.

        Args:
            key: Configuration key.
            default: Returned when the key is absent.

        Returns:
            The configured value or ``default``.
        """
        return self.config.get(key, default)

    def require_option(self, key: str) -> Any:
        """Read a configuration value that must be present.

        Args:
            key: Configuration key.

        Returns:
            The configured value.

        Raises:
            ConfigurationError: When the key is missing or ``None``.
        """
        value = self.config.get(key)
        if value is None:
            raise ConfigurationError(
                f"{type(self).__name__} requires the {key!r} option.",
                hint=f"Pass {key}=... when constructing the {self.kind}.",
                context={"component": self.name, "kind": self.kind, "option": key},
            )
        return value

    def with_config(self, **overrides: Any) -> Component:
        """Return a new component of the same type with merged configuration.

        Components are treated as immutable once built; reconfiguring produces a
        copy so a shared component cannot be mutated out from under another
        pipeline.

        Args:
            **overrides: Configuration to merge over the current values.

        Returns:
            A new instance of the same class.
        """
        merged = {**self.config, **overrides}
        return type(self)(name=self.name, **merged)

    # -- lifecycle --------------------------------------------------------
    async def aclose(self) -> None:
        """Release resources held by the component.

        The default is a no-op. Adapters holding sockets, file handles or thread
        pools should override it. Safe to call more than once.
        """

    def close(self) -> None:
        """Blocking counterpart of :meth:`aclose`."""
        from windlass.core.concurrency import run_sync

        run_sync(self.aclose())

    # -- native access ----------------------------------------------------
    def native(self) -> Any:
        """Return the wrapped provider object.

        The default returns ``self``, which is correct for pure-Python
        components that wrap nothing.
        """
        return self

    # -- niceties ---------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe summary of this component.

        Used by tracing, by ``Pipeline.describe()`` and by the CLI.

        Returns:
            A dict with ``kind``, ``name``, ``class`` and ``config``.
        """
        return {
            "kind": self.kind,
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "config": {k: _safe(k, v) for k, v in self.config.items()},
        }

    def __repr__(self) -> str:
        opts = ", ".join(f"{k}={v!r}" for k, v in list(self.config.items())[:4])
        return f"{type(self).__name__}({opts})" if opts else f"{type(self).__name__}()"


#: Config keys whose values must never reach a log line or a trace payload.
_SECRET_HINTS = ("key", "secret", "token", "password", "credential")


def _safe(key: str, value: Any) -> Any:
    """Render a config value in a form that is safe to log and serialise.

    Secrets are masked by key name, containers are truncated, and anything
    exotic degrades to its type name rather than an unbounded ``repr``.
    """
    if any(hint in key.lower() for hint in _SECRET_HINTS) and value:
        return "***"
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, list | tuple):
        return [_safe(key, v) for v in value][:10]
    if isinstance(value, dict):
        return {k: _safe(k, v) for k, v in list(value.items())[:10]}
    return f"<{type(value).__name__}>"
