"""Dependency injection container.

Builders never construct their collaborators directly — they ask the container.
That is what makes every part of a pipeline swappable without touching Windlass
source: bind a different factory (or a ready-made instance) and every consumer
downstream picks it up.

The container understands three ways to name a dependency, which is what lets
the fluent API accept all of these:

    .llm("openai")                       # a registry name
    .llm(MyCustomLLM(model="x"))         # an instance
    .llm(lambda: MyCustomLLM(model="x")) # a factory

Containers nest. :meth:`Container.scope` returns a child that inherits every
binding and can override any of them without disturbing the parent — that is how
a sub-agent gets its own LLM while sharing the parent's tracer.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

from windlass.core.exceptions import ConfigurationError
from windlass.core.logging import get_logger
from windlass.core.registry import REGISTRY, Registry

__all__ = ["Binding", "Container", "Provide", "root_container"]

_T = TypeVar("_T")
_log = get_logger(__name__)
_UNSET = object()


@dataclass(slots=True)
class Binding:
    """How to produce one dependency.

    Attributes:
        factory: Callable producing the value. May take zero arguments or a
            single :class:`Container` argument.
        singleton: When True the value is produced once and reused.
        value: The memoised singleton value, once produced.
        produced: Whether ``value`` has been computed (distinguishes a cached
            ``None`` from "not yet built").
    """

    factory: Callable[..., Any]
    singleton: bool = True
    value: Any = None
    produced: bool = False


@dataclass(slots=True)
class Provide:
    """Marker describing a dependency to be resolved later.

    Use it as a default value to declare that a parameter should come from the
    container::

        def build(llm = Provide("llm", "openai")):
            ...

    Attributes:
        kind: Component kind, e.g. ``"llm"``.
        name: Registry name to fall back on when the container has no binding.
        config: Keyword arguments for construction.
    """

    kind: str
    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


class Container:
    """A hierarchical, thread-safe dependency container.

    Args:
        parent: Optional parent whose bindings are inherited.
        registry: Component registry used to resolve string names. Defaults to
            the global :data:`~windlass.core.registry.REGISTRY`.

    Example:
        >>> c = Container()
        >>> _ = c.bind("greeting", lambda: "hello")
        >>> c.resolve("greeting")
        'hello'
        >>> child = c.scope()
        >>> _ = child.bind("greeting", lambda: "hi")
        >>> c.resolve("greeting"), child.resolve("greeting")
        ('hello', 'hi')
    """

    def __init__(self, parent: Container | None = None, registry: Registry | None = None) -> None:
        self._parent = parent
        inherited: Registry = parent._registry if parent is not None else REGISTRY
        self._registry: Registry = registry or inherited
        self._bindings: dict[Any, Binding] = {}
        self._lock = threading.RLock()

    # -- binding ----------------------------------------------------------
    def bind(
        self,
        key: Any,
        factory: Callable[..., Any],
        *,
        singleton: bool = True,
        override: bool = True,
    ) -> Container:
        """Bind ``key`` to a factory.

        Args:
            key: A string, a type, or any hashable token.
            factory: Zero-argument callable, or one accepting the container.
            singleton: Reuse the first produced value.
            override: Replace an existing binding. When False, an existing
                binding is kept and the call is a no-op.

        Returns:
            ``self``, so calls chain.

        Raises:
            ConfigurationError: If ``factory`` is not callable.
        """
        if not callable(factory):
            raise ConfigurationError(
                f"Binding for {key!r} must be callable, got {type(factory).__name__}.",
                hint="Pass a factory (or use bind_instance for a ready-made object).",
            )
        with self._lock:
            if not override and key in self._bindings:
                return self
            self._bindings[key] = Binding(factory=factory, singleton=singleton)
        return self

    def bind_instance(self, key: Any, instance: Any, *, override: bool = True) -> Container:
        """Bind ``key`` to an already-constructed object.

        Args:
            key: Lookup key.
            instance: The object to return on resolution.
            override: Replace an existing binding.

        Returns:
            ``self``.
        """
        with self._lock:
            if not override and key in self._bindings:
                return self
            self._bindings[key] = Binding(
                factory=lambda: instance, singleton=True, value=instance, produced=True
            )
        return self

    def unbind(self, key: Any) -> None:
        """Remove a local binding. Parent bindings become visible again."""
        with self._lock:
            self._bindings.pop(key, None)

    def has(self, key: Any) -> bool:
        """Return whether ``key`` resolves here or in any ancestor."""
        if key in self._bindings:
            return True
        return self._parent.has(key) if self._parent else False

    # -- resolution -------------------------------------------------------
    @overload
    def resolve(self, key: type[_T]) -> _T: ...

    @overload
    def resolve(self, key: type[_T], default: _T) -> _T: ...

    @overload
    def resolve(self, key: Any, default: Any = ...) -> Any: ...

    def resolve(self, key: Any, default: Any = _UNSET) -> Any:
        """Produce the value bound to ``key``.

        Singletons are produced at most once, under a lock, so concurrent first
        resolutions do not both run the factory.

        Args:
            key: Lookup key.
            default: Returned instead of raising when nothing is bound.

        Returns:
            The resolved value.

        Raises:
            ConfigurationError: When nothing is bound and no ``default`` was
                given, or when the factory itself fails.
        """
        binding = self._find(key)
        if binding is None:
            if default is not _UNSET:
                return default
            raise ConfigurationError(
                f"Nothing is bound for {_label(key)}.",
                hint="Bind it with container.bind(...) or pass it explicitly to the builder.",
                context={"key": _label(key), "bound": [_label(k) for k in self.keys()]},
            )

        if binding.singleton and binding.produced:
            return binding.value

        with self._lock:
            if binding.singleton and binding.produced:
                return binding.value
            try:
                value = _invoke(binding.factory, self)
            except ConfigurationError:
                raise
            except Exception as exc:
                raise ConfigurationError(
                    f"Factory for {_label(key)} raised {type(exc).__name__}: {exc}"
                ) from exc
            if binding.singleton:
                binding.value = value
                binding.produced = True
            return value

    def _find(self, key: Any) -> Binding | None:
        node: Container | None = self
        while node is not None:
            binding = node._bindings.get(key)
            if binding is not None:
                return binding
            node = node._parent
        return None

    def keys(self) -> list[Any]:
        """Return every key visible from here, including inherited ones."""
        seen: dict[Any, None] = {}
        node: Container | None = self
        while node is not None:
            for key in node._bindings:
                seen.setdefault(key, None)
            node = node._parent
        return list(seen)

    # -- component helpers ------------------------------------------------
    def component(
        self,
        kind: str,
        spec: Any = None,
        /,
        **config: Any,
    ) -> Any:
        """Resolve a component from a name, an instance, or a factory.

        This is the single funnel through which every builder turns user input
        into a live component, and the reason ``.llm(...)`` accepts three
        different kinds of argument.

        Args:
            kind: Component kind, e.g. ``"chunker"``.
            spec: One of:

                * ``None`` — resolve the binding for ``kind`` if present,
                  otherwise fail with a clear message.
                * ``str`` — a registry name; ``config`` is passed to the
                  constructor.
                * an instance — used as-is (``config`` must be empty).
                * a callable — invoked, optionally with the container.
            **config: Constructor keyword arguments for the string form.

        Returns:
            The live component.

        Raises:
            ComponentNotFoundError: For an unknown registry name.
            ConfigurationError: For an unusable ``spec``.

        Example:
            >>> c = Container()
            >>> llm = c.component("llm", "fake", responses=["hi"])
            >>> llm.complete("x").content
            'hi'
        """
        if spec is None:
            if self.has(kind):
                return self.resolve(kind)
            raise ConfigurationError(
                f"No {kind} configured.",
                hint=f"Pass one explicitly, e.g. .{kind}('name'), "
                f"or bind it with container.bind('{kind}', factory).",
            )

        if isinstance(spec, str):
            return self._registry.create(kind, spec, **config)

        if callable(spec) and (inspect.isfunction(spec) or inspect.ismethod(spec)):
            produced = _invoke(spec, self)
            return produced

        if isinstance(spec, type):
            return spec(**config)

        if config:
            raise ConfigurationError(
                f"Cannot apply configuration to an already-constructed {kind}.",
                hint=f"Either pass .{kind}('name', **config) or configure the instance yourself.",
                context={"kind": kind, "config": sorted(config)},
            )
        return spec

    def bind_component(
        self, kind: str, spec: Any = None, /, *, singleton: bool = True, **config: Any
    ) -> Container:
        """Bind ``kind`` to a lazily resolved component.

        The component is not constructed until something resolves it, which
        keeps builder chains cheap and lets configuration keep changing right up
        until the first call.

        Args:
            kind: Component kind, used as the binding key.
            spec: Name, instance or factory — see :meth:`component`.
            singleton: Reuse the constructed component.
            **config: Constructor keyword arguments.

        Returns:
            ``self``.
        """
        return self.bind(kind, lambda: self.component(kind, spec, **config), singleton=singleton)

    # -- scoping ----------------------------------------------------------
    def scope(self) -> Container:
        """Return a child container inheriting every binding from this one."""
        return Container(parent=self, registry=self._registry)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.keys())

    def __contains__(self, key: Any) -> bool:
        return self.has(key)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Container bindings={[_label(k) for k in self._bindings]}>"


def _invoke(factory: Callable[..., Any], container: Container) -> Any:
    """Call ``factory`` with the container when it declares a parameter for it."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # builtins, C callables
        return factory()
    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    if len(positional) == 1:
        return factory(container)
    return factory()


def _label(key: Any) -> str:
    """Render a binding key for error messages."""
    if isinstance(key, type):
        return key.__name__
    return str(key)


_ROOT = Container()


def root_container() -> Container:
    """Return the process-wide root container.

    Bindings placed here are inherited by every builder created afterwards,
    which makes it the right place for application-wide wiring::

        root_container().bind_instance("tracer", MyTracer())

    Returns:
        The shared root :class:`Container`.
    """
    return _ROOT
