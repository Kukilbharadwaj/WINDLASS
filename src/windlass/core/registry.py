"""The component registry — how Windlass stays open for extension.

Every replaceable part of the framework (LLMs, chunkers, retrievers, vector
stores, ...) is a *component kind*. Implementations register a name against a
kind; builders and the DI container resolve names to instances. Nothing in
Windlass imports a concrete provider module directly, which is what keeps the
core install tiny and the architecture pluggable.

Three ways to register
----------------------

**1. Decorator (in-process)** — the common case for application code::

    from windlass import register
    from windlass.interfaces import Chunker

    @register.chunker("sentence")
    class SentenceChunker(Chunker):
        ...

**2. Lazy path (used by Windlass itself)** — records a dotted path without
importing the module, so ``import windlass`` never pulls in optional deps::

    REGISTRY.register_lazy("vectordb", "faiss", "windlass.providers.vectordb.faiss:FaissStore")

**3. Entry points (distributable plugins)** — a third-party package publishes::

    [project.entry-points."windlass.chunker"]
    sentence = "my_pkg.chunkers:SentenceChunker"

and it appears in ``windlass`` automatically, with no code change on either side.

Thread safety
-------------
All mutating operations take a re-entrant lock, and lazy resolution is
idempotent, so the registry is safe to use from worker threads.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

from windlass.core.exceptions import (
    ComponentNotFoundError,
    DuplicateComponentError,
    PluginError,
    RegistryError,
)
from windlass.core.logging import get_logger

__all__ = [
    "COMPONENT_KINDS",
    "REGISTRY",
    "ComponentSpec",
    "Registry",
    "available",
    "create",
    "load_plugins",
    "register",
]

_T = TypeVar("_T")
_log = get_logger(__name__)

#: Every extension point in Windlass. Adding a kind here is all that is needed
#: to make a new category of component pluggable end to end.
COMPONENT_KINDS: tuple[str, ...] = (
    "llm",
    "embedding",
    "loader",
    "preprocessor",
    "chunker",
    "retriever",
    "vectordb",
    "memory",
    "guardrail",
    "evaluator",
    "tracer",
    "tool",
    "mcp",
    "cache",
    "checkpointer",
)

#: Entry-point group scanned for plugin bundles that register many components.
PLUGIN_GROUP = "windlass.plugins"


@dataclass(slots=True)
class ComponentSpec:
    """A registry entry: how to obtain one implementation of a component kind.

    Exactly one of ``target`` or ``path`` is set. ``path`` entries are resolved
    on first use, which is what makes registration free of import cost.

    Attributes:
        kind: The component kind, e.g. ``"chunker"``.
        name: The lookup name, e.g. ``"semantic"``.
        target: The resolved class or factory, once known.
        path: Dotted ``"module:Attribute"`` path for deferred resolution.
        extra: Extras group needed for this component, used in error messages.
        aliases: Additional names that resolve to this spec.
        description: One-line summary shown by ``windlass list``.
        defaults: Keyword arguments merged under user config at construction.
        origin: ``"builtin"``, ``"plugin"`` or ``"user"``. Purely informational.
    """

    kind: str
    name: str
    target: Any = None
    path: str | None = None
    extra: str | None = None
    aliases: tuple[str, ...] = ()
    description: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)
    origin: str = "user"

    def resolve(self) -> Any:
        """Import and cache the target if it was registered lazily.

        Returns:
            The class or factory for this component.

        Raises:
            RegistryError: If the dotted path cannot be imported. Missing
                *optional dependencies* surface as
                :class:`~windlass.core.exceptions.MissingDependencyError` from
                inside the provider module instead.
        """
        if self.target is not None:
            return self.target
        if not self.path:  # pragma: no cover - guarded at registration
            raise RegistryError(f"Component {self.kind}:{self.name} has neither target nor path.")
        module_path, _, attr = self.path.partition(":")
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            from windlass.core.exceptions import MissingDependencyError

            if isinstance(exc, MissingDependencyError):
                raise
            raise RegistryError(
                f"Failed to import {self.kind} {self.name!r} from {self.path!r}: {exc}",
                hint=(f'Try: pip install "windlass[{self.extra}]"' if self.extra else None),
                context={"kind": self.kind, "name": self.name, "path": self.path},
            ) from exc
        self.target = getattr(module, attr) if attr else module
        return self.target


class Registry:
    """A thread-safe, kind-partitioned component registry.

    You almost never construct this yourself — use the module-level
    :data:`REGISTRY` singleton, or the :data:`register` decorator namespace.

    Args:
        discover: Whether the first lookup should scan installed distributions
            for entry-point plugins. The process-wide :data:`REGISTRY` does;
            a registry you construct yourself does **not**, so it stays an
            isolated container holding exactly what you put in it. Without that
            distinction, installing any third-party Windlass plugin would change
            the contents of every registry in the process, including the ones
            tests build to get a clean slate.

    Example:
        >>> reg = Registry()
        >>> _ = reg.register("chunker", "shouty", lambda: None, description="demo")
        >>> reg.has("chunker", "shouty")
        True
        >>> reg.names("chunker")
        ['shouty']

        Opt in explicitly when you do want the installed plugins:

        >>> discovered = Registry(discover=True)
        >>> isinstance(discovered.load_plugins(), list)
        True
    """

    def __init__(self, *, discover: bool = False) -> None:
        self._specs: dict[str, dict[str, ComponentSpec]] = {k: {} for k in COMPONENT_KINDS}
        self._aliases: dict[str, dict[str, str]] = {k: {} for k in COMPONENT_KINDS}
        self._lock = threading.RLock()
        self._discover = discover
        self._plugins_loaded = not discover

    # -- kind bookkeeping -------------------------------------------------
    def add_kind(self, kind: str) -> None:
        """Declare a brand new component kind at runtime.

        Args:
            kind: The new kind's name.
        """
        with self._lock:
            self._specs.setdefault(kind, {})
            self._aliases.setdefault(kind, {})

    def kinds(self) -> list[str]:
        """Return every known component kind, sorted."""
        return sorted(self._specs)

    def _bucket(self, kind: str) -> dict[str, ComponentSpec]:
        try:
            return self._specs[kind]
        except KeyError:
            raise RegistryError(
                f"Unknown component kind {kind!r}.",
                hint=f"Known kinds: {', '.join(self.kinds())}. "
                "Call REGISTRY.add_kind(...) to introduce a new one.",
            ) from None

    # -- registration -----------------------------------------------------
    def register(
        self,
        kind: str,
        name: str,
        target: Any,
        *,
        aliases: tuple[str, ...] | str = (),
        description: str = "",
        extra: str | None = None,
        defaults: dict[str, Any] | None = None,
        origin: str = "user",
        override: bool = False,
    ) -> ComponentSpec:
        """Register a concrete class or factory.

        Args:
            kind: Component kind.
            name: Lookup name; case-insensitive.
            target: A class or a zero-or-more-argument factory callable.
            aliases: Extra names resolving to the same component.
            description: One-line summary for CLI listings and docs.
            extra: Extras group required, used in error hints.
            defaults: Config defaults merged beneath user-supplied kwargs.
            origin: ``builtin`` / ``plugin`` / ``user``.
            override: Allow replacing an existing registration.

        Returns:
            The stored :class:`ComponentSpec`.

        Raises:
            DuplicateComponentError: If ``name`` is taken and ``override`` is False.
        """
        return self._store(
            ComponentSpec(
                kind=kind,
                name=name.lower(),
                target=target,
                aliases=(aliases,) if isinstance(aliases, str) else tuple(aliases),
                description=description,
                extra=extra,
                defaults=dict(defaults or {}),
                origin=origin,
            ),
            override=override,
        )

    def register_lazy(
        self,
        kind: str,
        name: str,
        path: str,
        *,
        aliases: tuple[str, ...] | str = (),
        description: str = "",
        extra: str | None = None,
        defaults: dict[str, Any] | None = None,
        origin: str = "builtin",
        override: bool = False,
    ) -> ComponentSpec:
        """Register a component by dotted path without importing it.

        This is how every built-in provider is wired up: the module is only
        imported the first time somebody actually asks for that component.

        Args:
            kind: Component kind.
            name: Lookup name.
            path: ``"package.module:Attribute"``.
            aliases: Extra names resolving here.
            description: One-line summary.
            extra: Extras group required by the target module.
            defaults: Config defaults.
            origin: Provenance label.
            override: Allow replacing an existing registration.

        Returns:
            The stored :class:`ComponentSpec`.
        """
        return self._store(
            ComponentSpec(
                kind=kind,
                name=name.lower(),
                path=path,
                aliases=(aliases,) if isinstance(aliases, str) else tuple(aliases),
                description=description,
                extra=extra,
                defaults=dict(defaults or {}),
                origin=origin,
            ),
            override=override,
        )

    def _store(self, spec: ComponentSpec, *, override: bool) -> ComponentSpec:
        with self._lock:
            bucket = self._bucket(spec.kind)
            existing = bucket.get(spec.name)
            if existing is not None and not override:
                # Built-ins are declared lazily in the manifest and then again by
                # the @register decorator when their module is finally imported.
                # That is the same component arriving twice, not a conflict.
                if _same_component(existing, spec):
                    existing.target = spec.target or existing.target
                    return existing
                raise DuplicateComponentError(spec.kind, spec.name)
            bucket[spec.name] = spec
            for alias in spec.aliases:
                self._aliases[spec.kind][alias.lower()] = spec.name
            return spec

    def unregister(self, kind: str, name: str) -> None:
        """Remove a component and its aliases. No-op when absent."""
        with self._lock:
            bucket = self._bucket(kind)
            resolved = self._aliases[kind].pop(name.lower(), name.lower())
            spec = bucket.pop(resolved, None)
            if spec is not None:
                for alias in spec.aliases:
                    self._aliases[kind].pop(alias.lower(), None)

    # -- lookup -----------------------------------------------------------
    def get(self, kind: str, name: str) -> ComponentSpec:
        """Return the spec registered under ``name``.

        Args:
            kind: Component kind.
            name: Name or alias, case-insensitive.

        Returns:
            The matching spec.

        Raises:
            ComponentNotFoundError: If nothing matches. The error lists all
                available names for that kind.
        """
        self.ensure_plugins_loaded()
        with self._lock:
            bucket = self._bucket(kind)
            key = name.lower()
            key = self._aliases[kind].get(key, key)
            spec = bucket.get(key)
            if spec is None:
                raise ComponentNotFoundError(kind, name, list(bucket))
            return spec

    def has(self, kind: str, name: str) -> bool:
        """Return whether ``name`` resolves for ``kind``."""
        try:
            self.get(kind, name)
        except (ComponentNotFoundError, RegistryError):
            return False
        return True

    def names(self, kind: str) -> list[str]:
        """Return every registered name for ``kind``, sorted."""
        self.ensure_plugins_loaded()
        with self._lock:
            return sorted(self._bucket(kind))

    def specs(self, kind: str) -> list[ComponentSpec]:
        """Return every spec for ``kind``, sorted by name."""
        self.ensure_plugins_loaded()
        with self._lock:
            return [self._bucket(kind)[n] for n in sorted(self._bucket(kind))]

    def catalog(self) -> dict[str, list[ComponentSpec]]:
        """Return every spec grouped by kind — powers ``windlass list``."""
        return {kind: self.specs(kind) for kind in self.kinds()}

    def __iter__(self) -> Iterator[ComponentSpec]:
        for kind in self.kinds():
            yield from self.specs(kind)

    # -- construction -----------------------------------------------------
    def create(self, kind: str, name: str, /, **config: Any) -> Any:
        """Instantiate a registered component.

        Registered defaults are merged *underneath* ``config``, so caller
        arguments always win.

        Args:
            kind: Component kind.
            name: Name or alias.
            **config: Constructor keyword arguments.

        Returns:
            The constructed component.

        Raises:
            ComponentNotFoundError: If the name is not registered.
            ConfigurationError: If the constructor rejects the arguments.

        Example:
            >>> from windlass.core.registry import REGISTRY
            >>> llm = REGISTRY.create("llm", "fake", responses=["hi"])
            >>> llm.complete("anything").content
            'hi'
        """
        spec = self.get(kind, name)
        factory = spec.resolve()
        merged = {**spec.defaults, **config}
        try:
            return factory(**merged)
        except TypeError as exc:
            from windlass.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Could not construct {kind} {name!r}: {exc}",
                hint=f"Check the accepted options for {getattr(factory, '__name__', factory)}.",
                context={"kind": kind, "name": name, "config": sorted(merged)},
            ) from exc

    # -- plugins ----------------------------------------------------------
    def ensure_plugins_loaded(self) -> None:
        """Discover entry-point plugins exactly once, on first lookup.

        A no-op unless this registry was constructed with ``discover=True``.
        :meth:`load_plugins` remains available on any registry for callers that
        want discovery on demand.
        """
        if self._plugins_loaded:
            return
        with self._lock:
            if self._plugins_loaded:
                return
            self._plugins_loaded = True  # set first: discovery must not recurse
        self.load_plugins()

    def load_plugins(self, *, strict: bool = False) -> list[str]:
        """Scan installed distributions for Windlass plugins.

        Two entry-point conventions are supported:

        * ``windlass.plugins`` — the value is a callable taking the registry;
          use this to register many components or to do custom setup.
        * ``windlass.<kind>`` (e.g. ``windlass.chunker``) — the value is the
          component class itself, registered under the entry-point name.

        Args:
            strict: Raise on the first plugin that fails instead of warning.

        Returns:
            Names of the plugins that loaded successfully.

        Raises:
            PluginError: Only when ``strict`` is True and a plugin fails.
        """
        from importlib.metadata import entry_points

        loaded: list[str] = []

        def _handle(err: Exception, label: str) -> None:
            if strict:
                raise PluginError(
                    f"Plugin {label!r} failed to load: {err}", context={"plugin": label}
                ) from err
            _log.warning("Skipping Windlass plugin %r: %s", label, err)

        for ep in entry_points(group=PLUGIN_GROUP):
            try:
                hook = ep.load()
                hook(self)
                loaded.append(ep.name)
            except Exception as exc:
                _handle(exc, ep.name)

        for kind in self.kinds():
            for ep in entry_points(group=f"windlass.{kind}"):
                try:
                    self.register_lazy(
                        kind,
                        ep.name,
                        ep.value.replace(":", ":", 1),
                        description=f"Plugin from {ep.value}",
                        origin="plugin",
                        override=True,
                    )
                    loaded.append(f"{kind}:{ep.name}")
                except Exception as exc:
                    _handle(exc, f"{kind}:{ep.name}")

        return loaded

    # -- test helpers -----------------------------------------------------
    def snapshot(self) -> dict[str, dict[str, ComponentSpec]]:
        """Return a shallow copy of the registry state (for tests)."""
        with self._lock:
            return {kind: dict(bucket) for kind, bucket in self._specs.items()}

    def restore(self, snapshot: dict[str, dict[str, ComponentSpec]]) -> None:
        """Restore a previously captured :meth:`snapshot` (for tests)."""
        with self._lock:
            self._specs = {kind: dict(bucket) for kind, bucket in snapshot.items()}


def _same_component(existing: ComponentSpec, incoming: ComponentSpec) -> bool:
    """Return whether two specs describe the same implementation.

    Compares the dotted path a lazy spec points at against the module and
    qualified name of a concrete target, which is how a manifest entry and the
    ``@register`` decorator in the same module are recognised as one component.
    """
    if existing.target is not None and existing.target is incoming.target:
        return True
    for lazy, concrete in ((existing, incoming), (incoming, existing)):
        target = concrete.target
        if lazy.path and target is not None:
            module = getattr(target, "__module__", None)
            qualname = getattr(target, "__qualname__", None)
            if module and qualname and lazy.path == f"{module}:{qualname}":
                return True
    return bool(existing.path and existing.path == incoming.path)


#: The process-wide registry every builder resolves against. This is the one
#: that discovers installed plugins; registries you construct do not.
REGISTRY = Registry(discover=True)


class _RegisterNamespace:
    """Decorator namespace: ``@register.llm("name")``, ``@register.chunker(...)``.

    One attribute per component kind, each returning a class decorator. The
    decorator returns the class unchanged so it stays usable directly.

    Example:
        >>> from windlass import register
        >>> @register.chunker("noop", description="passes text through")
        ... class NoopChunker:
        ...     pass
        >>> from windlass.core.registry import REGISTRY
        >>> REGISTRY.has("chunker", "noop")
        True
    """

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def __getattr__(self, kind: str) -> Callable[..., Callable[[_T], _T]]:
        if kind.startswith("_"):
            raise AttributeError(kind)
        if kind not in self._registry.kinds():
            raise AttributeError(
                f"{kind!r} is not a Windlass component kind. "
                f"Known kinds: {', '.join(self._registry.kinds())}"
            )

        def decorator(
            name: str,
            *,
            aliases: tuple[str, ...] | str = (),
            description: str = "",
            defaults: dict[str, Any] | None = None,
            override: bool = False,
        ) -> Callable[[_T], _T]:
            def wrap(target: _T) -> _T:
                self._registry.register(
                    kind,
                    name,
                    target,
                    aliases=aliases,
                    description=description or (target.__doc__ or "").strip().split("\n")[0],
                    defaults=defaults,
                    origin="user",
                    override=override,
                )
                return target

            return wrap

        return decorator

    def component(self, kind: str, name: str, **kwargs: Any) -> Callable[[_T], _T]:
        """Register against an arbitrary (possibly custom) kind.

        Args:
            kind: Component kind, created on demand if unknown.
            name: Lookup name.
            **kwargs: Forwarded to :meth:`Registry.register`.

        Returns:
            A class decorator.
        """
        self._registry.add_kind(kind)
        return getattr(self, kind)(name, **kwargs)

    def __dir__(self) -> list[str]:  # pragma: no cover - REPL nicety
        return [*self._registry.kinds(), "component"]


#: Decorator namespace bound to :data:`REGISTRY`.
register = _RegisterNamespace(REGISTRY)


def create(kind: str, name: str, /, **config: Any) -> Any:
    """Instantiate a component from the global registry.

    Thin module-level shortcut for :meth:`Registry.create`.
    """
    return REGISTRY.create(kind, name, **config)


def available(kind: str) -> list[str]:
    """List registered names for ``kind`` in the global registry."""
    return REGISTRY.names(kind)


def load_plugins(*, strict: bool = False) -> list[str]:
    """Force plugin discovery on the global registry."""
    return REGISTRY.load_plugins(strict=strict)
