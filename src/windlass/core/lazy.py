"""Optional dependency loading.

Windlass ships a tiny core and pushes every integration behind an extras group.
This module is the single place where those optional imports happen, so that:

* ``import windlass`` never pulls in torch, chromadb or langgraph;
* a missing package produces an actionable :class:`MissingDependencyError`
  instead of a raw ``ImportError``;
* modules are imported at most once and cached.

Usage inside an adapter::

    from windlass.core.lazy import require

    openai = require("openai", extra="openai", feature="The OpenAI provider")
    client = openai.AsyncOpenAI()

Nothing here is exported publicly; adapters are the only callers.
"""

from __future__ import annotations

import importlib
import importlib.util
import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeVar

from windlass.core.exceptions import MissingDependencyError

__all__ = ["LazyModule", "is_available", "lazy_import", "require", "require_attr"]

_T = TypeVar("_T")

_CACHE: dict[str, ModuleType] = {}
_LOCK = threading.RLock()


def is_available(module: str) -> bool:
    """Return whether ``module`` can be imported without importing it.

    Uses :func:`importlib.util.find_spec`, so the module's top-level code does
    not execute. Safe to call on hot paths and at registration time.

    Args:
        module: Dotted module path, e.g. ``"faiss"``.

    Returns:
        True when the module is installed and importable.

    Example:
        >>> is_available("json")
        True
        >>> is_available("definitely_not_installed_xyz")
        False
    """
    if module in _CACHE:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def require(module: str, *, extra: str, feature: str) -> ModuleType:
    """Import an optional module or raise a helpful error.

    Thread-safe and memoised: concurrent first-time callers import exactly once.

    Args:
        module: Dotted module path to import.
        extra: The ``windlass`` extras group that provides it, used to build
            the ``pip install`` hint.
        feature: Human readable feature name for the error message.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: When the module is not installed. The message
            contains the exact install command.

    Example:
        >>> require("json", extra="core", feature="JSON")  # doctest: +ELLIPSIS
        <module 'json' ...>
    """
    cached = _CACHE.get(module)
    if cached is not None:
        return cached
    with _LOCK:
        cached = _CACHE.get(module)
        if cached is not None:
            return cached
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            # Distinguish "the package is missing" from "the package is broken".
            missing_root = module.split(".")[0]
            if _is_missing(exc, missing_root):
                raise MissingDependencyError(
                    feature, extra=extra, package=missing_root, original=exc
                ) from exc
            raise MissingDependencyError(
                f"{feature} failed to import ({exc})",
                extra=extra,
                package=missing_root,
                original=exc,
            ) from exc
        _CACHE[module] = mod
        return mod


def _is_missing(exc: ImportError, root: str) -> bool:
    """Heuristic: did ``exc`` mean 'not installed' rather than 'broken install'?"""
    name = getattr(exc, "name", None) or ""
    return isinstance(exc, ModuleNotFoundError) and (name == root or name.startswith(root))


def require_attr(module: str, attr: str, *, extra: str, feature: str) -> Any:
    """Import an optional module and pull one attribute off it.

    Args:
        module: Dotted module path.
        attr: Attribute (usually a class) to retrieve.
        extra: Extras group providing the module.
        feature: Human readable feature name.

    Returns:
        The requested attribute.

    Raises:
        MissingDependencyError: When the module is absent, or present but does
            not expose ``attr`` (which usually means a version mismatch).
    """
    mod = require(module, extra=extra, feature=feature)
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise MissingDependencyError(
            f"{feature} requires {module}.{attr}, which this installed version does not provide",
            extra=extra,
            package=module.split(".")[0],
            original=exc,
        ) from exc


class LazyModule:
    """A module proxy that defers the import until first attribute access.

    Lets a module hold a reference to an optional dependency at import time
    without paying the import cost or failing when it is absent.

    Args:
        module: Dotted module path.
        extra: Extras group providing it.
        feature: Human readable feature name.

    Example:
        >>> np = LazyModule("json", extra="core", feature="JSON")
        >>> np.dumps([1])          # import happens here
        '[1]'
    """

    __slots__ = ("_extra", "_feature", "_module", "_resolved")

    def __init__(self, module: str, *, extra: str, feature: str) -> None:
        self._module = module
        self._extra = extra
        self._feature = feature
        self._resolved: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._resolved is None:
            self._resolved = require(self._module, extra=self._extra, feature=self._feature)
        return self._resolved

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "loaded" if self._resolved else "deferred"
        return f"<LazyModule {self._module!r} ({state})>"


def lazy_import(path: str) -> Callable[[], Any]:
    """Return a zero-arg callable that imports ``path`` on demand.

    Used by the registry to describe a provider without importing its module.

    Args:
        path: Either ``"pkg.module"`` or ``"pkg.module:Attribute"``.

    Returns:
        A callable resolving to the module or attribute.

    Example:
        >>> loader = lazy_import("json:dumps")
        >>> loader()([1])
        '[1]'
    """

    def _resolve() -> Any:
        module_path, _, attr = path.partition(":")
        module = importlib.import_module(module_path)
        return getattr(module, attr) if attr else module

    return _resolve
