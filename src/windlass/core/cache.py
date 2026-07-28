"""Caching for expensive, repeatable calls.

Embedding the same corpus twice, or re-asking the same question at temperature
0, should not cost twice. Windlass ships two backends behind one interface:

* :class:`MemoryCache` — a thread-safe TTL + LRU dict. No dependencies.
* :class:`DiskCache` — persistent, process-shared, backed by ``diskcache``
  (``pip install "windlass[cache]"``).

Caching is **opt-in**::

    configure(cache={"enabled": True, "ttl": 3600})

Keys are content hashes of the call arguments, so nothing collides across
providers or models.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

from windlass.core.config import CacheConfig, settings
from windlass.core.lazy import require
from windlass.core.logging import get_logger

__all__ = [
    "Cache",
    "DiskCache",
    "MemoryCache",
    "NullCache",
    "build_cache",
    "cached_call",
    "make_key",
]

_T = TypeVar("_T")
_log = get_logger(__name__)
_MISS = object()


def make_key(*parts: Any, namespace: str = "") -> str:
    """Build a deterministic cache key from arbitrary arguments.

    Arguments are JSON-encoded with sorted keys, so dict ordering never changes
    the key. Values that are not JSON-serialisable fall back to ``repr``.

    Args:
        *parts: Anything identifying the call — provider, model, prompt, params.
        namespace: Optional prefix to keep different caches from colliding.

    Returns:
        A ``namespace:hexdigest`` string.

    Example:
        >>> make_key("openai", {"b": 1, "a": 2}) == make_key("openai", {"a": 2, "b": 1})
        True
    """
    payload = json.dumps(parts, sort_keys=True, default=repr, ensure_ascii=False)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    return f"{namespace}:{digest}" if namespace else digest


class Cache(ABC):
    """Key/value cache interface used across Windlass.

    Implementations must be safe to call from multiple threads. Async methods
    default to running the sync ones, which is right for in-process backends;
    a network-backed cache should override them.
    """

    @abstractmethod
    def get(self, key: str) -> Any:
        """Return the cached value, or ``None`` when absent or expired."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL in seconds."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key``. No-op when it is not present."""

    @abstractmethod
    def clear(self) -> None:
        """Drop every entry."""

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    async def aget(self, key: str) -> Any:
        """Async :meth:`get`."""
        return self.get(key)

    async def aset(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Async :meth:`set`."""
        self.set(key, value, ttl)


class NullCache(Cache):
    """A cache that never stores anything.

    Used whenever caching is disabled, so call sites need no ``if cache:``
    branches.

    Example:
        >>> c = NullCache()
        >>> c.set("k", 1)
        >>> c.get("k") is None
        True
    """

    def get(self, key: str) -> Any:
        """Always returns ``None``."""
        return None

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Discards the value."""

    def delete(self, key: str) -> None:
        """No-op."""

    def clear(self) -> None:
        """No-op."""


class MemoryCache(Cache):
    """Thread-safe in-process cache with TTL expiry and LRU eviction.

    Args:
        max_size: Maximum number of live entries. The least recently used entry
            is evicted when the cache is full.
        ttl: Default lifetime in seconds; ``None`` means entries never expire.

    Attributes:
        hits: Number of successful lookups since construction.
        misses: Number of failed lookups since construction.

    Example:
        >>> cache = MemoryCache(max_size=2)
        >>> cache.set("a", 1)
        >>> cache.get("a")
        1
        >>> cache.set("b", 2); cache.set("c", 3)   # evicts "a"
        >>> cache.get("a") is None
        True
    """

    def __init__(self, max_size: int = 1024, ttl: float | None = 3600.0) -> None:
        self.max_size = max(1, max_size)
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self._data: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Any:
        """Return the value for ``key``, or ``None`` when missing or expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            value, expires_at = entry
            if expires_at is not None and expires_at <= time.time():
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store ``value``, evicting the least recently used entry if needed."""
        lifetime = self.ttl if ttl is None else ttl
        expires_at = time.time() + lifetime if lifetime else None
        with self._lock:
            self._data[key] = (value, expires_at)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        """Remove ``key`` if present."""
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        """Drop every entry and reset the counters."""
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def stats(self) -> dict[str, Any]:
        """Return hit/miss counters and the current size.

        Returns:
            A dict with ``hits``, ``misses``, ``size``, ``max_size`` and
            ``hit_rate``.
        """
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self._data),
                "max_size": self.max_size,
                "hit_rate": (self.hits / total) if total else 0.0,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class DiskCache(Cache):
    """Persistent cache backed by the ``diskcache`` package.

    Survives process restarts and is safe to share between processes on one
    machine, which makes it the right choice for ingestion pipelines that are
    re-run during development.

    Args:
        directory: Where to store the cache files. Created if absent.
        ttl: Default lifetime in seconds.
        size_limit: Approximate maximum on-disk size in bytes.

    Raises:
        MissingDependencyError: When ``diskcache`` is not installed.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        ttl: float | None = 3600.0,
        size_limit: int = 1_000_000_000,
    ) -> None:
        diskcache = require("diskcache", extra="cache", feature="The disk cache backend")
        self.directory = Path(directory or settings().cache.directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._cache = diskcache.Cache(str(self.directory), size_limit=size_limit)

    def get(self, key: str) -> Any:
        """Return the stored value, or ``None``."""
        return self._cache.get(key, default=None)

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store ``value`` with an expiry."""
        self._cache.set(key, value, expire=self.ttl if ttl is None else ttl)

    def delete(self, key: str) -> None:
        """Remove ``key``."""
        self._cache.delete(key)

    def clear(self) -> None:
        """Drop every entry."""
        self._cache.clear()

    def close(self) -> None:
        """Release the underlying file handles."""
        self._cache.close()

    def __len__(self) -> int:
        return len(self._cache)


def build_cache(config: CacheConfig | None = None) -> Cache:
    """Construct the cache described by ``config``.

    Falls back to :class:`MemoryCache` (with a warning) if the disk backend is
    requested but ``diskcache`` is missing — a broken cache should never take
    down the application.

    Args:
        config: Cache policy. Defaults to the global configuration.

    Returns:
        A ready-to-use cache. :class:`NullCache` when caching is disabled.

    Example:
        >>> isinstance(build_cache(CacheConfig(enabled=False)), NullCache)
        True
    """
    cfg = config or settings().cache
    if not cfg.enabled:
        return NullCache()
    if cfg.backend == "disk":
        try:
            return DiskCache(cfg.directory, ttl=cfg.ttl)
        except Exception as exc:
            _log.warning("Disk cache unavailable (%s); falling back to memory cache.", exc)
    return MemoryCache(max_size=cfg.max_size, ttl=cfg.ttl)


async def cached_call(
    cache: Cache,
    key: str,
    factory: Callable[[], Awaitable[_T]],
    *,
    ttl: float | None = None,
) -> _T:
    """Return a cached value or compute, store and return it.

    Args:
        cache: The cache to consult.
        key: Precomputed key, usually from :func:`make_key`.
        factory: Zero-argument async callable producing the value on a miss.
        ttl: Override for this entry's lifetime.

    Returns:
        The cached or freshly computed value.

    Example:
        >>> import asyncio
        >>> cache = MemoryCache()
        >>> async def compute(): return 42
        >>> asyncio.run(cached_call(cache, "k", compute))
        42
        >>> asyncio.run(cached_call(cache, "k", compute))   # served from cache
        42
    """
    hit = await cache.aget(key)
    if hit is not None:
        return hit  # type: ignore[return-value]
    value = await factory()
    if value is not None:
        await cache.aset(key, value, ttl)
    return value
